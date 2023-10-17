import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import tinycudann as tcnn
import mcubes
import trimesh
import numpy as np


def normalized_3d_coordinate(p, bound):
    """
    Normalize coordinate to [-1, 1], corresponds to the bounding box given
    Args:
        p:                              (Tensor), coordiate in 3d space
                                        [N, 3], 3: x, y, z
        bound:                          (Tensor), the scene bound
                                        [3, 2], 3: x, y, z

    Returns:
        p:                              (Tensor), normalized coordiate in 3d space
                                        [N, 3]

    """

    p = p.reshape(-1, 3)
    bound = bound.to(p.device)
    p = (p - bound[:, 0]) / (bound[:, 1] - bound[:, 0]) * 2.0 - 1.0
    p = p.clamp(min=-1.0, max=1.0)

    return p


class Encoding(nn.Module):
    def __init__(self, n_input_dims=3, device='cuda:0', direction:bool=False):
        super(Encoding, self).__init__()
        self.n_input_dims = n_input_dims
        self.device = device
        self.include_xyz = True
        self.direction = direction

        if not direction:
            encoding_config = {
                'otype': 'HashGrid',  # 'otype': 'Grid', 'type': 'Hash'
                'n_levels': 16,
                'n_features_per_level': 2,
                'log2_hashmap_size': 19,
                'base_resolution': 16,
                'per_level_scale': 1.447269237440378,  # "per_level_scale": 2.0,
                'include_xyz': self.include_xyz,
            }
            embed_dim = 3
        else:
            encoding_config = {
                'otype': 'SphericalHarmonics',
                'degree': 4,
            }
            embed_dim = 3

        with torch.cuda.device(device):
            encoding = tcnn.Encoding(n_input_dims=n_input_dims, encoding_config=encoding_config)

        self._B = nn.Parameter(torch.randn(n_input_dims, embed_dim) * torch.Tensor([25.0]))
        self.encoding = encoding
        self.n_output_dims = int(self.include_xyz) * embed_dim + self.encoding.n_output_dims

    def forward(self, x, *args):
        eps = 1e-5

        if self.direction:
            embedded_x = torch.sin(x @ self._B.to(x.device))
            # Expects 3D inputs that represent normalized vectors v transformed into the unit cube as (v+1)/2.
            view_dirs = (embedded_x + 1) / 2
            assert view_dirs.min() >= 0-eps and view_dirs.max() <= 1+eps, f'dir value range ' \
                                                                  f'[{view_dirs.min().item(), view_dirs.max().item()}]!'
            out = self.encoding(view_dirs, *args)
        else:
            # assuming the x range within [-1, 1]
            embedded_x = x
            # Expects 3D inputs that represent normalized vectors v transformed into the unit cube as (v+1)/2.
            view_pts = (x + 1) / 2
            assert view_pts.min() >= 0-eps and view_pts.max() <= 1+eps, f'3d points value range ' \
                                                                          f'[{view_pts.min().item(), view_pts.max().item()}]!'

            out = self.encoding(view_pts, *args)

        if self.include_xyz:
            out = torch.cat([
                    embedded_x,
                    out,
                ], dim=-1)

        return out


class SDFNetwork(nn.Module):
    def __init__(self,
                 d_in: int = 3,
                 d_out: int = 32,
                 device: str ='cuda:0'):
        super(SDFNetwork, self).__init__()
        self.d_in = d_in
        self.d_out = d_out
        self.device = device

        self.encoding = Encoding(n_input_dims=d_in, device=device, direction=False)
        self.sdf_layer = nn.Linear(self.encoding.n_output_dims, d_out)
        torch.nn.init.constant_(self.sdf_layer.bias, 0.0)
        torch.nn.init.constant_(self.sdf_layer.weight[:, 3:], 0.0)
        torch.nn.init.normal_(self.sdf_layer.weight[:, :3], mean=0.0, std=math.sqrt(2)/math.sqrt(d_out))

    def get_training_parameters(self, ignore_keys=()):
        params = {
            'network': list(self.sdf_layer.parameters()) + [self.encoding._B, ],
            'volume': list(self.encoding.encoding.parameters())
        }

        return params

    def forward(self, pts, bound=None):
        n_pts, _ = pts.shape

        if bound is not None:
            pts = normalized_3d_coordinate(pts, bound.to(pts.device))
        # pts should be range in [-1, 1]
        pts = self.encoding(pts)
        out = self.sdf_layer(pts)
        sdf, feat = out[:, 0:1], out[:, 1:]

        return sdf, feat

    def sdf(self, pts, bound=None, require_feature=False, require_gradient=False):
        if require_gradient:
            with torch.enable_grad():
                pts.requires_grad_(True) # [n_3dpoints, d_in]
                sdf, feat = self.forward(pts, bound)
                d_output = torch.ones_like(sdf, requires_grad=False, device=sdf.device)
                # ! Distributed Data Parallel doesn't work with torch.autograd.grad()
                # ! (i.e., it will only work if gradients are to be accumulated in x.grad attributes of parameters)
                gradient = torch.autograd.grad(
                    outputs=sdf,
                    inputs=pts,
                    grad_outputs=d_output,
                    create_graph=True,
                    retain_graph=True,
                    only_inputs=True
                )[0]  # [n_3dpoints, d_in]

            if require_feature:
                return sdf, feat, gradient
            else:
                return sdf, gradient
        else:
            sdf, feat = self.forward(pts, bound)
            if require_feature:
                return sdf, feat
            else:
                return sdf


class ColorNetwork(nn.Module):
    def __init__(self,
                 d_in: int = 3,
                 d_feat: int = 32 - 1,
                 d_hidden=64,
                 n_layers=2,
                 device='cuda:0'):
        super(ColorNetwork, self).__init__()
        self.d_in = d_in
        self.d_feat = d_feat
        self.d_hidden = d_hidden
        self.n_layers = n_layers
        self.device = device

        embed_dim = 33
        self._B = nn.Parameter(torch.randn(3, embed_dim) * torch.Tensor([25.0]))

        # self.dir_encoding = Encoding(n_input_dims=d_in, device=self.device, direction=True)
        # n_input_dims = embed_dim + self.dir_encoding.n_output_dims + 1 + 3 + d_feat
        n_input_dims = embed_dim + 3 + d_feat

        with torch.cuda.device(device):
            network_config = {
                'otype': 'FullyFusedMLP',
                'activation': 'ReLU',
                'output_activation': 'none',
                'n_neurons': d_hidden,
                'n_hidden_layers': n_layers,
            }

            self.network = tcnn.Network(n_input_dims=n_input_dims, n_output_dims=3, network_config=network_config)

    def forward(self, view_pts, view_dirs, sdf, normals, feature_vectors):
        # view_dirs = self.dir_encoding(view_dirs)
        view_pts = torch.sin(view_pts @ self._B.to(view_pts.device))

        # refer to https://arxiv.org/abs/2003.09852
        rendering_input = torch.cat([view_pts, normals, feature_vectors], dim=1)

        x = self.network(rendering_input)

        x = torch.sigmoid(x)

        return x


class SingleVarianceNetwork(nn.Module):
    def __init__(self, init_val=0.2, scale_factor=10.0):
        super(SingleVarianceNetwork, self).__init__()
        self.scale_factor = scale_factor
        self.register_parameter('variance', nn.Parameter(torch.tensor(init_val)))

    def forward(self, x):
        B, _ = x.shape
        return torch.ones(size=[B, 1]).to(x.device) * torch.exp(self.variance * self.scale_factor)


class InstantNeuS(nn.Module):
    def __init__(self, cfg, bound, device='cuda:0'):
        super(InstantNeuS, self).__init__()
        self.cfg = cfg
        self.register_buffer('bound', torch.tensor(bound).float())  # [3, 2]
        self.register_buffer('realtime_bound', torch.tensor(bound).float())  # [3, 2]
        self.device = device

        self.sdf_network = SDFNetwork(**cfg['sdf_network'], device=device)
        self.color_network = ColorNetwork(**cfg['color_network'], device=device)
        self.variance_network = SingleVarianceNetwork(**cfg['variance_network'])
        self.sdf_smooth_std = cfg['sdf_smooth_std']
        self.sdf_sparse_factor = cfg['sdf_sparse_factor']
        self.sdf_truncation = cfg['sdf_truncation']
        self.sdf_random_weight = cfg['sdf_random_weight']
        self.cos_anneal_ratio = 1.0

    def get_training_parameters(self, ignore_keys=()):
        params = []
        all_params = {
            'sdf_network': list(self.sdf_network.get_training_parameters()['network']),
            'color_network': list(self.color_network.parameters()),
            'variance_network': list(self.variance_network.parameters()),
        }
        for k, v in all_params.items():
            if k not in ignore_keys:
                params += v

        return params

    def get_volume_parameters(self):
        params = list(self.sdf_network.get_training_parameters()['volume'])

        return params

    @torch.no_grad()
    def update_bound(self, bound):
        self.realtime_bound[:] = bound.float().to(self.realtime_bound.device)

    @torch.no_grad()
    def in_bound(self, pts, bound):
        """
        Args:
            pts:                        (Tensor), 3d points
                                        [n_points, 3]
            bound:                      (Tensor), bound
                                        [3, 2]
        """
        # mask for points out of bound
        bound = bound.to(pts.device)
        mask_x = (pts[:, 0] < bound[0, 1]) & (pts[:, 0] > bound[0, 0])
        mask_y = (pts[:, 1] < bound[1, 1]) & (pts[:, 1] > bound[1, 0])
        mask_z = (pts[:, 2] < bound[2, 1]) & (pts[:, 2] > bound[2, 0])
        mask = (mask_x & mask_y & mask_z).bool()

        return mask

    def get_alpha(self, sdf, gradients, dirs, dists):
        n_pts, _ = sdf.shape
        device = sdf.device
        inv_s = self.variance_network(torch.ones([n_pts, 1]).to(device)).clip(1e-6, 1e6)

        true_cos = (dirs * gradients).sum(dim=1, keepdim=True)  # v * n, [n_rays*n_samples, 1]
        # "cos_anneal_ratio" grows from 0 to 1 in the beginning training iterations. The anneal strategy below makes
        # the cos value "not dead" at the beginning training iterations, for better convergence.
        iter_cos = -(F.relu(-true_cos * 0.5 + 0.5) * (1.0 - self.cos_anneal_ratio) +
                     F.relu(-true_cos) * self.cos_anneal_ratio)  # always non-negative

        est_next_sdf = sdf + iter_cos * dists.reshape(-1, 1) / 2.0
        est_prev_sdf = sdf - iter_cos * dists.reshape(-1, 1) / 2.0
        prev_cdf = torch.sigmoid(est_prev_sdf * inv_s)
        next_cdf = torch.sigmoid(est_next_sdf * inv_s)
        alpha = ((prev_cdf - next_cdf + 1e-5) / (prev_cdf + 1e-5)).clip(0.0, 1.0)

        return alpha

    def forward(self,
                rays_o,
                rays_d,
                z_vals,
                dists,
                render_params: dict = None):
        
        n_rays, n_samples = z_vals.shape
        device = z_vals.device

        z_vals = z_vals + dists / 2.0
        pts = rays_o[:, None, :] + rays_d[:, None, :] * z_vals[:, :, None]
        dirs = rays_d[:, None, :].expand(n_rays, n_samples, 3)
        pts = pts.reshape(-1, 3)
        dirs = dirs.reshape(-1, 3)
        pts_mask = self.in_bound(pts, self.realtime_bound)
        if torch.sum(pts_mask.float()) < 1:  # it may happen when render out image
            pts_mask[:100] = True

        def alpha_rgb_fn(pts_3d, dirs_3d, dists_3d, mask):
            # [n_pts, 1], [n_pts, 3]
            n_pts, _ = pts_3d.shape
            out_sdf, out_feat, out_gradients = self.sdf_network.sdf(pts_3d[mask], self.bound,
                                                                    require_feature=True,
                                                                    require_gradient=True)
            sdf = torch.ones([n_pts, 1]).to(device) * 100
            feat = torch.zeros([n_pts, out_feat.shape[1]]).to(device).to(out_feat.dtype)
            gradients = torch.zeros([n_pts, 3]).to(device).to(out_gradients.dtype)

            sdf[mask] = out_sdf
            feat[mask] = out_feat
            gradients[mask] = out_gradients

            alpha = self.get_alpha(sdf, gradients, dirs_3d, dists_3d)  # [n_pts, 1]
            out_rgb = self.color_network(pts_3d[mask], dirs_3d[mask], sdf[mask], gradients[mask], feat[mask])  # [n_pts, 3]

            rgb = torch.zeros([n_pts, 3]).to(device).to(out_rgb.dtype)
            rgb[mask] = out_rgb

            return sdf, rgb, alpha, gradients

        sdf, rgb, alpha, gradients = alpha_rgb_fn(pts, dirs, dists, pts_mask)
        sdf = sdf.reshape(n_rays, n_samples)
        rgb = rgb.reshape(n_rays, n_samples, 3)
        alpha = (alpha * pts_mask[:, None]).reshape(n_rays, n_samples)
        gradients = gradients.reshape(n_rays, n_samples, 3)
        pts_mask = pts_mask.reshape(n_rays, n_samples)

        weights = alpha * torch.cumprod(torch.cat([
            torch.ones([n_rays, 1]).to(device),
            1 - alpha + 1e-7,
        ], dim=1), dim=1)[:, :-1]  # [n_rays, n_samples]
        weight_sum = weights.sum(dim=1, keepdim=True)  # [n_rays, 1]
        rgb = (rgb * weights[:, :, None]).sum(dim=1, keepdim=False)

        depth = (z_vals * weights).sum(dim=1, keepdim=True)  # [n_rays, 1]
        depth_vars = ((z_vals - depth) ** 2 * weights[:, :n_samples]).sum(dim=1, keepdim=True)  # [n_rays, 1]
        normals = gradients * weights[:, :, None]
        normals = (normals * pts_mask[:, :, None]).sum(dim=1, keepdim=False)

        # Eikonal loss
        gradient_error = (torch.linalg.norm(gradients, ord=2, dim=2, keepdim=False) - 1.0) ** 2  # [n_pts,]
        gradient_error = gradient_error * pts_mask
        gradient_error = gradient_error.mean().unsqueeze(dim=0)  # [1, ]

        return {
            'color': rgb,  # [n_rays, 3]
            'depth': depth,  # [n_rays, 1]
            'depth_variance': depth_vars,  # [n_rays, 1]
            'normal': normals,  # [n_rays, 3]
            'weight_sum': weight_sum,  # [n_rays, 1]
            'sdf_variance':  (1.0 / self.variance_network(torch.ones_like(depth))),  # [n_rays, 1]
            'sdf': sdf, # [n_rays, n_samples]
            'z_vals': z_vals,  # [n_pts, n_samples]
            'gradient_error': gradient_error,  # [1, ]
        }

    def compute_sdf_error(self, sdf, z_vals, gt_depth):
        N_rays, N_surface = z_vals.shape

        pred_sdf = sdf.reshape(N_rays, N_surface)  # [n_rays, n_surface]
        truncation = self.sdf_truncation
        gt_depth = gt_depth.reshape(N_rays, 1)
        valid_mask = (gt_depth > 0).reshape(-1)
        gt_depth = gt_depth[valid_mask]
        z_vals = z_vals[valid_mask]
        pred_sdf = pred_sdf[valid_mask]

        front_mask = z_vals < (gt_depth - truncation)  # [n_rays, n_surface]
        bound = (gt_depth - z_vals)
        sdf_mask = bound.abs() <= truncation  # [n_rays, n_surface]

        n_valid_samples = front_mask.sum(dim=1, keepdim=False) + \
                          sdf_mask.sum(dim=1, keepdim=False) + 1e-8  # [n_rays,]
        n_valid_rays = valid_mask.sum()

        # refer to https://arxiv.org/pdf/2204.02296v2.pdf Eq(6)
        front_loss = (torch.max(
            torch.exp((-self.sdf_sparse_factor * pred_sdf).clamp(max=10.0)) - torch.ones_like(pred_sdf),
            pred_sdf - bound
        ).clamp(min=0.0)) * front_mask
        sdf_front_error = (front_loss.sum(dim=1, keepdim=False) / n_valid_samples).sum() / n_valid_rays
        sdf_error = torch.abs(pred_sdf - bound)
        sdf_error = ((sdf_error * sdf_mask).sum(dim=1, keepdim=False) / n_valid_samples).sum() / n_valid_rays

        return sdf_error, sdf_front_error

    @torch.no_grad()
    def extract_color(self, bound, vertices):
        N = int(64 * 64 * 64)
        rgbs = []
        points = torch.from_numpy(vertices).float().to(bound.device)
        for i, pts in enumerate(torch.split(points, N, dim=0)):
            sdf, feat, gradident = self.sdf_network.sdf(pts,
                                                        bound=bound,
                                                        require_feature=True,
                                                        require_gradient=True)
            out_rgb = self.color_network(pts, None, sdf, gradident, feat)
            rgbs.append(out_rgb.float())

        rgbs = torch.cat(rgbs, dim=0)
        vertex_colors = rgbs.detach().cpu().numpy()
        vertex_colors = np.clip(vertex_colors, 0, 1) * 255
        vertex_colors = vertex_colors.astype(np.uint8)

        return vertex_colors

    @torch.no_grad()
    def extract_fields(self, bound_min: torch.Tensor, bound_max: torch.Tensor, resolution: int):
        N = 64
        X = torch.linspace(bound_min[0], bound_max[0], resolution).split(N)
        Y = torch.linspace(bound_min[1], bound_max[1], resolution).split(N)
        Z = torch.linspace(bound_min[2], bound_max[2], resolution).split(N)

        u = np.zeros([resolution, resolution, resolution], dtype=np.float32)
        bound = torch.cat([bound_min[:, None], bound_max[:, None]], dim=1)

        with torch.no_grad():
            for xi, xs in enumerate(X):
                for yi, ys in enumerate(Y):
                    for zi, zs in enumerate(Z):
                        xx, yy, zz = torch.meshgrid(xs, ys, zs, indexing='ij')
                        pts = torch.cat([
                            xx.reshape(-1, 1),
                            yy.reshape(-1, 1),
                            zz.reshape(-1, 1),
                        ], dim=1).to(bound.device)
                        mask = self.in_bound(pts, self.realtime_bound)
                        n_pts, _ = pts.shape
                        device = pts.device
                        sdf = torch.ones([n_pts, 1]).to(device) * 100
                        if mask.sum() > 0:
                            val = self.sdf_network.sdf(pts[mask],
                                                       bound=bound,
                                                       require_feature=False,
                                                       require_gradient=False)
                            sdf[mask] = val
                        sdf = sdf.reshape(len(xs), len(ys), len(zs)).detach().cpu().numpy()
                        u[xi * N:xi * N + len(xs), yi * N:yi * N + len(ys), zi * N:zi * N + len(zs)] = -sdf

        return u

    @torch.no_grad()
    def extract_geometry(self,
                         resolution: int,
                         threshold: float,
                         c2w_ref: None,
                         save_path='./mesh.ply',
                         color=False):
        bound_min = self.bound[:, 0]
        bound_max = self.bound[:, 1]

        u = self.extract_fields(bound_min, bound_max, resolution)
        b_max_np = bound_max.detach().cpu().numpy()
        b_min_np = bound_min.detach().cpu().numpy()

        vertices, triangles = mcubes.marching_cubes(u, threshold)
        # vertices, triangles = go_mcubes.marching_cubes(-u, threshold, 3.0)
        vertices = vertices / (resolution - 1.0) * (b_max_np - b_min_np)[None, :] + b_min_np[None, :]

        if c2w_ref is not None:
            c2w_ref = c2w_ref.cpu().numpy()  # [4, 4]
            vertices_homo = np.concatenate([vertices, np.ones_like(vertices[:, :1])], axis=1)
            # [1, 4, 4] @ [n_pts, 4, 1] = [n_pts, 4, 1]
            vertices = np.matmul(c2w_ref[None, :, :], vertices_homo[:, :, None])[:, :3, 0]

        vertex_colors = None
        if color:
            vertex_colors = self.extract_color(bound=self.bound.clone(), vertices=vertices)
        mesh = trimesh.Trimesh(vertices, triangles, vertex_colors=vertex_colors)

        eps = 0.01
        bound = self.realtime_bound.detach().cpu().numpy()
        vertices = mesh.vertices[:, :3]
        bound_mask = np.all(vertices >= (bound[:, 0] - eps), axis=1) & np.all(vertices <= (bound[:, 1] + eps), axis=1)
        face_mask = bound_mask[mesh.faces].all(axis=1)
        mesh.update_faces(face_mask)
        mesh.remove_unreferenced_vertices()

        if save_path is not None:
            mesh.export(save_path)

        return mesh


