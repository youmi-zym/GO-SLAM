import torch

from .nerf_func import build_all_rays


class Renderer(object):
    def __init__(self, cfg, args, slam, points_batch_size=1e4, ray_batch_size=5e3):
        """
        Mesher class, given a scene representation, the mesher extracts the mesh from it.
        Args:
            cfg:                        (dict), parsed config dict
            args:                       (class 'argparse.Namespace'), argparse arguments
            slam:                       (class NICE-SLAM), NICE-SLAM main class
            points_batch_size:          (int), maximum points size for query in one batch
                                        Used to alleviate GPU memory usage. Defaults to 5e5
            ray_batch_size:             (int), maximum ray size for query in one batch
                                        Used to alleviate GPU memory usage. Defaults to 1e5
        """
        self.ray_batch_size = int(ray_batch_size)
        self.points_batch_size = int(points_batch_size)

        self.lindisp = cfg['rendering']['lindisp']
        self.perturb = cfg['rendering']['perturb']
        self.N_samples = cfg['rendering']['N_samples']
        self.N_surface = cfg['rendering']['N_surface']

        self.H, self.W, self.fx, self.fy, self.cx, self.cy = slam.H, slam.W, slam.fx, slam.fy, slam.cx, slam.cy

    def eval_points(self,
                    rays_o,
                    rays_d,
                    z_vals,
                    dists,
                    net,
                    render_params):
        """
        Evaluate the occupancy and/or color value for the points
        Args:
            rays_o:                     (Tensor), the origin of each ray
                                        [N_rays, 3]
            rays_d:                     (Tensor), the direction of each ray
                                        [N_rays, 3]
            z_vals:                     (Tensor),
                                        [N_rays, N_samples]
            dists:                      (Tensor)
                                        [N_rays, N_samples]
            net:                        (nn.Module), network

        Returns:
            ret:                        (Tensor), occupancy (and color) value of input points

        """
        rays_o_split = torch.split(rays_o, self.points_batch_size)
        rays_d_split = torch.split(rays_d, self.points_batch_size)
        z_vals_split = torch.split(z_vals, self.points_batch_size)
        dists_split = torch.split(dists, self.points_batch_size)
        render_out = {}
        for ro, rd, zv, dst in zip(rays_o_split, rays_d_split, z_vals_split, dists_split):
            out = net(ro, rd, zv, dst, render_params=render_params)
            if len(render_out) == 0:
                render_out = out
                continue
            for k, v in out.items():
                if torch.is_tensor(v):
                    render_out[k] = torch.cat([render_out[k], v], dim=0)
                else:
                    assert v is None, type(v)
                    render_out[k] = v


        return render_out

    def render_batch_ray(self,
                         rays_o,
                         rays_d,
                         net,
                         render_params,
                         device='cuda:0',
                         gt_depth=None):
        """
        Render color, depth and uncertainty of a batch of rays.
        Args:
            c:                          (dict), feature grids
            net:                        (nn.Module), network.
            rays_d:                     (Tensor), rays direction.
                                        [N_rays, 3]
            rays_o:                     (Tensor), rays origin.
                                        [N_rays, 3]
            device:                     (str), device name to compute on.
            gt_depth:                   (Tensor, optional), sensor depth image. Defaults to None.
                                        [N_rays, ]

        Returns:
            depth:                      (Tensor), rendered depth.
            uncertainty:                (Tensor), rendered uncertainty.
            color:                      (Tensor), rendered color.

        """
        N_samples = self.N_samples
        N_surface = self.N_surface

        N_rays, _ = rays_o.shape

        if gt_depth is None:
            N_surface = 0
            near = 0.01
        else:
            gt_depth = gt_depth.reshape(-1, 1)
            gt_depth_samples = gt_depth.repeat(1, N_samples)  # [n_rays, n_samples]
            near = gt_depth_samples * 0.01

        with torch.no_grad():
            det_rays_o = rays_o.clone().detach()[:, :, None]  # [n_rays, 3, 1]
            det_rays_d = rays_d.clone().detach()[:, :, None]  # [n_rays, 3, 1]
            # TODO: check the bound should be real or enlarged
            t = (net.bound[None, :, :].to(device) - det_rays_o) / det_rays_d  # [n_rays, 3, 2]
            far_bb, _ = torch.min(torch.max(t, dim=2)[0], dim=1)  # [n_rays, ]
            far_bb = far_bb[:, None] + 0.01

        if gt_depth is not None:
            # in case the bound is too large
            far = torch.clamp(far_bb, 0, (gt_depth * 1.2).max())
        else:
            far = far_bb

        z_vals_surface = None
        if N_surface > 0:
            # since we want to colorize even on regions with no depth sensor readings,
            # meaning colorize on interpolated geometry region,
            # we sample all pixels (not using depth mask) for color loss.
            # Therefore, for pixels with non-zero depth value, we sample near the surface,
            # since it is not a good idea to sample 16 points near (half even behind) camera,
            # for pixels with zero depth value, we sample uniformly from camera to max_depth.
            valid_mask = gt_depth > 0  # [n_rays, 1]
            valid_depth = (gt_depth * valid_mask).repeat(1, N_surface)
            t_vals_surface = torch.linspace(0, 1, steps=N_surface).float().to(device)
            t_vals_surface = t_vals_surface[None, :].repeat(N_rays, 1)
            perct = 0.1
            snr, sfar = (1 - perct) * valid_depth, (1 + perct) * valid_depth
            valid_z_vals_surface = snr + (sfar - snr) * t_vals_surface  # not self.lindisp
            invalid_z_vals_surface = 0.001 + (gt_depth.max() - 0.001) * t_vals_surface  # not self.lindisp
            z_vals_surface = valid_z_vals_surface * valid_mask + invalid_z_vals_surface * (1 - valid_mask.float())

        t_vals = torch.linspace(0, 1, steps=N_samples, device=device)
        t_vals = t_vals[None, :].repeat(N_rays, 1)

        if not self.lindisp:
            z_vals = near + (far - near) * t_vals
            sample_dist = ((far - near) / N_samples).mean(dim=1, keepdim=True)  # [n_rays, 1]
        else:
            z_vals = 1.0 / (1.0 / far + (1.0 / near - 1.0 / far) * t_vals)
            sample_dist = 1.0 / ((1.0 / near - 1.0 / far) / N_samples).mean(dim=1, keepdim=True)  # [n_rays, 1]

        if self.perturb > 0:
            z_vals_mid = 0.5 * (z_vals[:, :-1] + z_vals[:, 1:])  # [n_rays, n_samples-1]
            upper = torch.cat([z_vals_mid, z_vals[:, -1:]], dim=1)
            lower = torch.cat([z_vals[:, :1], z_vals_mid], dim=1)
            # stratified samples in those intervals
            perturb_rand = torch.rand(N_samples, device=device)
            z_vals = lower + (upper - lower) * perturb_rand

        if N_surface > 0:
            z_vals, _ = torch.sort(torch.cat([
                z_vals,
                z_vals_surface.float(),
            ], dim=1), dim=1)

        dists = torch.cat([
            z_vals[..., 1:] - z_vals[..., :-1],
            sample_dist,
        ], dim=-1)

        rets = self.eval_points(rays_o, rays_d, z_vals, dists, net, render_params=render_params)

        return rets

    def render_img(self, net, c2w, device, gt_depth=None):
        """
        Renders out depth, uncertainty, and color image
        Args:
            net:                        (nn.Module), network
            c2w:                        (Tensor), camera to world matrix of current frame
            device:                     (str), GPU device
            gt_depth:                   (Tensor, optional), depth image
                                        [H, W]

        Returns:
            depth:                      (Tensor), rendered depth image
                                        [H, W]
            uncertainty:                (Tensor), rendered uncertainty image
                                        [H, W]
            color:                      (Tensor), rendered color image
                                        [H, W]

        """
        with torch.no_grad():
            H = self.H
            W = self.W
            rays_o, rays_d = build_all_rays(H, W, self.fx, self.fy, self.cx, self.cy, c2w, device,
                                            nerf_coordinate=False, dir_normalize=False)
            rays_o = rays_o.reshape(-1, 3)
            rays_d = rays_d.reshape(-1, 3)
            K = torch.eye(4).to(device)
            K[0, 0], K[0, 2], K[1, 1], K[1, 2] = self.fx, self.cx, self.fy, self.cy

            render_params = {
                'global_step': -1,
                'gt_depth': None,
                'stratified': False,
                'update_state': False,
                'compute_sdf_smooth_error': False,
            }

            render_out = {}
            ray_batch_size = self.ray_batch_size
            gt_depth = gt_depth.reshape(-1)
            for i in range(0, H*W, ray_batch_size):
                rays_d_batch = rays_d[i:i+ray_batch_size]
                rays_o_batch = rays_o[i:i+ray_batch_size]
                if gt_depth is not None:
                    gt_depth_batch = gt_depth[i:i+ray_batch_size]

                out = self.render_batch_ray(rays_o=rays_o_batch, rays_d=rays_d_batch, net=net,
                                            render_params=render_params, device=device, gt_depth=gt_depth_batch)
                if len(render_out) == 0:
                    render_out = out
                    continue
                for k, v in out.items():
                    if torch.is_tensor(v):
                        render_out[k] = torch.cat([render_out[k], v], dim=0)
                    else:
                        assert v is None, type(v)
                        render_out[k] = v

            return render_out

