import torch
import numpy as np
DIR_NORMALIZE=False


def normalize_3d_coordinate(p, bound):
    """
    Normalize coordinate to [-1, 1], corresponds to the bounding box given
    Args:
        p:                              (Tensor), coordiate in 3d space
                                        [N, 3]
        bound:                          (Tensor), the scene bound
                                        [3, 2]

    Returns:
        p:                              (Tensor), normalized coordiate in 3d space
                                        [N, 3]

    """

    p = p.reshape(-1, 3)
    bound = bound.to(p.device)
    p = (p - bound[:, 0]) / (bound[:, 1] - bound[:, 0]) * 2.0 - 1.0

    return p


def random_select(l, k, start=0):
    """
    Random select k values from 0, 1, ..., l-1
    """
    # return list(np.random.permutation(np.array(range(l))))[:min(l, k)]

    m = (l-start) / k
    idx = np.linspace(start, l-1-m, k) + np.random.rand(k) * m
    idx = idx.clip(start, l-1)
    idx = list(idx)
    idx = [int(i) for i in idx if i > 0]

    return idx



def quad2rotation(quad):
    """
    Convert quaternion to rotation in batch. Since all operation in pytorch, support gradient passing.

    Args:
        quad (tensor, batch_size*4): quaternion.

    Returns:
        rot_mat (tensor, batch_size*3*3): rotation.
    """
    bs = quad.shape[0]
    qr, qi, qj, qk = quad[:, 0], quad[:, 1], quad[:, 2], quad[:, 3]
    two_s = 2.0 / (quad * quad).sum(-1)
    rot_mat = torch.zeros(bs, 3, 3).to(quad.get_device())
    rot_mat[:, 0, 0] = 1 - two_s * (qj ** 2 + qk ** 2)
    rot_mat[:, 0, 1] = two_s * (qi * qj - qk * qr)
    rot_mat[:, 0, 2] = two_s * (qi * qk + qj * qr)
    rot_mat[:, 1, 0] = two_s * (qi * qj + qk * qr)
    rot_mat[:, 1, 1] = 1 - two_s * (qi ** 2 + qk ** 2)
    rot_mat[:, 1, 2] = two_s * (qj * qk - qi * qr)
    rot_mat[:, 2, 0] = two_s * (qi * qk - qj * qr)
    rot_mat[:, 2, 1] = two_s * (qj * qk + qi * qr)
    rot_mat[:, 2, 2] = 1 - two_s * (qi ** 2 + qj ** 2)
    return rot_mat


def Rt_to_quaternion(Rt, Tquad=False):
    """
    Convert transformation matrix to quaternion and translation
    """
    from mathutils import Matrix

    device = Rt.device
    Rt = Rt.detach().cpu().numpy()
    R, t = Rt[:3, :3], Rt[:3, 3]
    rot = Matrix(R)
    quad = rot.to_quaternion()
    if Tquad:
        pose = np.concatenate([t, quad], axis=0)
    else:
        pose = np.concatenate([quad, t], axis=0)

    pose = torch.from_numpy(pose).float().to(device)

    return pose


def quaternion_to_Rt(quadT):
    """
    Convert quaternion and translation to transformation matrix in 4x4
    """
    N = len(quadT.shape)
    device = quadT.device
    if N == 1:
        quadT = quadT.unsqueeze(dim=0)
    quad, t = quadT[:, :4], quadT[:, 4:]
    batch_size, _ = t.shape

    R = quad2rotation(quad)
    Rt = torch.cat([R, t[:, :, None]], dim=2)
    bottom = torch.Tensor([0, 0, 0, 1.0]).reshape(1, 1, 4).to(device)
    bottom = bottom.repeat(batch_size, 1, 1)

    Rt = torch.cat([Rt, bottom], dim=1)

    if N == 1:
        Rt = Rt[0]

    return Rt


def build_rays(H0, H1, W0, W1, n_rays, H, W, fx, fy, cx, cy, c2w, depth, color, device, nerf_coordinate=True, dir_normalize=False, mask=None):
    # !!! do not normalize the dir, otherwise the scale of pose will be deteriorated
    dir_normalize = DIR_NORMALIZE

    depth = depth[H0:H1, W0:W1]
    color = color[H0:H1, W0:W1]

    x, y = torch.meshgrid(
        torch.linspace(W0, W1 - 1, W1 - W0).to(device),
        torch.linspace(H0, H1 - 1, H1 - H0).to(device),
        indexing='ij'
    )
    x, y = x.t(), y.t()  # [W, H] -> [H, W]

    x, y = x.reshape(-1), y.reshape(-1)
    depth = depth.reshape(-1)
    color = color.reshape(-1, 3)
    N = x.shape[0]
    if mask is not None:
        mask = mask[H0:H1, W0:W1]
        mask = mask.reshape(-1).bool()
        indices = torch.arange(N, dtype=torch.long, device=device)
        indices = torch.masked_select(indices, mask)
        x, y = x[indices], y[indices]  # (n_rays, )

        depth = depth[indices]  # (n_rays, )
        color = color[indices]  # (n_rays, 3)

    N = x.shape[0]
    if n_rays > 0 and n_rays < N // 2:
        indices = torch.randint(N, (n_rays, ), device=device)
        indices = indices.clamp(0, N-1)
        x, y = x[indices], y[indices]  # (n_rays, )

        depth = depth[indices]  # (n_rays, )
        color = color[indices]  # (n_rays, 3)

    N = x.shape[0]

    # get rays
    if isinstance(c2w, np.ndarray):
        c2w = torch.from_numpy(c2w).to(device)

    if nerf_coordinate:
        # https://github.com/cvg/nice-slam/issues/10#issuecomment-1128590013
        dirs = torch.stack([
            (x - cx) / fx,
            -(y - cy) / fy,
            -torch.ones_like(x),
        ], dim=-1).to(device)  # (N, 3)
    else:
        dirs = torch.stack([
            (x - cx) / fx,
            (y - cy) / fy,
            torch.ones_like(x),
        ], dim=-1).to(device)  # (N, 3)

    if dir_normalize:
        dirs = dirs / torch.linalg.norm(dirs, ord=2, dim=-1, keepdim=True)  # N, 3
        # comment it if you are sure the pose is fine
        raise TypeError("Ray direction shouldn't be normalized, otherwise the scale of pose will be destroyed!")

    # [c2w @ dirs]^T = dirs^T @ c2w^T
    rays_d = dirs @ c2w[:3, :3].t()  # [N, 3]
    rays_o = c2w[:3, 3].reshape(1, 3).repeat(N, 1)  # [N, 3]

    return rays_o, rays_d, depth, color


def build_all_rays(H, W, fx, fy, cx, cy, c2w, device, nerf_coordinate=True, dir_normalize=False):
    """
    Build rays for a whole image.
    """
    # !!! do not normalize the dir, otherwise the scale of pose will be deteriorated
    dir_normalize = DIR_NORMALIZE
    if isinstance(c2w, np.ndarray):
        c2w = torch.from_numpy(c2w).to(device)

    x, y = torch.meshgrid(
        torch.linspace(0, W-1, W).to(device),
        torch.linspace(0, H-1, H).to(device),
        indexing='ij',
    )
    x, y = x.t(), y.t()  # [W, H] -> [H, W]
    if nerf_coordinate:
        # https://github.com/cvg/nice-slam/issues/10#issuecomment-1128590013
        dirs = torch.stack([
            (x - cx) / fx,
            -(y - cy) / fy,
            -torch.ones_like(x),
            ], dim=-1).to(device)  # (H, W, 3)
    else:
        dirs = torch.stack([
            (x - cx) / fx,
            (y - cy) / fy,
            torch.ones_like(x),
        ], dim=-1).to(device)  # (H, W, 3)

    if dir_normalize:
        dirs = dirs / torch.linalg.norm(dirs, ord=2, dim=-1, keepdim=True)  # H, W, 3
        # comment it if you are sure the pose is fine
        raise TypeError("Ray direction shouldn't be normalized, otherwise the scale of pose will be destroyed!")

    rays_d = dirs @ c2w[:3, :3].t()  # [H, W, 3]
    rays_o = c2w[:3, 3].reshape(1, 1, 3).repeat(H, W, 1)  # [n_rays, 3]

    return rays_o, rays_d


def sample_pdf(bins, weights, N_importance, det=False):
    N_rays, N_samples_ = weights.shape
    # get pdf
    weights = weights + 1e-5  # prevernt nans
    pdf = weights / torch.sum(weights, dim=1, keepdim=True) # (N_rays, N_samples_)
    cdf = torch.cumsum(pdf, dim=1)
    cdf = torch.cat([torch.zeros_like(cdf[..., :1]), cdf], dim=1) # (N_rays, N_samples_+1)
    # take uniform samples
    if det:
        u = torch.linspace(0.0 + 0.5 / N_importance, 1 - 0.5 / N_importance, steps=N_importance)
        u = u.expand(N_rays, N_importance).to(weights.device)
    else:
        u = torch.rand(N_rays, N_importance, device=weights.device)

    # inverse cdf
    u = u.contiguous()
    inds = torch.searchsorted(cdf, u, right=True)
    below = torch.clamp_min(inds-1, 0)
    above = torch.clamp_max(inds, N_samples_)
    inds_g = torch.stack([below, above], dim=-1) # [batch, N_importance, 2]

    matched_shape = [N_rays, N_importance, N_samples_+1]
    cdf_g = torch.gather(cdf.unsqueeze(dim=1).expand(matched_shape), dim=2, index=inds_g)
    bins_g = torch.gather(bins.unsqueeze(dim=1).expand(matched_shape), dim=2, index=inds_g)

    denom = (cdf_g[..., 1] - cdf_g[..., 0])
    denom[denom < 1e-5] = 1.0 # denom equals to 0 means a bin has weight 0
    t = (u - cdf_g[..., 0]) / denom
    samples = bins_g[..., 0] + t * (bins_g[..., 1] - bins_g[..., 0])

    return samples