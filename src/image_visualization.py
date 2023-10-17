import os
import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt
from .nerf_func import quaternion_to_Rt


class Visualizer(object):
    """
    Visualize itermediate results, render out depth, color and depth uncertainty images.
    It can be called per iteration, which is good for debugging
    (to see how each tracking/mapping iteration performs)
    """

    def __init__(self,
                 vis_dir,
                 renderer,
                 device='cuda:0'):
        self.device = device
        self.vis_dir = vis_dir
        self.renderer = renderer
        os.makedirs(f'{vis_dir}', exist_ok=True)

    def vis(self,
            idx,
            gt_depth,
            gt_color,
            c2w_or_camera_tensor,
            net):
        """
        Visualization of depth, color images and save to file
        Args:
            idx:                        (int), current frame index
            gt_depth:                   (Tensor), gt depth image of current frame
                                        [H, W]
            gt_color:                   (Tensor), gt color image of current frame
                                        [H, W, 3]
            c2w_or_camera_tensor:       (Tensor), camera pose, represented in camera to wolrd matrix
                                        [7, [4, 4]], or quaternion and translation tensor
            c:                          (dict), feature grids
            decoders:                   (nn.Module), decoders
            writer:                     Tensorboard SummaryWriter

        """
        with torch.no_grad():
            gt_depth_np = gt_depth.cpu().numpy()
            gt_color_np = gt_color.cpu().numpy()
            H, W = gt_depth_np.shape

            if len(c2w_or_camera_tensor.shape) == 1: # quaternion and translation tensor
                c2w = quaternion_to_Rt(c2w_or_camera_tensor.clone().detach())
            else:
                c2w = c2w_or_camera_tensor

            render_out = self.renderer.render_img(
                net, c2w, self.device, gt_depth=gt_depth
            )

            color = render_out['color']
            depth = render_out['depth']
            uncertainty = render_out['depth_variance']
            weight_sum = render_out['weight_sum']  # [n_rays, 1]
            sdf = render_out['sdf']  # [n_pts, 1]
            # - convert normal from world space to camera space
            w2c = torch.inverse(c2w)
            surface_normal = torch.matmul(w2c[None, :3, :3].cpu(), render_out['normal'][:, :, None].cpu())

            depth = depth.reshape(H, W)
            uncertainty = uncertainty.reshape(H, W)
            color = color.reshape(H, W, 3)

            depth_np = depth.detach().cpu().numpy()
            color_np = color.detach().cpu().numpy()
            uncertainty_np = uncertainty.detach().cpu().numpy()
            surface_normal = (surface_normal.numpy().reshape((H, W, 3)) * 128 + 128).clip(0, 255)
            depth_residual = np.abs(gt_depth_np - depth_np)
            depth_residual[gt_depth_np < 1e-3] = 0.0
            color_residual = np.abs(gt_color_np - color_np)
            color_residual[gt_depth_np < 1e-3] = 0.0

            mse = (np.abs(gt_color_np - color_np) ** 2)[gt_depth_np > 1e-3].mean()
            psnr = -10.0 * np.log10(mse)
            mae = (np.abs(gt_depth_np - depth_np))[gt_depth_np > 1e-3].mean()
            rmse = np.sqrt((np.abs(gt_depth_np - depth_np)**2)[gt_depth_np > 1e-3].mean())

            print(f'Idx {idx} Iter {iter}  MAE: {mae:.4f}, PSNR: {psnr:.4f}, '
                  f'S0.01: {(torch.abs(sdf) < 0.01).float().mean():.4f}, S0.02: {(torch.abs(sdf) < 0.02).float().mean():.4f}, ')


            fig, axs = plt.subplots(2, 4)
            fig.tight_layout()
            max_depth = np.max(gt_depth_np)
            min_depth = np.min(gt_depth_np[gt_depth_np>0])
            cmap = plt.cm.get_cmap('plasma')
            # cmap = plt.cm.get_cmap('jet')
            norm = plt.Normalize(vmin=min_depth, vmax=max_depth)

            gt_depth_color = cmap(norm(gt_depth_np))[..., :3]
            depth_color = cmap(norm(depth_np))[..., :3]
            depth_residual_color = depth_err_to_colorbar(depth_np, gt_depth_np, with_bar=False, cmap='jet')[..., :3]

            axs[0, 0].imshow(gt_depth_color)
            axs[0, 0].set_title('GT Depth', fontsize=8)
            axs[0, 0].set_xticks([])
            axs[0, 0].set_yticks([])

            axs[0, 1].imshow(depth_color)
            axs[0, 1].set_title('Predicted Depth', fontsize=8)
            axs[0, 1].set_xticks([])
            axs[0, 1].set_yticks([])

            axs[0, 2].imshow(depth_residual_color)
            axs[0, 2].set_title('Depth Residual', fontsize=8)
            axs[0, 2].set_xticks([])
            axs[0, 2].set_yticks([])

            axs[0, 3].imshow(plt.cm.get_cmap('gray')(uncertainty_np)[..., :3])
            axs[0, 3].set_title('Uncertainty', fontsize=8)
            axs[0, 3].set_xticks([])
            axs[0, 3].set_yticks([])


            axs[1, 0].imshow(gt_color_np.clip(0, 1), cmap='plasma')
            axs[1, 0].set_title('GT RGB', fontsize=8)
            axs[1, 0].set_xticks([])
            axs[1, 0].set_yticks([])

            axs[1, 1].imshow(color_np.clip(0, 1), cmap='plasma')
            axs[1, 1].set_title('Predicted RGB', fontsize=8)
            axs[1, 1].set_xticks([])
            axs[1, 1].set_yticks([])

            axs[1, 2].imshow(color_residual.clip(0, 1), cmap='plasma')
            axs[1, 2].set_title('RGB Residual', fontsize=8)
            axs[1, 2].set_xticks([])
            axs[1, 2].set_yticks([])

            axs[1, 3].imshow((surface_normal / 255.0).clip(0, 1))
            axs[1, 3].set_title('Surface Normal', fontsize=8)
            axs[1, 3].set_xticks([])
            axs[1, 3].set_yticks([])

            plt.subplots_adjust(wspace=0, hspace=0)

            plt.savefig(
                f'{self.vis_dir}/{idx:05d}.jpg', bbox_inches='tight', pad_inches=0.2
            )
            plt.close()

            print(f"INFO: Saved rendering visualization of color/depth image at {self.vis_dir}/{idx:05d}.jpg")


def depth_err_to_colorbar(est, gt=None, with_bar=False, cmap='jet'):
    error_bar_height = 50
    if gt is None:
        gt = np.zeros_like(est)
        valid = est > 0
        max_depth = est.max()
    else:
        valid = gt > 0
        max_depth = gt.max()
    error_map = np.abs(est - gt) * valid
    h, w= error_map.shape

    maxvalue = error_map.max()
    breakpoints = np.array([0,      0.05,      0.2,      0.5,     1.0,    2.0,       max(5.0, maxvalue)])
    points      = np.array([0,      0.25,   0.38,   0.66,  0.83,  0.95,     1])
    num_bins    = np.array([0,      w//8,   w//8,   w//4,  w//4,  w//8,     w - (w//4 + w//4 + w//8 + w//8 + w//8)])
    acc_num_bins = np.cumsum(num_bins)

    for i in range(1, len(breakpoints)):
        scale = points[i] - points[i-1]
        start = points[i-1]
        lower = breakpoints[i-1]
        upper = breakpoints[i]
        error_map = revalue(error_map, lower, upper, start, scale)

    # [0, 1], [H, W, 3]
    error_map = plt.cm.get_cmap(cmap)(error_map)[:, :, :3]

    if not with_bar:
        return error_map

    error_bar = np.array([])
    for i in range(1, len(num_bins)):
        error_bar = np.concatenate((error_bar, np.linspace(points[i-1], points[i], num_bins[i])))

    error_bar = np.repeat(error_bar, error_bar_height).reshape(w, error_bar_height).transpose(1, 0) # [error_bar_height, w]
    error_bar_map = plt.cm.get_cmap(cmap)(error_bar)[:, :, :3]
    plt.xticks(ticks=acc_num_bins, labels=[str(f) for f in breakpoints])
    plt.axis('on')

    # [0, 1], [H, W, 3]
    error_map = np.concatenate((error_map, error_bar_map[..., :3]), axis=0)[..., :3]

    return error_map

def revalue(map, lower, upper, start, scale):
    mask = (map > lower) & (map <= upper)
    if np.sum(mask) >= 1.0:
        mn, mx = map[mask].min(), map[mask].max()
        map[mask] = ((map[mask] - mn) / (mx -mn + 1e-7)) * scale + start

    return map


