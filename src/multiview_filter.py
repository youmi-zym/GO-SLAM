import torch
import torch.nn as nn
import droid_backends
from lietorch import SE3
from colorama import Fore, Style
import torch.nn.functional as F


class MultiviewFilter(nn.Module):
    def __init__(self, cfg, args, slam):
        super(MultiviewFilter, self).__init__()

        self.args = args
        self.cfg = cfg
        self.device = args.device
        self.warmup = cfg['tracking']['warmup']
        self.filter_thresh = cfg['tracking']['multiview_filter']['thresh']  # dpeth error < 0.01m
        self.filter_visible_num = cfg['tracking']['multiview_filter']['visible_num']  # points viewed by at least 3 cameras
        self.kernel_size = cfg['tracking']['multiview_filter']['kernel_size']  # 3
        self.bound_enlarge_scale = cfg['tracking']['multiview_filter']['bound_enlarge_scale']
        self.net = slam.net
        self.video = slam.video
        self.verbose = slam.verbose
        self.mode = slam.mode

        self.H, self.W, self.fx, self.fy, self.cx, self.cy = slam.H, slam.W, slam.fx, slam.fy, slam.cx, slam.cy

    def pose_dist(self, Tquad0, Tquad1):
        # Tquad with shape [batch_size, 7]
        def quat_to_euler(Tquad):
            """
            Convert a quaternion into euler angles (roll, pitch, yaw)
            roll is rotation around x in radians (counterclockwise)
            pitch is rotation around y in radians (counterclockwise)
            yaw is rotation around z in radians (counterclockwise)
            """
            tx, ty, tz, x, y, z, w = torch.unbind(Tquad, dim=-1)
            t0 = 2.0 * (w * x + y * z)
            t1 = 1.0 - 2.0 * (x * x + y * y)
            roll_x = torch.atan2(t0, t1)

            t2 = 2.0 * (w * y - z * x)
            t2 = torch.clamp(t2, min=-1.0, max=1.0)
            pitch_y = torch.asin(t2)

            t3 = 2.0 * (w * z + x * y)
            t4 = 1.0 - 2.0 * (y * y + z * z)
            yaw_z = torch.atan2(t3, t4)

            Teuler = torch.stack([tx, ty, tz, roll_x, pitch_y, yaw_z], dim=-1)

            return Teuler

        # Refer to BundleFusion Sec5.3
        Teuler0 = quat_to_euler(Tquad0)
        Teuler1 = quat_to_euler(Tquad1)
        dist = (Teuler0 - Teuler1).abs()
        # [batch_size, ]
        dist = 1.0 * dist[:, :3].sum(dim=-1) + 2.0 * dist[:, 3:].sum(dim=-1)

        return dist

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

    @torch.no_grad()
    def get_bound_from_pointcloud(self, pts, enlarge_scale=1.0):
        bound = torch.stack([
            torch.min(pts, dim=0, keepdim=False).values,
            torch.max(pts, dim=0, keepdim=False).values,
        ], dim=-1)  # [3, 2]
        enlarge_bound_length = (bound[:, 1] - bound[:, 0]) * (enlarge_scale - 1.0)
        # extend max 1.0m on boundary
        # enlarge_bound_length = torch.min(enlarge_bound_length, torch.ones_like(enlarge_bound_length) * 1.0)
        bound_edge = torch.stack([
            -enlarge_bound_length / 2.0,
            enlarge_bound_length / 2.0,
        ], dim=-1)
        bound = bound + bound_edge

        return bound

    def forward(self):
        cur_t = self.video.counter.value
        filtered_t = int(self.video.filtered_id.item())
        if filtered_t < cur_t and cur_t > self.warmup:
            with self.video.get_lock():
                dirty_index = torch.arange(0, cur_t).long().to(self.device)
                poses = torch.index_select(self.video.poses.detach(), dim=0, index=dirty_index)
                disps = torch.index_select(self.video.disps_up.detach(), dim=0, index=dirty_index)
                common_intrinsic_id = 0  # we assume the intrinsics are the same within one scene
                intrinsic = self.video.intrinsics[common_intrinsic_id].detach() * self.video.scale_factor
                w2w = SE3(self.video.pose_compensate[0].clone().unsqueeze(dim=0)).to(self.device)

            points = droid_backends.iproj((w2w * SE3(poses).inv()).data, disps, intrinsic).cpu() # [b, h, w 3]
            thresh = self.filter_thresh * torch.ones_like(disps.mean(dim=[1, 2]))
            count = droid_backends.depth_filter(
                poses, disps, intrinsic, dirty_index, thresh
            )  # [b, h, w]

            count = count.cpu()
            disps = disps.cpu()

            masks = (count >= self.filter_visible_num)
            masks = masks & (disps > 0.01 * disps.mean(dim=[1, 2], keepdim=True))  # filter out far points, [b, h, w]
            if masks.sum() < 100:
                return
            sel_points = points.reshape(-1, 3)[masks.reshape(-1)]
            bound = self.get_bound_from_pointcloud(sel_points) # [3, 2]

            if isinstance(self.kernel_size, str) and self.kernel_size == 'inf':
                extended_masks = torch.ones_like(masks).bool()
            elif int(self.kernel_size) < 2:
                extended_masks = masks
            else:
                kernel = int(self.kernel_size)
                kernel = (kernel // 2) * 2 + 1  # odd number

                extended_masks = F.conv2d(
                    masks.unsqueeze(dim=1).float(),
                    weight=torch.ones(1, 1, kernel, kernel, dtype=torch.float, device=masks.device),
                    stride=1,
                    padding=kernel//2,
                    bias=None,
                ).bool().squeeze(dim=1)  # [b, h, w]

            if extended_masks.sum() < 100:
                return
            sel_points = points.reshape(-1, 3)[extended_masks.reshape(-1)]
            in_bound_mask = self.in_bound(sel_points, bound)  # N'
            extended_masks[extended_masks.clone()] = in_bound_mask

            sel_points = points.reshape(-1, 3)[extended_masks.reshape(-1)]
            bound = self.get_bound_from_pointcloud(sel_points) # [3, 2]

            priority = self.pose_dist(self.video.poses_filtered[:cur_t].detach(), poses)

            with self.video.mapping.get_lock():
                self.video.update_priority[:cur_t] += priority.detach()
                self.video.mask_filtered[:cur_t] = extended_masks.detach()
                self.video.disps_filtered[:cur_t] = disps.detach()
                self.video.poses_filtered[:cur_t] = poses.detach()
                self.video.filtered_id[0] = cur_t
                self.video.bound[0] = bound

            prefix = "Bound: ["
            bd = bound.tolist()
            prefix += f'[{bd[0][0]:.1f}, {bd[0][1]:.1f}], '
            prefix += f'[{bd[1][0]:.1f}, {bd[1][1]:.1f}], '
            prefix += f'[{bd[2][0]:.1f}, {bd[2][1]:.1f}]]!'
            print(Fore.CYAN)
            print(f'\n\n Multiview filtering: previous at {filtered_t}, now at {cur_t}, {masks.sum()} valid points found! {prefix}\n')
            print(Style.RESET_ALL)
            del points, masks, poses, disps
            torch.cuda.empty_cache()

