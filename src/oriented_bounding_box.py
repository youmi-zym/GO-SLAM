import numpy as np
import torch
import torch.nn as nn
import open3d as o3d


class OrientedBoundingBox(nn.Module):
    def __init__(self):
        super(OrientedBoundingBox, self).__init__()

        self.register_buffer('center', torch.zeros(3,).double())
        self.register_buffer('R', torch.zeros(3, 3).double())
        self.register_buffer('extent', torch.zeros(3,).double())

    def _init(self, center, R, extent):
        device = self.center.device
        self.center[:] = center.to(device).double()
        self.R[:] = R.to(device).double()
        self.extent[:] = extent.to(device).double()

    def _clone(self, aabb):
        center = aabb.center
        R = aabb.R
        extent = aabb.extent

        self._init(center, R, extent)

    def compute_from_pointcloud(self, pointcloud:np.ndarray, extend:float=0.0):
        device = self.center.device

        aabb = o3d.geometry.OrientedBoundingBox.create_from_points(
            o3d.utility.Vector3dVector(np.array(pointcloud))
        )

        center = torch.from_numpy(np.array(aabb.center).astype(np.float64)).to(device)
        R = torch.from_numpy(np.array(aabb.R).astype(np.float64)).to(device)
        extent = torch.from_numpy(np.array(aabb.extent).astype(np.float64)).to(device) + extend

        self.center[:] = center
        self.R[:] = R
        self.extent[:] = extent

    def in_bound(self, pointcloud:np.ndarray):
        n_pts, _ = pointcloud.shape
        valid = np.zeros(n_pts)

        aabb = self.get_aabb()

        indices = aabb.get_point_indices_within_bounding_box(
            o3d.utility.Vector3dVector(np.array(pointcloud))
        )

        valid[indices] = 1

        valid = valid.astype(np.bool_)

        return valid

    def get_aabb(self):
        center = self.center.detach().cpu().numpy().astype(np.float64)
        R = self.R.detach().cpu().numpy().astype(np.float64)
        extent = self.extent.detach().cpu().numpy().astype(np.float64)
        aabb = o3d.geometry.OrientedBoundingBox(center=center, R=R, extent=extent)

        return aabb

    def get_axis_aligned_bounding_box(self):
        aabb = self.get_aabb()
        min_bound = np.array(aabb.get_min_bound()).astype(np.float32)
        max_bound = np.array(aabb.get_max_bound()).astype(np.float32)
        bound = np.concatenate([
            min_bound[:, None],
            max_bound[:, None],
        ], axis=1)

        return bound
