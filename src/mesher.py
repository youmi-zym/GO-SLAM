import os
import copy
import numpy as np
import open3d as o3d
import torch
import torch.nn.functional as F
import trimesh
import matplotlib.pyplot as plt
import pyrender
from copy import deepcopy
from scipy.spatial import cKDTree as KDTree
from lietorch import SE3
from .oriented_bounding_box import OrientedBoundingBox


class Mesher(object):
    def __init__(self, cfg, args, slam, points_batch_size=5e5):
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

        self.points_batch_size = int(points_batch_size)
        self.output = slam.output
        self.shared_mapping_net = slam.mapping_net
        self.video = slam.video
        self.reload_map = slam.reload_map
        self.scale = 1.0

        self.resolution = cfg['meshing']['resolution']  # 256
        self.level_set = cfg['meshing']['level_set']  # 0.0
        self.remove_small_geometry_threshold = cfg['meshing']['remove_small_geometry_threshold']  # 0.2
        self.get_largest_components = cfg['meshing']['get_largest_components']  # False
        self.eval_rec = cfg['meshing']['eval_rec']
        self.n_points_to_eval = cfg['meshing']['n_points_to_eval']
        self.mesh_threshold_to_eval = cfg['meshing']['mesh_threshold_to_eval']
        self.gt_mesh_path = cfg['meshing']['gt_mesh_path']
        self.forecast_radius = cfg['meshing']['forecast_radius']

        assert self.forecast_radius >= 0, self.forecast_radius

        self.verbose = slam.verbose
        self.device = cfg['mapping']['device']

        os.makedirs(f'{self.output}/mesh/', exist_ok=True)

        self.H, self.W, self.fx, self.fy, self.cx, self.cy = slam.H, slam.W, slam.fx, slam.fy, slam.cx, slam.cy

    def point_masks(self,
                    input_points,
                    depth_list,
                    estimate_c2w_list):
        """
        Split the input points into seen, unseen, and forecast,
        according to the estimated camera pose and depth image.
        Args:
            input_points:               (Tensor), input points
            keyframe_dict:              (list), list of keyframe info dictionary
            estimate_c2w_list:          (list), estimated camera pose.
            idx:                        (int), current frame index
            device:                     (str), device name to compute on.
            get_mask_use_all_frames:

        Returns:
            seen_mask:                  (Tensor), the mask for seen area.
            forecast_mask:              (Tensor), the mask for forecast area.
            unseen_mask:                (Tensor), the mask for unseen area.

        """
        H, W, fx, fy, cx, cy = self.H, self.W, self.fx, self.fy, self.cx, self.cy
        device =self.device
        if not isinstance(input_points, torch.Tensor):
            input_points = torch.from_numpy(input_points)
        input_points = input_points.clone().detach().float()
        mask = []
        forecast_mask = []
        eps = 0.05
        for _, pnts in enumerate(torch.split(input_points, self.points_batch_size, dim=0)):
            n_pts, _ = pnts.shape
            valid = torch.zeros(n_pts).to(device).bool()
            valid_forecast = torch.zeros(n_pts).to(device).bool()
            r = self.forecast_radius
            for i in range(len(estimate_c2w_list)):
                points = pnts.to(device).float()
                c2w = estimate_c2w_list[i].to(device).float()
                depth = depth_list[i].to(device)
                w2c = torch.inverse(c2w).to(device).float()
                ones = torch.ones_like(points[:, 0]).reshape(-1, 1).to(device)
                homo_points = torch.cat([points, ones], dim=1).reshape(-1, 4, 1).to(device).float()
                cam_cord_homo = w2c @ homo_points
                cam_cord = cam_cord_homo[:, :3, :]  # [N, 3, 1]
                K = np.eye(3)
                K[0, 0], K[0, 2], K[1, 1], K[1, 2] = fx, cx, fy, cy
                K = torch.from_numpy(K).to(device)

                uv = K.float() @ cam_cord.float()
                z = uv[:, -1:] + 1e-8
                uv = uv[:, :2] / z  # [N, 2, 1]
                u, v = uv[:, 0, 0].float(), uv[:, 1, 0].float()
                z = z[:, 0, 0].float()

                in_frustum = (u >= 0) & (u <= W-1) & (v >= 0) & (v <= H-1) & (z > 0)
                forecast_frustum = (u >= -r) & (u <= W-1+r) & (v >= -r) & (v <= H-1+r) & (z > 0)

                depth = depth.reshape(1, 1, H, W)
                vgrid = uv.reshape(1, 1, -1, 2)
                # normalized to [-1, 1]
                vgrid[..., 0] = (vgrid[..., 0] / (W - 1) * 2.0 - 1.0)
                vgrid[..., 1] = (vgrid[..., 1] / (H - 1) * 2.0 - 1.0)

                depth_sample = F.grid_sample(depth, vgrid, padding_mode='border', align_corners=True)
                depth_sample = depth_sample.reshape(-1)
                is_front_face = torch.where((depth_sample > 0.0), (z < (depth_sample + eps)), torch.ones_like(z).bool())
                is_forecast_face = torch.where((depth_sample > 0.0), (z < (depth_sample + eps)), torch.ones_like(z).bool())
                in_frustum = in_frustum & is_front_face

                valid = valid | in_frustum.bool()

                forecast_frustum = forecast_frustum & is_forecast_face
                forecast_frustum = in_frustum | forecast_frustum
                valid_forecast = valid_forecast | forecast_frustum.bool()

            mask.append(valid.cpu().numpy())
            forecast_mask.append(valid_forecast.cpu().numpy())

        mask = np.concatenate(mask, axis=0)
        forecast_mask = np.concatenate(forecast_mask, axis=0)

        return mask, forecast_mask


    @torch.no_grad()
    def get_connected_mesh(self, mesh, get_largest_components=False):
        components = mesh.split(only_watertight=False)
        if get_largest_components:
            areas = np.array([c.area for c in components], dtype=np.float)
            mesh = components[areas.argmax()]
        else:
            new_components = []
            global_area = mesh.area
            for comp in components:
                if comp.area > self.remove_small_geometry_threshold * global_area:
                    new_components.append(comp)
            mesh = trimesh.util.concatenate(new_components)

        return mesh

    @torch.no_grad()
    def cull_mesh(self,
                  mesh,
                  estimate_c2w_list,
                  bound,
                  mesh_out_file):
        """
        Extract mesh from scene representation and save mesh to file.
        Args:
            mesh_out_file:              (str), output mesh filename
            estimate_c2w_list:          (Tensor), estimated camera pose, camera coordinate system is same with OpenCV
                                        [N, 4, 4]
        """

        step = 1
        with torch.no_grad():

            if bound is not None:
                # cull with bound
                print(f'Start Mesh Culling:  {step}(Bound)', end='')
                vertices = mesh.vertices[:, :3]
                if isinstance(bound, np.ndarray):
                    eps = 0.001
                    bound_mask = np.all(vertices >= (bound[:, 0] - eps), axis=1) & \
                                 np.all(vertices <= (bound[:, 1] + eps), axis=1)
                else:
                    bound_mask = bound.in_bound(np.array(vertices)) # N'
                face_mask = bound_mask[mesh.faces].all(axis=1)
                mesh.update_faces(face_mask)
                mesh.remove_unreferenced_vertices()
                mesh.export(f'{self.output}/mesh/bound_mesh.ply')
                step += 1

            # cull with 3d projection
            print(f' --->> {step}(Projection)', end='')
            depth_list = extract_depth_from_mesh(
                mesh, estimate_c2w_list, H=self.H, W=self.W, fx=self.fx, fy=self.fy, cx=self.cx, cy=self.cy, far=20.0
            )

            vertices = mesh.vertices[:, :3]
            mask, forecast_mask = self.point_masks(
                vertices, depth_list, estimate_c2w_list
            )
            face_mask = mask[mesh.faces].all(axis=1)
            mesh_with_hole = deepcopy(mesh)
            mesh_with_hole.update_faces(face_mask)
            mesh_with_hole.remove_unreferenced_vertices()
            mesh_with_hole.export(f'{self.output}/mesh/mesh_with_hole.ply')
            step += 1

            # cull by computing connected components
            print(f' --->> {step}(Component)', end='')
            cull_mesh = self.get_connected_mesh(mesh_with_hole, self.get_largest_components)
            step += 1

            if abs(self.forecast_radius) > 0:
                # for forecasting
                print(f' --->> {step}(Forecast:{self.forecast_radius})', end='')
                forecast_face_mask = forecast_mask[mesh.faces].all(axis=1)
                forecast_mesh = deepcopy(mesh)
                forecast_mesh.update_faces(forecast_face_mask)
                forecast_mesh.remove_unreferenced_vertices()

                cull_pc = o3d.geometry.PointCloud(
                    o3d.utility.Vector3dVector(np.array(cull_mesh.vertices))
                )
                aabb = cull_pc.get_oriented_bounding_box()
                indices = aabb.get_point_indices_within_bounding_box(
                    o3d.utility.Vector3dVector(np.array(forecast_mesh.vertices))
                )
                bound_mask = np.zeros(len(forecast_mesh.vertices))
                bound_mask[indices] = 1.0
                bound_mask = bound_mask.astype(np.bool_)
                forecast_face_mask = bound_mask[forecast_mesh.faces].all(axis=1)
                forecast_mesh.update_faces(forecast_face_mask)
                forecast_mesh.remove_unreferenced_vertices()
                forecast_mesh = self.get_connected_mesh(forecast_mesh, self.get_largest_components)
                step += 1
            else:
                forecast_mesh = deepcopy(cull_mesh)

            cull_mesh.export(mesh_out_file)
            forecast_mesh.export(mesh_out_file.replace('.ply', '_forecast.ply'))
            print(' --->> Done!')

            return cull_mesh, forecast_mesh

    def update_param_from_mapping(self, the_end=False):
        """
        Update the parameters of scene representation from the mapping thread.

        """
        net = copy.deepcopy(self.shared_mapping_net).to(self.device)
        cur_idx = self.video.counter.value
        timestamp = self.video.timestamp[cur_idx-1]
        aabb = None
        kf_c2w_list = SE3(self.video.poses.detach()[:cur_idx]).inv().matrix().data.cpu()

        if the_end:
            import droid_backends
            import open3d as o3d
            filter_thresh = 0.01
            filter_visible_num = 3

            dirty_index = torch.arange(0, cur_idx).long().to(self.device)
            poses = torch.index_select(self.video.poses.detach(), dim=0, index=dirty_index)
            disps = torch.index_select(self.video.disps_up.detach(), dim=0, index=dirty_index)
            common_intrinsic_id = 0  # we assume the intrinsics are the same within one scene
            intrinsic = self.video.intrinsics[common_intrinsic_id].detach() * self.video.scale_factor
            w2w = SE3(self.video.pose_compensate[0].clone().unsqueeze(dim=0)).to(self.device)

            points = droid_backends.iproj((w2w * SE3(poses).inv()).data, disps, intrinsic).cpu() # [b, h, w 3]
            thresh = filter_thresh * torch.ones_like(disps.mean(dim=[1, 2]))
            count = droid_backends.depth_filter(
                poses, disps, intrinsic, dirty_index, thresh
            )  # [b, h, w]

            count = count.cpu()
            disps = disps.cpu()
            masks = (count >= filter_visible_num)
            masks = masks & (disps > 0.01 * disps.mean(dim=[1, 2], keepdim=True))  # filter out far points, [b, h, w]
            sel_points = points.reshape(-1, 3)[masks.reshape(-1)]

            aabb = OrientedBoundingBox()
            aabb.compute_from_pointcloud(sel_points.detach().cpu().numpy(), extend=0.1)

        return timestamp, cur_idx-1, net, aabb, kf_c2w_list


    def __call__(self, the_end=False, estimate_c2w_list=None, gt_c2w_list=None, trans_init=None):
        if self.reload_map > 0 or the_end:

            timestamp, cur_idx, net, bound, kf_c2w_list = self.update_param_from_mapping(the_end=True)
            prefix = f'final_raw' if the_end else f'{int(timestamp):05d}'
            mesh_out_file = f'{self.output}/mesh/{prefix}_mesh.ply'

            mesh = net.extract_geometry(
                resolution=self.resolution,
                threshold=self.level_set,
                c2w_ref=None,
                save_path=None,
                color=True)
            mesh.export(mesh_out_file)

            if len(mesh.vertices) < 500:
                return

            c2w_list = estimate_c2w_list if estimate_c2w_list is not None else kf_c2w_list
            cull_mesh, forecast_mesh = self.cull_mesh(
                mesh=mesh,
                bound=bound,
                estimate_c2w_list=c2w_list,
                mesh_out_file=mesh_out_file,
            )

            if the_end:
                if os.path.exists(self.gt_mesh_path) and self.gt_mesh_path.find('.ply') > -1:
                    gt_mesh = trimesh.load_mesh(self.gt_mesh_path, process=False)

                    aligned_mesh, transformation = align_mesh(cull_mesh, gt_mesh, threshold=0.1, trans_init=trans_init, return_transformation=True)
                    aligned_mesh.export(f'{self.output}/mesh/aligned_mesh.ply')

                    forecast_mesh.apply_transform(transformation)
                    forecast_mesh.export(f'{self.output}/mesh/forecast_aligned_mesh.ply')

                    if self.eval_rec:

                        eval_mesh(
                            forecast_mesh, gt_mesh,
                            N3d=self.n_points_to_eval,
                            dist_th=self.mesh_threshold_to_eval,
                            out_path=f'{self.output}/metrics_mesh.txt',
                        )

            if self.verbose:
                print("\nINFO: Save mesh at {}!\n".format(mesh_out_file))

            del estimate_c2w_list, mesh, net, cull_mesh, forecast_mesh

            torch.cuda.empty_cache()

            self.reload_map -= 1


def align_mesh(est_mesh, gt_mesh, threshold=0.1, trans_init=None, return_transformation=False):
    """
    Get the transformation matrix to align the reconstructed mesh to the ground truth mesh.
    """
    o3d_rec_pc = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(np.array(est_mesh.vertices)))
    o3d_gt_pc = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(np.array(gt_mesh.vertices)))
    if trans_init is None:
        trans_init = np.eye(4)
    reg_p2p = o3d.pipelines.registration.registration_icp(
        o3d_rec_pc, o3d_gt_pc, threshold, trans_init,
        o3d.pipelines.registration.TransformationEstimationPointToPoint())
    transformation = reg_p2p.transformation

    aligned_mesh = est_mesh.apply_transform(transformation)

    if return_transformation:
        return aligned_mesh, transformation
    else:
        return aligned_mesh


def draw_mesh_error(est_mesh, gt_mesh, out_path=None, cmap='jet', display=False, error_type='accuracy'):
    if error_type == 'accuracy':
        src_pc = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(np.array(est_mesh.vertices)))
        target_pc = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(np.array(gt_mesh.vertices)))
    else: #  'completion'
        src_pc = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(np.array(gt_mesh.vertices)))
        target_pc = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(np.array(est_mesh.vertices)))

    # For each point in the source point cloud, compute the distance to the target point cloud.
    dists = src_pc.compute_point_cloud_distance(target_pc)
    dists = np.array(dists).astype(np.float32)

    color = np.zeros_like(dists)
    breakpoints = np.array([0.0, 0.02, 0.05, 0.10, 0.20, np.max(dists)])
    bins =        np.array([0.0, 0.25, 0.38, 0.66, 0.83, 1.00])
    for i in range(1, len(breakpoints)):
        mask = (dists > breakpoints[i-1]) & (dists <= breakpoints[i])
        if np.sum(mask) >= 1.0:
            scale = bins[i] - bins[i-1]
            color[mask] = (dists[mask] - breakpoints[i-1]) / (breakpoints[i] - breakpoints[i-1] + 1e-7) * scale + bins[i-1]
    error_color = plt.cm.get_cmap(cmap)(color)[..., :3]
    src_pc.colors = o3d.utility.Vector3dVector(error_color)

    if display:
        o3d.visualization.draw_geometries([src_pc])

    if out_path is not None:
        o3d.io.write_point_cloud(out_path, src_pc)


def eval_mesh(est_mesh, gt_mesh, N3d=2e5, dist_th=0.05, out_path=None, metric_2d=False):
    N3d = int(N3d)

    est_pc = trimesh.PointCloud(vertices=trimesh.sample.sample_surface(est_mesh, N3d)[0]).vertices
    gt_pc = trimesh.PointCloud(vertices=trimesh.sample.sample_surface(gt_mesh, N3d)[0]).vertices

    est_tree = KDTree(est_pc)
    gt_tree = KDTree(gt_pc)

    # completion ratio
    dist, _ = est_tree.query(gt_pc)
    completion = np.mean(dist) * 100  # cm
    completion_ratio = np.mean((dist < dist_th).astype(np.float32)) * 100  # %

    # accuracy
    dist, _ = gt_tree.query(est_pc)
    accuracy = np.mean(dist) * 100  # cm
    accuracy_ratio = np.mean((dist < dist_th).astype(np.float32)) * 100 # %

    f_score = (2.0 * accuracy_ratio * completion_ratio) / (accuracy_ratio + completion_ratio)

    msg = f'\n\nMetrics of reconstructed mesh are:\n' \
          f'\tAccuracy: {accuracy:.2f}cm\n' \
          f'\tCompletion: {completion:.2f}cm\n' \
          f'\tAccuracy Ratio: {accuracy_ratio:.2f}%\n' \
          f'\tCompletion Ratio: {completion_ratio:.2f}%\n' \
          f'\tF-score: {f_score:.2f}%\n\n'

    if out_path is not None:
        with open(out_path, 'w') as fp:
            fp.write(msg)
    print(msg)


def convex_hull(mesh, mesh_with_hole):
    o3d_pc = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(np.array(mesh_with_hole.vertices)))
    mesh_hull, _ = o3d_pc.compute_convex_hull()
    mesh_hull.compute_vertex_normals()
    # mesh_hull = mesh_hull.scale(1.02, mesh_hull.get_center())
    vertices = np.array(mesh_hull.vertices)
    faces = np.array(mesh_hull.triangles)
    mesh_hull = trimesh.Trimesh(vertices=vertices, faces=faces)
    contain_mask = []

    vertices_list = np.array_split(mesh.vertices, int(5e5), axis=0)
    for i, pnts in enumerate(vertices_list):
        contain_mask.append(mesh_hull.contains(pnts))
    contain_mask = np.concatenate(contain_mask, axis=0)
    face_mask = contain_mask[mesh.faces].all(axis=1)
    mesh.update_faces(face_mask)

    return mesh


def extract_depth_from_mesh(mesh,
                            c2w_list,
                            H, W, fx, fy, cx, cy,
                            far=20.0, ):
    """Adapted from Go-Surf: https://github.com/JingwenWang95/go-surf"""
    os.environ['PYOPENGL_PLATFORM'] = 'egl'  # allows for GPU-accelerated rendering

    mesh = pyrender.Mesh.from_trimesh(mesh)
    scene = pyrender.Scene()
    scene.add(mesh)
    # pyrender.Viewer(scene, use_raymond_lighting=True, show_world_axis=True)
    camera = pyrender.IntrinsicsCamera(fx=fx, fy=fy, cx=cx, cy=cy, znear=0.001, zfar=far)
    camera_node = pyrender.Node(camera=camera, matrix=np.eye(4))
    scene.add_node(camera_node)
    renderer = pyrender.OffscreenRenderer(W, H)
    flags = pyrender.RenderFlags.OFFSCREEN | pyrender.RenderFlags.DEPTH_ONLY | pyrender.RenderFlags.SKIP_CULL_FACES

    depths = []
    for i, c2w in enumerate(c2w_list):
        c2w = c2w.detach().numpy()
        # Convert camera coordinate system from OpenCV to OpenGL
        # Details refer to: https://pyrender.readthedocs.io/en/latest/examples/cameras.html
        c2w_gl = deepcopy(c2w)
        c2w_gl[:3, 1] *= -1
        c2w_gl[:3, 2] *= -1
        scene.set_pose(camera_node, c2w_gl)
        depth = renderer.render(scene, flags)
        # plt.imshow(depth, cmap='jet')
        # plt.show()
        depth = torch.from_numpy(depth)
        # print(f'Depth {i:04d} Min: {depth.min():.4f}, Max: {depth.max():.4f}.')
        depths.append(depth)

    renderer.delete()

    return depths

