import glob
import os

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset


def readEXR_onlydepth(filename):
    """
    Read depth data from EXR image file.

    Args:
        filename (str): File path.

    Returns:
        Y (numpy.array): Depth buffer in float32 format.
    """
    # move the import here since only CoFusion needs these package
    # sometimes installation of openexr is hard, you can run all other datasets
    # even without openexr
    import Imath
    import OpenEXR as exr

    exrfile = exr.InputFile(filename)
    header = exrfile.header()
    dw = header['dataWindow']
    isize = (dw.max.y - dw.min.y + 1, dw.max.x - dw.min.x + 1)

    channelData = dict()

    for c in header['channels']:
        C = exrfile.channel(c, Imath.PixelType(Imath.PixelType.FLOAT))
        C = np.fromstring(C, dtype=np.float32)
        C = np.reshape(C, isize)

        channelData[c] = C

    Y = None if 'Y' not in header['channels'] else channelData['Y']

    return Y


def get_dataset(cfg, args, device='cuda:0'):
    return dataset_dict[cfg['dataset']](cfg, args, device=device)


class BaseDataset(Dataset):
    def __init__(self, cfg, args, device='cuda:0'):
        super(BaseDataset, self).__init__()
        self.name = cfg['dataset']
        self.stereo = (cfg['mode'] == 'stereo')
        self.device = device
        self.png_depth_scale = cfg['cam']['png_depth_scale']
        self.n_img = -1
        self.depth_paths = None
        self.color_paths = None
        self.poses = None
        self.image_timestamps = None

        self.H, self.W, self.fx, self.fy, self.cx, self.cy = cfg['cam']['H'], cfg['cam'][
            'W'], cfg['cam']['fx'], cfg['cam']['fy'], cfg['cam']['cx'], cfg['cam']['cy']
        self.H_out, self.W_out = cfg['cam']['H_out'], cfg['cam']['W_out']
        self.H_edge, self.W_edge = cfg['cam']['H_edge'], cfg['cam']['W_edge']

        self.distortion = np.array(
            cfg['cam']['distortion']) if 'distortion' in cfg['cam'] else None

        if args.input_folder is None:
            self.input_folder = cfg['data']['input_folder']
        else:
            self.input_folder = args.input_folder

    def __len__(self):
        return self.n_img

    def depthloader(self, index):
        if self.depth_paths is None:
            return None
        depth_path = self.depth_paths[index]
        if '.png' in depth_path:
            depth_data = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
        elif '.exr' in depth_path:
            depth_data = readEXR_onlydepth(depth_path)
        else:
            raise TypeError(depth_path)
        depth_data = depth_data.astype(np.float32) / self.png_depth_scale

        return depth_data

    def __getitem__(self, index):
        color_path = self.color_paths[index]
        color_data = cv2.imread(color_path)
        if self.distortion is not None:
            K = np.eye(3)
            K[0, 0], K[0, 2], K[1, 1], K[1, 2] = self.fx, self.cx, self.fy, self.cy
            # undistortion is only applied on color image, not depth!
            color_data = cv2.undistort(color_data, K, self.distortion)

        H_out_with_edge, W_out_with_edge = self.H_out + self.H_edge * 2, self.W_out + self.W_edge * 2
        outsize = (H_out_with_edge, W_out_with_edge)

        color_data = cv2.resize(color_data, (W_out_with_edge, H_out_with_edge))
        color_data = torch.from_numpy(color_data).float().permute(2, 0, 1)[[2, 1, 0], :, :] / 255.0  # bgr -> rgb, [0, 1]
        color_data = color_data.unsqueeze(dim=0)  # [1, 3, h, w]

        depth_data = self.depthloader(index)
        if depth_data is not None:
            depth_data = torch.from_numpy(depth_data).float()
            depth_data = F.interpolate(
                depth_data[None, None], outsize, mode='nearest')[0, 0]

        intrinsic = torch.as_tensor([self.fx, self.fy, self.cx, self.cy]).float()
        intrinsic[0] *= W_out_with_edge / self.W
        intrinsic[1] *= H_out_with_edge / self.H
        intrinsic[2] *= W_out_with_edge / self.W
        intrinsic[3] *= H_out_with_edge / self.H

        # crop image edge, there are invalid value on the edge of the color image
        if self.W_edge > 0:
            edge = self.W_edge
            color_data = color_data[:, :, :, edge:-edge]
            depth_data = depth_data[:, edge:-edge]
            intrinsic[2] -= edge

        if self.H_edge > 0:
            edge = self.H_edge
            color_data = color_data[:, :, edge:-edge, :]
            depth_data = depth_data[edge:-edge, :]
            intrinsic[3] -= edge

        if self.poses is not None:
            pose = torch.from_numpy(self.poses[index]).float()
        else:
            pose = None

        return index, color_data, depth_data, intrinsic, pose


class Replica(BaseDataset):
    def __init__(self, cfg, args, device='cuda:0'):
        super(Replica, self).__init__(cfg, args, device)
        stride = cfg['stride']
        self.color_paths = sorted(
            glob.glob(f'{self.input_folder}/results/frame*.jpg'))
        self.depth_paths = sorted(
            glob.glob(f'{self.input_folder}/results/depth*.png'))
        self.n_img = len(self.color_paths)

        self.load_poses(f'{self.input_folder}/traj.txt')
        self.color_paths = self.color_paths[::stride]
        self.depth_paths = self.depth_paths[::stride]
        self.poses = self.poses[::stride]
        self.n_img = len(self.color_paths)

    def load_poses(self, path):
        self.poses = []
        with open(path, "r") as f:
            lines = f.readlines()
        for i in range(self.n_img):
            line = lines[i]
            c2w = np.array(list(map(float, line.split()))).reshape(4, 4)
            self.poses.append(c2w)


class Azure(BaseDataset):
    def __init__(self, cfg, args, device='cuda:0'
                 ):
        super(Azure, self).__init__(cfg, args, device)
        self.color_paths = sorted(
            glob.glob(os.path.join(self.input_folder, 'color', '*.jpg')))
        self.depth_paths = sorted(
            glob.glob(os.path.join(self.input_folder, 'depth', '*.png')))
        self.load_poses(os.path.join(
            self.input_folder, 'scene', 'trajectory.log'))
        self.n_img = len(self.color_paths)

    def load_poses(self, path):
        self.poses = []
        if os.path.exists(path):
            with open(path) as f:
                content = f.readlines()

                # Load .log file.
                for i in range(0, len(content), 5):
                    # format %d (src) %d (tgt) %f (fitness)
                    data = list(map(float, content[i].strip().split(' ')))
                    ids = (int(data[0]), int(data[1]))
                    fitness = data[2]

                    # format %f x 16
                    c2w = np.array(
                        list(map(float, (''.join(
                            content[i + 1:i + 5])).strip().split()))).reshape((4, 4))

                    self.poses.append(c2w)
        else:
            for i in range(self.n_img):
                c2w = np.eye(4)
                self.poses.append(c2w)


class ScanNet(BaseDataset):
    def __init__(self, cfg, args, device='cuda:0'):
        super(ScanNet, self).__init__(cfg, args, device)
        stride = cfg['stride']
        max_frames = args.max_frames
        if max_frames < 0:
            max_frames = int(1e5)
        self.color_paths = sorted(glob.glob(os.path.join(
            self.input_folder, 'color', '*.jpg')), key=lambda x: int(os.path.basename(x)[:-4]))[:max_frames][::stride]
        self.depth_paths = sorted(glob.glob(os.path.join(
            self.input_folder, 'depth', '*.png')), key=lambda x: int(os.path.basename(x)[:-4]))[:max_frames][::stride]
        self.load_poses(os.path.join(self.input_folder, 'pose'))
        self.poses = self.poses[:max_frames][::stride]

        self.n_img = len(self.color_paths)
        print("INFO: {} images got!".format(self.n_img))

    def load_poses(self, path):
        self.poses = []
        pose_paths = sorted(glob.glob(os.path.join(path, '*.txt')),
                            key=lambda x: int(os.path.basename(x)[:-4]))
        for pose_path in pose_paths:
            with open(pose_path, "r") as f:
                lines = f.readlines()
            ls = []
            for line in lines:
                l = list(map(float, line.split(' ')))
                ls.append(l)
            c2w = np.array(ls).reshape(4, 4)
            self.poses.append(c2w)


class CoFusion(BaseDataset):
    def __init__(self, cfg, args, device='cuda:0'
                 ):
        super(CoFusion, self).__init__(cfg, args, device)
        self.input_folder = os.path.join(self.input_folder)
        self.color_paths = sorted(
            glob.glob(os.path.join(self.input_folder, 'colour', '*.png')))
        self.depth_paths = sorted(glob.glob(os.path.join(
            self.input_folder, 'depth_noise', '*.exr')))
        self.n_img = len(self.color_paths)
        self.load_poses(os.path.join(self.input_folder, 'trajectories'))

    def load_poses(self, path):
        # We tried, but cannot align the coordinate frame of cofusion to ours.
        # So here we provide identity matrix as proxy.
        # But it will not affect the calculation of ATE since camera trajectories can be aligned.
        self.poses = []
        for i in range(self.n_img):
            c2w = np.eye(4)

            self.poses.append(c2w)


class TUM_RGBD(BaseDataset):
    def __init__(self, cfg, args, device='cuda:0'
                 ):
        super(TUM_RGBD, self).__init__(cfg, args, device)
        self.color_paths, self.depth_paths, self.poses = self.loadtum(
            self.input_folder, frame_rate=32)
        self.n_img = len(self.color_paths)

    def parse_list(self, filepath, skiprows=0):
        """ read list data """
        data = np.loadtxt(filepath, delimiter=' ',
                          dtype=np.unicode_, skiprows=skiprows)
        return data

    def associate_frames(self, tstamp_image, tstamp_depth, tstamp_pose, max_dt=0.08):
        """ pair images, depths, and poses """
        associations = []
        for i, t in enumerate(tstamp_image):
            if tstamp_pose is None:
                j = np.argmin(np.abs(tstamp_depth - t))
                if (np.abs(tstamp_depth[j] - t) < max_dt):
                    associations.append((i, j))

            else:
                j = np.argmin(np.abs(tstamp_depth - t))
                k = np.argmin(np.abs(tstamp_pose - t))

                if (np.abs(tstamp_depth[j] - t) < max_dt) and \
                        (np.abs(tstamp_pose[k] - t) < max_dt):
                    associations.append((i, j, k))

        return associations

    def loadtum(self, datapath, frame_rate=-1):
        """ read video data in tum-rgbd format """
        if os.path.isfile(os.path.join(datapath, 'groundtruth.txt')):
            pose_list = os.path.join(datapath, 'groundtruth.txt')
        elif os.path.isfile(os.path.join(datapath, 'pose.txt')):
            pose_list = os.path.join(datapath, 'pose.txt')

        image_list = os.path.join(datapath, 'rgb.txt')
        depth_list = os.path.join(datapath, 'depth.txt')

        image_data = self.parse_list(image_list)
        depth_data = self.parse_list(depth_list)
        pose_data = self.parse_list(pose_list, skiprows=1)
        pose_vecs = pose_data[:, 1:].astype(np.float64)

        tstamp_image = image_data[:, 0].astype(np.float64)
        tstamp_depth = depth_data[:, 0].astype(np.float64)
        tstamp_pose = pose_data[:, 0].astype(np.float64)
        associations = self.associate_frames(
            tstamp_image, tstamp_depth, tstamp_pose)

        indicies = [0]
        for i in range(1, len(associations)):
            t0 = tstamp_image[associations[indicies[-1]][0]]
            t1 = tstamp_image[associations[i][0]]
            if t1 - t0 > 1.0 / frame_rate:
                indicies += [i]

        images, poses, depths, intrinsics = [], [], [], []
        inv_pose = None
        for ix in indicies:
            (i, j, k) = associations[ix]
            images += [os.path.join(datapath, image_data[i, 1])]
            depths += [os.path.join(datapath, depth_data[j, 1])]
            # timestamp tx ty tz qx qy qz qw
            c2w = self.pose_matrix_from_quaternion(pose_vecs[k])
            if inv_pose is None:
                inv_pose = np.linalg.inv(c2w)
                c2w = np.eye(4)
            else:
                c2w = inv_pose@c2w

            poses += [c2w]

        return images, depths, poses

    def pose_matrix_from_quaternion(self, pvec):
        """ convert 4x4 pose matrix to (t, q) """
        from scipy.spatial.transform import Rotation

        pose = np.eye(4)
        pose[:3, :3] = Rotation.from_quat(pvec[3:]).as_matrix()
        pose[:3, 3] = pvec[:3]
        return pose


class ETH3D(BaseDataset):
    def __init__(self, cfg, args, device='cuda:0'
                 ):
        super(ETH3D, self).__init__(cfg, args, device)
        stride = cfg['stride']
        self.color_paths, self.depth_paths, self.poses, self.image_timestamps = self.loadtum(
            self.input_folder, frame_rate=-1)
        self.color_paths = self.color_paths[::stride]
        self.depth_paths = self.depth_paths[::stride]
        self.poses = None if self.poses is None else self.poses[::stride]
        self.image_timestamps = self.image_timestamps[::stride]

        self.n_img = len(self.color_paths)

    def parse_list(self, filepath, skiprows=0):
        """ read list data """
        data = np.loadtxt(filepath, delimiter=' ',
                          dtype=np.unicode_, skiprows=skiprows)
        return data

    def associate_frames(self, tstamp_image, tstamp_depth, tstamp_pose, max_dt=0.08):
        """ pair images, depths, and poses """
        associations = []
        for i, t in enumerate(tstamp_image):
            if tstamp_pose is None:
                j = np.argmin(np.abs(tstamp_depth - t))
                # we need all images for benchmark, no max_dt checking here
                # if (np.abs(tstamp_depth[j] - t) < max_dt):
                associations.append((i, j))

            else:
                j = np.argmin(np.abs(tstamp_depth - t))
                k = np.argmin(np.abs(tstamp_pose - t))

                if (np.abs(tstamp_depth[j] - t) < max_dt) and \
                        (np.abs(tstamp_pose[k] - t) < max_dt):
                    associations.append((i, j, k))

        return associations

    def loadtum(self, datapath, frame_rate=-1):
        """ read video data in tum-rgbd format """
        if os.path.isfile(os.path.join(datapath, 'groundtruth.txt')):
            pose_list = os.path.join(datapath, 'groundtruth.txt')
        elif os.path.isfile(os.path.join(datapath, 'pose.txt')):
            pose_list = os.path.join(datapath, 'pose.txt')
        else:
            pose_list = None

        image_list = os.path.join(datapath, 'rgb.txt')
        depth_list = os.path.join(datapath, 'depth.txt')

        image_data = self.parse_list(image_list)
        depth_data = self.parse_list(depth_list)
        tstamp_image = image_data[:, 0].astype(np.float64)
        tstamp_depth = depth_data[:, 0].astype(np.float64)

        if pose_list is not None:
            pose_data = self.parse_list(pose_list, skiprows=1)
            pose_vecs = pose_data[:, 1:].astype(np.float64)

            tstamp_pose = pose_data[:, 0].astype(np.float64)
        else:
            tstamp_pose = None
            pose_vecs = None

        associations = self.associate_frames(
            tstamp_image, tstamp_depth, tstamp_pose)

        images, poses, depths, timestamps = [], [], [], tstamp_image
        if pose_list is not None:
            inv_pose = None
            for ix in range(len(associations)):
                (i, j, k) = associations[ix]
                images += [os.path.join(datapath, image_data[i, 1])]
                depths += [os.path.join(datapath, depth_data[j, 1])]
                # timestamp tx ty tz qx qy qz qw
                c2w = self.pose_matrix_from_quaternion(pose_vecs[k])
                if inv_pose is None:
                    inv_pose = np.linalg.inv(c2w)
                    c2w = np.eye(4)
                else:
                    c2w = inv_pose@c2w

                poses += [c2w]
        else:
            assert len(associations) == len(tstamp_image), 'Not all images are loaded. While benchmark need all images\' pose!'
            print('\nDataset: no gt pose avaliable, {} images found\n'.format(len(tstamp_image)))
            for ix in range(len(associations)):
                (i, j) = associations[ix]
                images += [os.path.join(datapath, image_data[i, 1])]
                depths += [os.path.join(datapath, depth_data[j, 1])]

            poses = None

        return images, depths, poses, timestamps

    def pose_matrix_from_quaternion(self, pvec):
        """ convert 4x4 pose matrix to (t, q) """
        from scipy.spatial.transform import Rotation

        pose = np.eye(4)
        pose[:3, :3] = Rotation.from_quat(pvec[3:]).as_matrix()
        pose[:3, 3] = pvec[:3]
        return pose


class EuRoC(BaseDataset):
    def __init__(self, cfg, args, device='cuda:0'
                 ):
        super(EuRoC, self).__init__(cfg, args, device)
        stride = cfg['stride']
        self.color_paths, self.right_color_paths, self.poses = self.loadtum(
            self.input_folder, frame_rate=-1)
        self.color_paths = self.color_paths[::stride]
        self.right_color_paths = self.right_color_paths[::stride]
        self.poses = None if self.poses is None else self.poses[::stride]

        self.n_img = len(self.color_paths)

        K_l = np.array([458.654, 0.0, 367.215, 0.0, 457.296, 248.375, 0.0, 0.0, 1.0]).reshape(3, 3)
        d_l = np.array([-0.28340811, 0.07395907, 0.00019359, 1.76187114e-05, 0.0])
        R_l = np.array([
            0.999966347530033, -0.001422739138722922, 0.008079580483432283,
            0.001365741834644127, 0.9999741760894847, 0.007055629199258132,
            -0.008089410156878961, -0.007044357138835809, 0.9999424675829176
        ]).reshape(3, 3)

        P_l = np.array([435.2046959714599, 0, 367.4517211914062, 0, 0, 435.2046959714599, 252.2008514404297, 0, 0, 0, 1,
                        0]).reshape(3, 4)
        map_l = cv2.initUndistortRectifyMap(K_l, d_l, R_l, P_l[:3, :3], (752, 480), cv2.CV_32F)

        K_r = np.array([457.587, 0.0, 379.999, 0.0, 456.134, 255.238, 0.0, 0.0, 1]).reshape(3, 3)
        d_r = np.array([-0.28368365, 0.07451284, -0.00010473, -3.555907e-05, 0.0]).reshape(5)
        R_r = np.array([
            0.9999633526194376, -0.003625811871560086, 0.007755443660172947,
            0.003680398547259526, 0.9999684752771629, -0.007035845251224894,
            -0.007729688520722713, 0.007064130529506649, 0.999945173484644
        ]).reshape(3, 3)

        P_r = np.array(
            [435.2046959714599, 0, 367.4517211914062, -47.90639384423901, 0, 435.2046959714599, 252.2008514404297, 0, 0,
             0, 1, 0]).reshape(3, 4)
        map_r = cv2.initUndistortRectifyMap(K_r, d_r, R_r, P_r[:3, :3], (752, 480), cv2.CV_32F)

        self.map_l = map_l
        self.map_r = map_r

    def load_left_image(self, path):
        img = cv2.remap(cv2.imread(path), self.map_l[0], self.map_l[1], interpolation=cv2.INTER_LINEAR)

        return img

    def load_right_image(self, path):
        img = cv2.remap(cv2.imread(path), self.map_r[0], self.map_r[1], interpolation=cv2.INTER_LINEAR)

        return img

    def __getitem__(self, index):
        color_path = self.color_paths[index]
        color_data = self.load_left_image(color_path)

        H_out_with_edge, W_out_with_edge = self.H_out + self.H_edge * 2, self.W_out + self.W_edge * 2

        color_data = cv2.resize(color_data, (W_out_with_edge, H_out_with_edge))
        color_data = torch.from_numpy(color_data).float().permute(2, 0, 1)[[2, 1, 0], :, :] / 255.0  # bgr -> rgb, [0, 1]
        color_data = color_data.unsqueeze(dim=0)  # [1, 3, h, w]

        if self.stereo:
            right_color_path = self.right_color_paths[index]
            right_color_data = self.load_right_image(right_color_path)
            right_color_data = cv2.resize(right_color_data, (W_out_with_edge, H_out_with_edge))
            right_color_data = torch.from_numpy(right_color_data).float().permute(2, 0, 1)[[2, 1, 0], :, :] / 255.0  # bgr -> rgb, [0, 1]
            right_color_data = right_color_data.unsqueeze(dim=0)  # [1, 3, h, w]
            color_data = torch.cat([color_data, right_color_data], dim=0)

        depth_data = None
        intrinsic = torch.as_tensor([self.fx, self.fy, self.cx, self.cy]).float()
        intrinsic[0] *= W_out_with_edge / self.W
        intrinsic[1] *= H_out_with_edge / self.H
        intrinsic[2] *= W_out_with_edge / self.W
        intrinsic[3] *= H_out_with_edge / self.H

        # crop image edge, there are invalid value on the edge of the color image
        if self.W_edge > 0:
            edge = self.W_edge
            color_data = color_data[:, :, :, edge:-edge]
            intrinsic[2] -= edge

        if self.H_edge > 0:
            edge = self.H_edge
            color_data = color_data[:, :, edge:-edge, :]
            intrinsic[3] -= edge

        if self.poses is not None:
            pose = torch.from_numpy(self.poses[index]).float()
        else:
            pose = None

        return index, color_data, depth_data, intrinsic, pose

    def parse_list(self, filepath, skiprows=0):
        """ read list data """
        data = np.loadtxt(filepath, delimiter=' ',
                          dtype=np.unicode_, skiprows=skiprows)
        return data

    def associate_frames(self, tstamp_image, tstamp_depth, tstamp_pose, max_dt=0.08):
        """ pair images, depths, and poses """
        associations = []
        for i, t in enumerate(tstamp_image):
            if tstamp_pose is None:
                j = np.argmin(np.abs(tstamp_depth - t))
                if (np.abs(tstamp_depth[j] - t) < max_dt):
                    associations.append((i, j))

            if tstamp_depth is None:
                k = np.argmin(np.abs(tstamp_pose - t))
                if (np.abs(tstamp_pose[k] - t) < max_dt):
                    associations.append((i, k))

            else:
                j = np.argmin(np.abs(tstamp_depth - t))
                k = np.argmin(np.abs(tstamp_pose - t))

                if (np.abs(tstamp_depth[j] - t) < max_dt) and \
                        (np.abs(tstamp_pose[k] - t) < max_dt):
                    associations.append((i, j, k))

        return associations

    def loadtum(self, datapath, frame_rate=-1):
        """ read video data in tum-rgbd format """
        # download from: https://github.com/princeton-vl/DROID-SLAM/tree/main/data/euroc_groundtruth
        scene_name = datapath.split('/')[-1]
        if os.path.isfile(os.path.join(datapath, f'{scene_name}.txt')):
            pose_list = os.path.join(datapath, f'{scene_name}.txt')
        else:
            raise ValueError(f'EuRoC_DATA_ROOT/{scene_name}/{scene_name}.txt doesn\'t exist!')

        pose_data = self.parse_list(pose_list, skiprows=1)
        pose_vecs = pose_data[:, 1:].astype(np.float64)

        image_list = sorted(glob.glob(os.path.join(datapath, 'mav0/cam0/data/*.png')))
        right_image_list = [x.replace('cam0', 'cam1') for x in image_list]
        tstamp_image = [float(img.split('/')[-1][:-4]) for img in image_list]
        tstamp_depth = None
        tstamp_pose = pose_data[:, 0].astype(np.float64)

        associations = self.associate_frames(
            tstamp_image, tstamp_depth, tstamp_pose)

        images, poses, right_images, intrinsics = [], [], [], []
        inv_pose = None
        for ix in range(len(associations)):
            (i, k) = associations[ix]
            images += [image_list[i]]
            right_images += [right_image_list[i]]
            # timestamp tx ty tz qx qy qz qw
            c2w = self.pose_matrix_from_quaternion(pose_vecs[k])
            if inv_pose is None:
                inv_pose = np.linalg.inv(c2w)
                c2w = np.eye(4)
            else:
                c2w = inv_pose@c2w

            poses += [c2w]

        return images, right_images, poses

    def pose_matrix_from_quaternion(self, pvec):
        """ convert 4x4 pose matrix to (t, q) """
        from scipy.spatial.transform import Rotation

        pose = np.eye(4)
        pose[:3, :3] = Rotation.from_quat(pvec[3:]).as_matrix()
        pose[:3, 3] = pvec[:3]
        return pose


dataset_dict = {
    "replica": Replica,
    "scannet": ScanNet,
    "cofusion": CoFusion,
    "azure": Azure,
    "tumrgbd": TUM_RGBD,
    'eth3d': ETH3D,
    'euroc': EuRoC,
}
