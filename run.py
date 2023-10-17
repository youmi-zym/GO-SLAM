import numpy as np
import torch
import argparse
import shutil
import os

from src import config
from src.slam import SLAM
from src.datasets import get_dataset


import random
def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def backup_source_code(backup_directory):
    ignore_hidden = shutil.ignore_patterns(
        '.', '..', '.git*', '*pycache*', '*build', '*.fuse*', '*_drive_*',
        '*pretrained*', '*output*', '*media*', '*.so', '*.pyc', '*.Python',
        '*.eggs*', '*.DS_Store*', '*.idea*', '*.pth', '*__pycache__*', '*.ply',
        '*exps*',
    )

    if os.path.exists(backup_directory):
        shutil.rmtree(backup_directory)

    shutil.copytree('.', backup_directory, ignore=ignore_hidden)
    os.system('chmod -R g+w {}'.format(backup_directory))


if __name__ == '__main__':
    setup_seed(43)

    parser = argparse.ArgumentParser()
    parser.add_argument('config', type=str, help='Path to config file.')
    parser.add_argument("--device", type=str, default='cuda:0')
    parser.add_argument("--max_frames", type=int, default=-1, help="Only [0, max_frames] Frames will be run")
    parser.add_argument("--only_tracking", action="store_true", help="Only tracking is triggered")
    parser.add_argument("--make_video", action="store_true", help="to generate video as in our project page")
    parser.add_argument('--input_folder', type=str,
                        help='input folder, this have higher priority, can overwrite the one in config file')
    parser.add_argument('--output', type=str,
                        help='output folder, this have higher priority, can overwrite the one in config file')
    parser.add_argument("--image_size", nargs='+', default=None,
                        help='image height and width, this have higher priority, can overwrite the one in config file')
    parser.add_argument('--calibration_txt', type=str, default=None,
                        help='calibration parameters: fx, fy, cx, cy, this have higher priority, can overwrite the one in config file')
    parser.add_argument('--mode', type=str,
                        help='slam mode: mono, rgbd or stereo')
    args = parser.parse_args()

    torch.multiprocessing.set_start_method('spawn')

    cfg = config.load_config(
        args.config, './configs/go_slam.yaml'
    )

    if args.mode is not None:
        cfg['mode'] = args.mode
    if args.only_tracking:
        cfg['only_tracking'] = True
    if args.image_size is not None:
        cfg['cam']['H'], cfg['cam']['W'] = args.image_size
    if args.calibration_txt is not None:
        cfg['cam']['fx'], cfg['cam']['fy'], cfg['cam']['cx'], cfg['cam']['cy'] = np.loadtxt(args.calibration_txt).tolist()

    assert cfg['mode'] in ['rgbd', 'mono', 'stereo'], cfg['mode']
    print(f"\n\n** Running {cfg['data']['input_folder']} in {cfg['mode']} mode!!! **\n\n")

    print(args)

    if args.output is None:
        output_dir = cfg['data']['output']
    else:
        output_dir = args.output

    backup_source_code(os.path.join(output_dir, 'code'))
    config.save_config(cfg, f'{output_dir}/cfg.yaml')

    dataset = get_dataset(cfg, args, device=args.device)

    slam = SLAM(args, cfg)
    slam.run(dataset)

    slam.terminate(rank=-1, stream=dataset)

    print('Done!')

