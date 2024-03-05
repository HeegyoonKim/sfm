import argparse
import logging
logging.basicConfig(format='%(asctime)s  %(message)s', level=logging.INFO)

from types_ import *
from utils import load_images
from calibration import get_camera_parameters, undistort_images
from feature_matching import build_matching_table


def parse_args() -> Dict[str, Any]:
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkerboard_path', type=str, default='data/checkerboard')
    parser.add_argument('--cb_n_rows', type=int, default=7)
    parser.add_argument('--cb_n_cols', type=int, default=10)
    parser.add_argument('--data_path', type=str, default='data/images')
    
    args = parser.parse_args()
    
    return args


def main(args):
    # Intrinsic K, distortion coefficient dist
    K, dist = get_camera_parameters(args)
    
    # Load images
    img_list = load_images(args.data_path)
    n_frames = len(img_list)
    
    # Undistort images
    img_list, K = undistort_images(img_list, dist, K)
    
    # Feature matching
    build_matching_table(img_list, K)


if __name__ == '__main__':
    args = parse_args()
    main(args)
    print('Done')