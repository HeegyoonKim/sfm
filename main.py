import os
import argparse

from types_ import *
from utils import load_images
from calibration import get_camera_parameters

import logging
logging.basicConfig(format='%(asctime)s  %(message)s', level=logging.INFO)


def parse_args() -> Dict[str, Any]:
    parser = argparse.ArgumentParser()
    parser.add_argument('--c_param_path', type=str, default='data')
    parser.add_argument('--checkerboard_path', type=str, default='data/checkerboard')
    parser.add_argument('--data_path', type=str, default='data')
    parser.add_argument('--cb_n_rows', type=int, default=7)
    parser.add_argument('--cb_n_cols', type=int, default=10)
   
    args = parser.parse_args()
    
    return args


def main(args):
    # Intrinsic K, distortion coefficient dist
    K, dist = get_camera_parameters(args)
    
    img_list = load_images(args.data_path)


if __name__ == '__main__':
    args = parse_args()
    main(args)
    print('Done')