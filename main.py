import os
import argparse

from types_ import *
from utils import load_images


def parse_args() -> Dict[str, Any]:
    parser = argparse.ArgumentParser()
    parser.add_argument('--calibrate_camera', default=False, action='store_true')
    parser.add_argument('--checkerboard_path', type=str, default='data/checkerboard')
    parser.add_argument('--data_path', type=str, default='data/')
    args = parser.parse_args()
    
    return args


def main(args):
    img_list = load_images(args.data_path)


if __name__ == '__main__':
    args = parse_args()
    main(args)
    print('Done')