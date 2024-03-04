import os
import cv2
import numpy as np
import logging

from types_ import *
from utils import load_images


def calibrate_camera(
    cb_img_list: list[np.ndarray],
    n_rows: int = 7,
    n_cols: int = 10
) -> tuple[float, np.ndarray, np.ndarray]:
    """
    Args:
        cb_img_list (list[np.ndarray]): list of images
        n_rows (int, optional): number of rows in checkerboard. Defaults to 7.
        n_cols (int, optional): nubmer of columns. Defaults to 10.

    Returns:
        tuple[float, np.ndarray, np.ndarray]: reprojection error, K, distortion coefficients.
    """
    pts_3d = np.zeros([n_rows*n_cols,3], dtype=np.float32)
    pts_3d[:, :2] = np.mgrid[0:n_cols, 0:n_rows].T.reshape(-1, 2)
    pts_3d_list = []; pts_2d_list = []
    
    for cb_img in cb_img_list:
        img_gray = cv2.cvtColor(cb_img, cv2.COLOR_BGR2GRAY)
        
        # Find corners (top-left -> bottom-right)
        ret, corners = cv2.findChessboardCorners(img_gray, (n_cols,n_rows), None)
        if ret:
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
            corners = cv2.cornerSubPix(img_gray, corners, (n_cols,n_rows), (-1,-1), criteria)
            corners = np.squeeze(corners, axis=1)   # (n, 1, 2) -> (n, 2)
            corners = np.flip(corners, axis=0)  # reverse order
            
            pts_3d_list.append(pts_3d)
            pts_2d_list.append(corners)
    
    H, W = cb_img_list[0].shape[:2]
    reproj_error, K, dist, _, _ = cv2.calibrateCamera(pts_3d_list, pts_2d_list,
                                                      (W,H), None, None)
    
    return reproj_error, K, dist


def get_camera_parameters(args: Dict[str, Any]) -> tuple[np.ndarray]:
    """
    Return camera parameters.
    K: intrinsic, dist: distortion coefficients

    Args:
        args (Dict): args

    Returns:
        tuple[np.ndarray]: K, dist
    """
    c_param_path = os.path.join(os.getcwd(), args.checkerboard_path, 'cam_info.npz')
    # Load or calibrate camera parameters
    if os.path.isfile(c_param_path):
        logging.info('Load camera parameters.')
        c_params = np.load(c_param_path)
        K = c_params['K']
        dist = c_params['dist']
        reproj_error = c_params['reproj_error']
    else:
        logging.info('Calibrate camera.')
        cb_img_list = load_images(args.checkerboard_path)
        reproj_error, K, dist = calibrate_camera(cb_img_list, args.cb_n_rows,
                                                 args.cb_n_cols)
        np.savez(c_param_path, K=K, dist=dist, reproj_error=reproj_error)
    
    logging.info(f'\nIntrinsic K\n{K}\nDistortion coefficients\n{dist}\n'
                 f'Reprojection error: {reproj_error:.6f}')
    
    return K, dist