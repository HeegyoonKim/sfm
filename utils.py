import os
import cv2
import numpy as np


def is_image_file(filename: str) -> bool:
    """
    Check the input is image file

    Args:
        filename (str): path to file

    Returns:
        bool: True or False
    """
    IMG_EXTENSIONS = ['.jpg', '.JPG', '.jpeg', '.JPEG', '.png', '.PNG',
                      '.ppm', '.PPM', '.bmp', '.BMP']
    
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


def load_images(data_path: str) -> list[np.ndarray]:
    """
    return list of images

    Args:
        data_path (str): path to data directory

    Returns:
        list[np.ndarray]: list of images
    """
    img_list = []
    img_files = os.listdir(os.path.join(os.getcwd(), data_path))
    img_files = [i for i in img_files if is_image_file(i)]
    for img_file in img_files:
        img_list.append(cv2.imread(os.path.join(data_path, img_file)))
    
    return img_list
        


def project_3d_points(P: np.ndarray, pts_3d: np.ndarray) -> np.ndarray:
    """
    Project 3d point to 2d

    Args:
        P (np.ndarray): 3x4 projection matrix.
        pts_3d (np.ndarray): inhomogeneous 3d points. shape: (N, 3).

    Returns:
        np.ndarray: projected 2d points (inhomogeneous). shape: (N, 2).
    """
    pts_3d = np.hstack([pts_3d, np.ones([len(pts_3d),1])])
    proj = P@pts_3d.T
    proj /= proj[-1]
    proj = proj[:2, :].T
    
    return proj


def calculate_reprojection_error(
    pts_2d: np.ndarray, proj_2d: np.ndarray
) -> float:
    """
    Return average error

    Args:
        pts_2d (np.ndarray): 2d points. (N, 2).
        proj_2d (np.ndarray): projected 2d points. (N, 2).

    Returns:
        float: average error
    """
    
    error = (pts_2d - proj_2d) ** 2
    error = np.sum(error, axis=1)
    error = error ** 0.5
    error = np.mean(error)
    
    return error