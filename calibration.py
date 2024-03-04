import os
import cv2
import numpy as np

from utils import project_3d_point, calculate_reprojection_error


def decompose_projection_matrix(
    P: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    P = K[R|t] = [KR|-KRC]. C: camera center
    Not unique.

    Args:
        P (np.ndarray): projection matrix.

    Returns:
        tuple[np.ndarray, np.ndarray, np.ndarray]: intrinsic, rotation, translation.
    """
    raise NotImplementedError


def estimate_homography(
    pts_3d: np.ndarray, pts_2d: np.ndarray
) -> tuple[np.ndarray, float]:
    """
    Estimate homography (or projection matrix) between checkerboard (3D)
    and captured image (2d).
    H = K[r1 r2 t].
    2d point x and 3d point X, x = H x X.
    svd(H).

    Args:
        pts_3d (np.ndarray): 3d points of checkerboard. (N, 3)-shaped, z-axis is 0.
        pts_2d (np.ndarray): 2d points in captured image. (N, 2)-shaped.

    Returns:
        tuple[np.ndarray, float]: homography H and reprojection error.
    """
    
    if len(pts_3d) < 4:
        raise ValueError('Number of points is greater than 3')
    
    # Ax=0 using svd
    A = []
    for i in range(len(pts_3d)):
        X = pts_3d[i]
        x = pts_2d[i]
        
        # Stack equations
        A.append([X[0], X[1], 1, 0, 0, 0, -x[0]*X[0], -x[0]*X[1], -x[0]])
        A.append([0, 0, 0, X[0], X[1], 1, -x[1]*X[0], -x[1]*X[1], -x[1]])
    
    _, _, VT = np.linalg.svd(A)
    H = VT.T[:, -1] # last column of V
    H /= H[-1]
    H = H.reshape(3, 3)
    
    P = np.hstack([H[:,:2], np.zeros([3,1]), H[:, 2:]])
    proj_2d = project_3d_point(P, pts_3d)
    error = calculate_reprojection_error(pts_2d, proj_2d)
    
    return H, error


# https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/tr98-71.pdf
def zhangs_method(
    H_list: list[np.ndarray]
) -> tuple[np.ndarray, list[np.ndarray]]:
    """
    Estimate K, R, t from multiple homography.
    K[r1 r2 t] = H = [h1 h2 h3].
    eq1: r1^T x r2 = h1^T x K^-T x K^-1 x h2 = 0.
    eq2: ||r1|| = ||r2|| = h1^T x K^-T x K^-1 x h1 = h2^T x K^-T x K^-1 x h2 = 1.
    Let B = K^-T x K^-1,
    eq1: h1^T x B x h2 = 0.
    eq2: h1^T x B x h1 = h2^T x B x h2.
    svd(B).
    B = K^-T x K^-1 = L x L^T

    Args:
        H_list (list[np.ndarray]): list of homography.

    Returns:
        tuple[np.ndarray, list[np.ndarray]]: intrinsic K, list of Rt.
    """
    
    if len(H_list) < 3:
        raise ValueError('Number of homographies is greater than 2')
    
    A = []
    for i in range(len(H_list)):
        H = H_list[i]
        h11 = H[0,0]; h12 = H[1,0]; h13 = H[2,0]
        h21 = H[0,1]; h22 = H[1,1]; h23 = H[2,1]
        
        eq1 = [h11*h21, (h11*h22+h12*h21), (h11*h23+h13*h21), h12*h22, (h12*h23+h13*h22), h13*h23]
        a = [h11*h11, (h11*h12+h12*h11), (h11*h13+h13*h11), h12*h12, (h12*h13+h13*h12), h13*h13]
        b = [h21*h21, (h21*h22+h22*h21), (h21*h23+h23*h21), h22*h22, (h22*h23+h23*h22), h23*h23]
        eq2 = list(np.array(a) - np.array(b))
        
        A.append(eq1)
        A.append(eq2)
    
    # SVD
    _, _, VT = np.linalg.svd(A)
    b = VT.T[:, -1]
    b /= b[-1]
    B = np.array([
        [b[0], b[1], b[2]],
        [b[1], b[3], b[4]],
        [b[2], b[4], b[5]]
    ])
    
    # Cholesky
    L = np.linalg.cholesky(B)
    K_inv = L.T
    K = np.linalg.pinv(K_inv)
    K /= K[2, 2]
    
    Rt_list = []
    for i in range(len(H_list)):
        H = H_list[i]
        h1 = H[:, 0]; h2 = H[:, 1]; h3 = H[:, 2]
        
        r1 = K_inv@h1 / np.linalg.norm(K_inv@h1)
        r2 = K_inv@h2 / np.linalg.norm(K_inv@h2)
        r3 = np.cross(r1, r2)
        R = np.array([r1, r2, r3]).T
        t = K_inv@h3 / np.linalg.norm(K_inv@h1)
        Rt_list.append(np.hstack([R, t.reshape(3,1)]))
        
        print(f'R\n{R}\nt\n{t}\n')
    
    return K, Rt_list


# def apply_radial_distortion(pts_pixel, K, k1, k2):
#     pts_pixel = np.hstack([pts_pixel, np.ones([len(pts_pixel),1])])
#     pts_norm = np.linalg.pinv(K)@pts_pixel.T
#     pts_norm /= pts_norm[-1]
#     pts_norm = pts_norm[:2] # (2, n)
#     r = np.sum(pts_norm**2, axis=0)
        


if __name__ == '__main__':
    n_rows = 7
    n_cols = 10    
    
    H_list = []
    error_list = []
    pts_3d = np.mgrid[0:n_cols, 0:n_rows].T.reshape(-1, 2)
    pts_3d = np.hstack([pts_3d, np.zeros([n_rows*n_cols,1])])
    
    data_path = os.path.join(os.getcwd(), 'data/checkerboard')
    for img_file in os.listdir(data_path):
        img = cv2.imread(os.path.join(data_path, img_file))
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Find corners (top-left -> bottom-right)
        ret, corners = cv2.findChessboardCorners(img_gray, (n_cols,n_rows), None)
        if ret:
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
            corners = cv2.cornerSubPix(img_gray, corners, (n_cols,n_rows), (-1,-1), criteria)
            corners = np.squeeze(corners, axis=1)   # (n, 1, 2) -> (n, 2)
            corners = np.flip(corners, axis=0)  # reverse order
            
            H, error = estimate_homography(pts_3d, corners)
            H_list.append(H)
            print(H, error)
    
    K, Rt_list = zhangs_method(H_list)
            

'''
[h11B11+h12B12+h13B13  h11B12+h12B22+h13B23  h11B13+h12B23+h13B33][h21  h22  h23]
h21(h11B11+h12B12+h13B13)+h22(h11B12+h12B22+h13B23)+h23(h11B13+h12B23+h13B33)
eq1: h11h21*B11 + (h11h22+h12h21)*B12 + (h11h23+h13h21)*B13 + h12h22*B22 + (h12h23+h13h22)*B23 + h13h23*B33 = 0
eq2: h11h11*B11 + (h11h12+h12h11)*B12 + (h11h13+h13h11)*B13 + h12h12*B22 + (h12h13+h13h12)*B23 + h13h13*B33
   = h21h21*B11 + (h21h22+h22h21)*B12 + (h21h23+h23h21)*B13 + h22h22*B22 + (h22h23+h23h22)*B23 + h23h23*B33

'''