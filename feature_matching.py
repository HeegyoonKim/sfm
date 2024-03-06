import cv2
import numpy as np
import logging


def build_matching_table(img_list, K):
    logging.info('')
    
    sift = cv2.SIFT_create()
    n_frames = len(img_list)
    table = None
    indices_list = []
    for i in range(n_frames - 1):
        img1 = img_list[i]
        img2 = img_list[i+1]
        new_img_idx = i + 2
        
        # If bgr, convert to gray
        if img1.ndim == 3:
            img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
            img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
        
        # Extract SIFT features  
        kp1, des1 = sift.detectAndCompute(img1, None)
        kp2, des2 = sift.detectAndCompute(img2, None)
		
        # Match features
        pts1, pts2 = match_features(kp1, des1, kp2, des2, img1, img2, K, False)  # (n, 2), (n, 2)
        
        if table is None:  # first two images
            table = np.array((pts1, pts2))
            indices_list.append(np.arange(len(pts1)))
            logging.info('\nBuild table from the first two images\n'
                        f'Table: {table.shape}  (n_frames, n_points, xy)\n')
        else:
            # Common points in new image
            _, tab_indices, pt_indices = np.intersect1d(table[i, :, 0], pts1[:, 0], return_indices=True)
            assert np.array_equal(table[i, tab_indices, :], pts1[pt_indices, :])
            
            n_common_pts = len(table[i])
            new_tab1 = np.zeros((1, n_common_pts, 2)) - 1  # -1 if not exist
            new_tab1[0, tab_indices, :] = pts2[pt_indices, :]
            
            # Update table
            table = np.vstack((table, new_tab1))
            logging.info(f'\n{len(pt_indices)} common points are found in image{new_img_idx}\n'
                         f'New table: {table.shape}')
            
            # New points in new image
            new_pt_indices = np.delete(np.arange(len(pts1)), pt_indices)
            n_new_pts = len(new_pt_indices)
            new_pts1 = pts1[new_pt_indices]
            new_pts2 = pts2[new_pt_indices]
            
            new_tab2 = np.zeros((new_img_idx, n_new_pts, 2)) - 1
            new_tab2[-2, :, :] = new_pts1
            new_tab2[-1, :, :] = new_pts2  # new view
            
            # Update table
            table = np.hstack((table, new_tab2))
            
            # Save matches & new points indices
            new_pt_indices = np.arange(n_new_pts) + n_common_pts
            indices_list.append([tab_indices, new_pt_indices])
            logging.info(f'\n{n_new_pts} new points are added to table\n'
                         f'New table: {table.shape}\n')
    
    # Visibility matrix
    visibility_matrix = np.zeros((n_frames, table.shape[1]), dtype=np.uint8)
    row, col = np.where(table[:, :, 0] != -1)
    for r, c in zip(row, col):
        visibility_matrix[r, c] = 1
    
    print('')
    return table, indices_list, visibility_matrix


def match_features(kp1, des1, kp2, des2, img1, img2, K, show_matches=False):
    # FLANN parameters
    FLANN_INDEX_KDTREE = 1
    idx_params = {'algorithm': FLANN_INDEX_KDTREE, 'tree':5}
    search_params = {'checks': 100}
    flann = cv2.FlannBasedMatcher(idx_params, search_params)
    matches = flann.knnMatch(des1, des2, k=2)
    matches_mask = [[0,0] for _ in range(len(matches))]
    pts1 = []
    pts2 = []            
    
    # ratio test as per Lowe's paper
    for i, (m, n) in enumerate(matches):
        if m.distance < 0.9*n.distance:
            matches_mask[i] = [1, 0]
            pts2.append(kp2[m.trainIdx].pt)
            pts1.append(kp1[m.queryIdx].pt)
    
    pts1 = np.array(pts1)
    pts2 = np.array(pts2)
    
    # Remove outlier
    _, mask = cv2.findEssentialMat(pts1, pts2, K, method=cv2.RANSAC, prob=0.999, threshold=2.0)
    pts1 = pts1[mask.ravel()==1]    # shape:(n, 2), dtype:np.float64
    pts2 = pts2[mask.ravel()==1]
        
    if show_matches:
        show_img(draw_matches(img1, pts1, img2, pts2))

    return pts1, pts2


def show_img(img):
    import matplotlib.pyplot as plt
    plt.imshow(img)
    plt.show()
    plt.close()
    return None


def draw_matches(img1, pts1, img2, pts2, color_circle=(0,255,0), color_line=(0,0,255)):
    W = img1.shape[1]
    img_out = np.concatenate((img1,img2), axis=1)
    img_out = cv2.cvtColor(img_out, cv2.COLOR_GRAY2BGR)

    for pt1, pt2 in zip(pts1, pts2):
        x1, y1 = int(pt1[0]), int(pt1[1])
        x2, y2 = int(pt2[0]) + W, int(pt2[1])
        
        # Draw circles and lines
        cv2.circle(img_out, (x1,y1), 10, color_circle, -1)
        cv2.circle(img_out, (x2,y2), 10, color_circle, -1)
        cv2.line(img_out, (x1,y1), (x2,y2), color_line, 5)
    
    return img_out