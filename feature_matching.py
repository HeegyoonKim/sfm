import cv2
import numpy as np
import logging


def build_matching_table(img_list, K):
    logging.info('')
    
    sift = cv2.SIFT_create()
    n_frames = len(img_list)
    table = None
    indices_list = []
    
    # First pair
    img1 = img_list[0]
    img2 = img_list[1]
    
    # If bgr, convert to gray
    if img1.ndim == 3:
        img1 == cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        img2 == cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    
    # Extract SIFT features
    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)
    
    
    

def match_features(img1, img2):
    pass