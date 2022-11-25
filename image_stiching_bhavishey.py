#!/usr/bin/env python
# coding: utf-8

"""
Authors      : Aditya Jain & Bhavishey Thapar
Date started : November 22, 2022
About        : AER1515 Project; image stitching for drone imagery
"""

import cv2
import numpy as np
from matplotlib import pyplot as plt
from PIL import Image
import glob


def stitch_images(img1, img2, H):
    """
    function for warping/stitching two images using the homography matrix H
    
    Args:
        img1  : first image, or source image
        img2  : second image, that needs to be mapped to the frame of img1
        H     : homography matrix that maps img2 to img1        
        
    Returns:
        output_img: img2 warped to img1 using H
    """
    
    rows1, cols1 = img1.shape[:2]
    rows2, cols2 = img2.shape[:2]

    points_1      = np.float32([[0,0], [0,rows1], [cols1,rows1], [cols1,0]]).reshape(-1,1,2)
    temp_points   = np.float32([[0,0], [0,rows2], [cols2,rows2], [cols2,0]]).reshape(-1,1,2)
    points_2      = cv2.perspectiveTransform(temp_points, H)
    points_concat = np.concatenate((points_1, points_2), axis=0)    

    [x_min, y_min]   = np.int32(points_concat.min(axis=0).ravel() - 0.5)
    [x_max, y_max]   = np.int32(points_concat.max(axis=0).ravel() + 0.5)
    translation_dist = [-x_min,-y_min]
    H_translation    = np.array([[1, 0, translation_dist[0]], [0, 1, translation_dist[1]], [0,0,1]])

    output_img = cv2.warpPerspective(img2, H_translation.dot(H), (x_max-x_min, y_max-y_min))
    frame_size = output_img.shape
    new_image  = img2.shape
    output_img[translation_dist[1]:rows1+translation_dist[1], translation_dist[0]:cols1+translation_dist[0]] = img1
    
    origin_r = int(points_2[0][0][1])
    origin_c = int(points_2[0][0][0])
    
    # if the origin of projected image is out of bounds, then mapping to ()
    if origin_r < 0:
        origin_r = 0
    if origin_c < 0:
        origin_c = 0
        
    # Clipping the new image, if it's size is more than the frame    
    if new_image[0] > frame_size[0]-origin_r:
        img2 = img2[0:frame_size[0]-origin_r,:]
        
    if new_image[1] > frame_size[1]-origin_c:
        img2 = img2[:,0:frame_size[1]-origin_c]    
            
    output_img[origin_r:new_image[0]+origin_r, origin_c:new_image[1]+origin_c] = img2
    return output_img


def build_mosaic(raw_image_list, num_imgs_to_use, mosaic_name, 
                 num_featues=1000, reproj_thresh=5.0):
    """
    main function for image stitching 
    
    Args:
        raw_image_list  : sorted list of raw image paths to be stitched
        num_imgs_to_use : number of images to build a mosaic for
        mosaic_name     : name of the final sitched image
        num_featues     : number of keypoints to extract from feature detector; optional; default=1000
        reproj_thresh   : reprojection error threshold in pixels; optional; default=5.0
        
        
    Returns:
        avg_repro_error : list of average reprojection error
        matches         : list of number of good matches
    """
    
    avg_repro_error = []  # list of average reprojection error
    matches_list    = []  # list of number of good matches at every stage
    sift            = cv2.SIFT_create(num_featues)
    
    # starting out with first image
    first_image   = cv2.imread(raw_image_list.pop(0))
    height, width = first_image.shape[:2]
    first_image   = cv2.resize(first_image, (int(width/4), int(height/4)))
    final_mosaic  = first_image
    cv2.imwrite(mosaic_name, final_mosaic)

    while raw_image_list:

        image = cv2.imread(raw_image_list.pop(0))              
        height, width = image.shape[:2]        
        image = cv2.resize(image, (int(width/4), int(height/4)))

        # Find the features
        orb=cv2.ORB_create(num_featues)
        kp1, des1 = orb.detectAndCompute(final_mosaic,None)
        kp2, des2 = orb.detectAndCompute(image,None)
        
        # Match the features
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = bf.match(des1,des2)
        matches = sorted(matches, key = lambda x:x.distance)
        src_pts=np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1,1,2)
        dst_pts=np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1,1,2)
        M, mask = cv2.findHomography(dst_pts, src_pts, cv2.RANSAC, reproj_thresh)
        
        final_mosaic = stitch_images(final_mosaic, image, M)
        cv2.imwrite(mosaic_name, final_mosaic) 
        
    return avg_repro_error, matches_list

num_imgs_to_use = 1
raw_image_list  = sorted(glob.glob('./raw_images/*.JPG'))[:num_imgs_to_use+1]
mosaic_name     = 'chandigarh_20images.png'

if __name__=='__main__':
    _, _ = build_mosaic(raw_image_list, num_imgs_to_use, mosaic_name)

cv2.imshow('mosaic', cv2.imread(mosaic_name))
cv2.waitKey(0)

