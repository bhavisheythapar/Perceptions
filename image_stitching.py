#!/usr/bin/env python
# coding: utf-8

"""
Authors      : Aditya Jain & Bhavishey Thapar
Date started : November 22, 2022
About        : AER1515 Project project; image stitching for drone imagery
"""

import cv2
import numpy as np
from matplotlib import pyplot as plt
import glob
import time
import os

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
    matches         = []  # list of number of good matches at every stage
    sift            = cv2.SIFT(num_featues)
    
    # starting out with first image
    first_image   = cv2.imread(raw_image_list[0])
    height, width = first_image.shape[:2]
    first_image   = cv2.resize(first_image, (int(width/4), int(height/4)))
    final_mosaic  = first_image
    img_count     = 0
    
    for img_name in raw_image_list[1:]:
        image = cv2.imread(img_name)              
        height, width = image.shape[:2]        
        image = cv2.resize(image, (int(width/4), int(height/4)))

        # Find the features
        kp1, des1 = sift.detectAndCompute(final_mosaic,None)  
        kp2, des2 = sift.detectAndCompute(image,None)
        
        # Feature matching      
        FLANN_INDEX_KDTREE = 0
        index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
        search_params = dict(checks = 50)
        flann = cv2.FlannBasedMatcher(index_params, search_params)
        matches = flann.knnMatch(des1,des2,k=2)        
        
        # Store all good matches as per Lowe's ratio test
        good       = []
        all_points = []
        for m,n in matches:
            if m.distance < 0.7*n.distance:
                good.append(m)                
            all_points.append(m)        
        matches.append(len(good))        
        
        # Find homography
        src_pts = np.float32([ kp1[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
        dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good ]).reshape(-1,1,2)        
        all_src_pts = np.float32([ kp1[m.queryIdx].pt for m in allPoints ]).reshape(-1,1,2)
        all_dst_pts = np.float32([ kp2[m.trainIdx].pt for m in allPoints ]).reshape(-1,1,2)        
        M, mask = cv2.findHomography(dst_pts, src_pts, cv2.RANSAC,reproj_thresh)
        
        # Find the euclidean distance error
        src_pts      = np.array(src_pts)    
        dst_pts      = np.array(dst_pts)
        dst_pts      = np.reshape(dst_pts, (len(dst_pts), 2))
        ones         = np.ones(len(src_pts))    
        test_pts     = np.transpose(np.reshape(src_pts, (len(src_pts), 2)))
        test_pts_hom = np.vstack((test_pts, ones))  
        ## projecting the points in test image to collage image using homography matrix
        projected_pts_H  = np.matmul(M, test_pts_hom)      
        projected_pts_nH = np.transpose(np.array([np.true_divide(projected_pts_H[0,:], projected_pts_H[2,:]),                                                   np.true_divide(projected_pts_H[1,:], projected_pts_H[2,:])]))        
        error     = int(np.sum(np.linalg.norm(projected_pts_nH-dst_pts, axis=1)))
        avg_error = np.divide(np.array(error), np.array(len(src_pts)))     
        avg_repro_error.append(avg_error)
        
        # Apply homography to current image and obtain the resultant mosaic
        final_mosaic = stitch_images(final_mosaic, image, M)
        cv2.imwrite(mosaic_name, final_mosaic)        
        
        img_count += 1        
        if img_count==num_imgs_to_use:
            break
    
    return avg_repro_error, matches

raw_image_list  = sorted(glob.glob('./raw_images/*.JPG'))
num_imgs_to_use = 20
mosaic_name     = 'chandigarh_20images.png'

if __name__=='__main__':
    _, _ = build_mosaic(raw_image_list, num_imgs_to_use, mosaic_name)





