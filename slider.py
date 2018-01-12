# -*- coding: utf-8 -*-
# encoding=utf8

"""
 Written by Mouad Hadji (@itismouad)
"""

import os
import sys
import glob
import cv2
import time
from tqdm import tqdm
import pandas as pd
import numpy as np
from skimage.feature import hog

from feature_engineering import FeatureExtraction


class Slider():
    
    def __init__(self, FeatureExtraction, search_params):
        self.FE = FeatureExtraction
        self.xy_window = (search_params["xy_window"][0], search_params["xy_window"][1]) 
        self.window_sizes = search_params["window_sizes"]
        self.xy_overlap = (search_params["xy_overlap"][0], search_params["xy_overlap"][1]) 
        self.x_start_stop = search_params["x_start_stop"]
        self.y_start_stop = search_params["y_start_stop"]
        
    
    # Draw boxes on an image
    def draw_boxes(self, img, bboxes, color=(0, 0, 255), thick=6):
        imcopy = np.copy(img)
        for bbox in bboxes:
            cv2.rectangle(imcopy, bbox[0], bbox[1], color, thick)
        return imcopy
    
    
    # Create list of windows
    def slide_window(self, img,
                     # start and stop positions in both x and y
                     x_start_stop=[None, None], y_start_stop=[None, None],
                     # window size (x and y dimensions), and overlap fraction (for both x and y)
                     xy_window=(64, 64), xy_overlap=(0.5, 0.5)):
        # If x and/or y start/stop positions not defined, set to image size
        if x_start_stop[0] == None:
            x_start_stop[0] = 0
        if x_start_stop[1] == None:
            x_start_stop[1] = img.shape[1]
        if y_start_stop[0] == None:
            y_start_stop[0] = 0
        if y_start_stop[1] == None:
            y_start_stop[1] = img.shape[0]
        # Compute the span of the region to be searched    
        xspan = x_start_stop[1] - x_start_stop[0]
        yspan = y_start_stop[1] - y_start_stop[0]
        # Compute the number of pixels per step in x/y
        nx_pix_per_step = np.int(xy_window[0]*(1 - xy_overlap[0]))
        ny_pix_per_step = np.int(xy_window[1]*(1 - xy_overlap[1]))
        # Compute the number of windows in x/y
        nx_buffer = np.int(xy_window[0]*(xy_overlap[0]))
        ny_buffer = np.int(xy_window[1]*(xy_overlap[1]))
        nx_windows = np.int((xspan-nx_buffer)/nx_pix_per_step) 
        ny_windows = np.int((yspan-ny_buffer)/ny_pix_per_step) 
        # Initialize a list to append window positions to
        window_list = []
        # Loop through finding x and y window positions
        # Note: you could vectorize this step, but in practice
        # you'll be considering windows one by one with your
        # classifier, so looping makes sense
        for ys in range(ny_windows):
            for xs in range(nx_windows):
                # Calculate window position
                startx = xs*nx_pix_per_step + x_start_stop[0]
                endx = startx + xy_window[0]
                starty = ys*ny_pix_per_step + y_start_stop[0]
                endy = starty + xy_window[1]
                # Append window position to list
                window_list.append(((startx, starty), (endx, endy)))
        # Return the list of windows
        return window_list
    
    
    # Define a function you will pass an image and the list of windows to be searched (output of slide_windows())
    def search_windows(self, img, windows, clf, scaler):

        #1) Create an empty list to receive positive detection windows
        on_windows = []
        #2) Iterate over all windows in the list
        for window in windows:
            #3) Extract the test window from original image
            test_img = cv2.resize(img[window[0][1]:window[1][1], window[0][0]:window[1][0]], (64, 64))      
            #4) Extract features for that window using single_img_features()
            features = self.FE.extract_features(test_img)
            #5) Scale extracted features to be fed to classifier
            test_features = scaler.transform(np.array(features).reshape(1, -1))
            #6) Predict using your classifier
            prediction = clf.predict(test_features)
            #7) If positive (prediction == 1) then save the window
            if prediction == 1:
                on_windows.append(window)
        #8) Return windows for positive detections
        return on_windows
    
    
    def run(self, img, clf, scaler, x_start_stop=[None, None], y_start_stop = [360, 700], xy_window=(64, 64), xy_overlap = (0.85, 0.85)):
        windows = self.slide_window(img, y_start_stop=y_start_stop, xy_window=xy_window, xy_overlap=xy_overlap)
        on_windows = self.search_windows(img, windows, clf, scaler)
        img_withboxes = self.draw_boxes(img.copy(), on_windows)
        return on_windows, img_withboxes
    
    
    def run_new(self, img, clf, scaler, window_sizes, x_start_stop=[None, None], y_start_stop = [360, 700], xy_overlap = (0.85, 0.85)):
        windows = []
        for window_size in window_sizes:
            xy_window = (window_size, window_size)
            current_windows = self.slide_window(img, y_start_stop=y_start_stop, xy_window=xy_window, xy_overlap=xy_overlap)
            windows += current_windows
        on_windows = self.search_windows(img, windows, clf, scaler)
        img_withboxes = self.draw_boxes(img.copy(), on_windows)
        return on_windows, img_withboxes



    def run_efficient(self, img, clf, scaler, y_start_stop=[350, 656], window=64, cells_per_step=1, scale=1.5):

	    feature_image = self.FE.convert_img(img, color_space=self.FE.cspace)
	    
	    ystart, ystop = y_start_stop
	    ctrans_tosearch = feature_image[ystart:ystop,:,:]
	    
	    if scale != 1:
	        imshape = ctrans_tosearch.shape
	        ctrans_tosearch = cv2.resize(ctrans_tosearch, (np.int(imshape[1]/scale), np.int(imshape[0]/scale)))
	    
	    if self.FE.hog_channel == 'ALL':
	        ch1 = ctrans_tosearch[:,:,0]
	        ch2 = ctrans_tosearch[:,:,1]
	        ch3 = ctrans_tosearch[:,:,2]
	    else:
	        ch1 = ctrans_tosearch[:,:,0]
	    
	    # Define blocks and steps as above
	    nxblocks = (ch1.shape[1] // self.FE.pix_per_cell) - self.FE.cell_per_block + 1
	    nyblocks = (ch1.shape[0] // self.FE.pix_per_cell) - self.FE.cell_per_block + 1 
	    nfeat_per_block = self.FE.orient*self.FE.cell_per_block**2
	    
	    nblocks_per_window = (window // self.FE.pix_per_cell) - self.FE.cell_per_block + 1
	    nxsteps = (nxblocks - nblocks_per_window) // cells_per_step
	    nysteps = (nyblocks - nblocks_per_window) // cells_per_step
	    
	    # Compute individual channel HOG features for the entire image
	    if self.FE.hog_channel == 'ALL':
	        hog1 = self.FE.get_hog_features(ch1, self.FE.orient, self.FE.pix_per_cell, self.FE.cell_per_block, feature_vec=False)
	        hog2 = self.FE.get_hog_features(ch2, self.FE.orient, self.FE.pix_per_cell, self.FE.cell_per_block, feature_vec=False)
	        hog3 = self.FE.get_hog_features(ch3, self.FE.orient, self.FE.pix_per_cell, self.FE.cell_per_block, feature_vec=False)
	    else:
	        hog1 = self.FE.get_hog_features(ch1, self.FE.orient, self.FE.pix_per_cell, self.FE.cell_per_block, feature_vec=False)
	    
	    car_windows = []
	    
	    for xb in range(nxsteps):
	        for yb in range(nysteps):
	            ypos = yb*cells_per_step
	            xpos = xb*cells_per_step
	            
	            # Extract HOG for this patch
	            if self.FE.hog_channel == 'ALL':
	                hog_feat1 = hog1[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel() 
	                hog_feat2 = hog2[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel() 
	                hog_feat3 = hog3[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel() 
	                hog_features = np.hstack((hog_feat1, hog_feat2, hog_feat3))
	            else:
	                hog_feat1 = hog1[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel() 
	                hog_features = hog_feat1
	            
	            xleft = xpos*self.FE.pix_per_cell
	            ytop = ypos*self.FE.pix_per_cell
	            
	            # Extract the image patch
	            subimg = cv2.resize(ctrans_tosearch[ytop:ytop+window, xleft:xleft+window], (64,64))
	          
	            # Get color features
	            spatial_features = self.FE.bin_spatial(subimg, size=self.FE.size)
	            _, _, _, _, hist_features = self.FE.color_hist(subimg, nbins=self.FE.hist_bins, bins_range=self.FE.hist_range)
	            
	            # Scale features and make a prediction
	            test_features = scaler.transform(np.hstack((spatial_features, hist_features, hog_features)).reshape(1, -1))    
	            test_prediction = clf.predict(test_features)
	            
	            if test_prediction == 1:
	                xbox_left = np.int(xleft*scale)
	                ytop_draw = np.int(ytop*scale)
	                win_draw = np.int(window*scale)
	                car_windows.append(((xbox_left, ytop_draw+ystart),(xbox_left+win_draw,ytop_draw+win_draw+ystart)))
	                
	    return car_windows