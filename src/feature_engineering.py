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


class FeatureExtraction():
    
    def __init__(self, params):
        self.size = (params["SPATIAL_SIZE"][0], params["SPATIAL_SIZE"][1])
        self.hist_bins = params["HIST_BINS"]
        self.hist_range = (params["HIST_RANGE"][0], params["HIST_RANGE"][1])
        self.cspace = params["COLOR_SPACE"]
        self.orient = params["ORIENT"]
        self.pix_per_cell = params["PIX_PER_CELL"]
        self.cell_per_block = params["CELL_PER_BLOCK"]
        self.hog_channel = params["HOG_CHANNEL"]
        self.spatial_feat = params["SPAT_FEAT"]
        self.hist_feat = params["HIST_FEAT"]
        self.hog_feat = params["HOG_FEAT"]

        
    def convert_img(self, img, color_space="RGB"):
        convert_dict = {'HLS':cv2.COLOR_RGB2HLS, 'HSV':cv2.COLOR_RGB2HSV, 'LUV':cv2.COLOR_RGB2LUV,
                        'YUV':cv2.COLOR_RGB2YUV, 'YCrCb':cv2.COLOR_RGB2YCrCb}
        if color_space!='RGB':
            return cv2.cvtColor(img, convert_dict[color_space])
        else:
            return np.copy(img) 

    
    def bin_spatial(self, img, size=(32, 32)):
        features = cv2.resize(img, size).ravel() 
        return features

    
    
    def color_hist(self, img, nbins=32, bins_range=(0, 256)):
        # Compute the histogram of the color channels separately
        channel1_hist = np.histogram(img[:,:,0], bins=nbins, range=bins_range)
        channel2_hist = np.histogram(img[:,:,1], bins=nbins, range=bins_range)
        channel3_hist = np.histogram(img[:,:,2], bins=nbins, range=bins_range)
        
        ## generating bin centers
        bin_edges = channel1_hist[1]
        bin_centers = (bin_edges[1:]  + bin_edges[0:len(bin_edges)-1])/2
        
        # Concatenate the histograms into a single feature vector
        hist_features = np.concatenate((channel1_hist[0], channel2_hist[0], channel3_hist[0]))
    
        # Return the individual histograms, bin_centers and feature vector
        return channel1_hist, channel2_hist, channel3_hist, bin_centers, hist_features
    
    
    
    def get_hog_features(self, img, orient=None, pix_per_cell=None, cell_per_block=None, vis=False, feature_vec=True):
        
        orient = self.orient if orient is None else orient
        pix_per_cell = self.pix_per_cell if pix_per_cell is None else pix_per_cell
        cell_per_block = self.cell_per_block if cell_per_block is None else cell_per_block
                
        # Call with two outputs if vis==True
        if vis == True:
            features, hog_image = hog(img, orientations=orient, pixels_per_cell=(pix_per_cell, pix_per_cell),
                                      cells_per_block=(cell_per_block, cell_per_block), transform_sqrt=True, 
                                      visualise=vis, feature_vector=feature_vec)
            return features, hog_image
        # Otherwise call with one output
        else:      
            features = hog(img, orientations=orient, pixels_per_cell=(pix_per_cell, pix_per_cell),
                           cells_per_block=(cell_per_block, cell_per_block), transform_sqrt=True, 
                           visualise=vis, feature_vector=feature_vec)
            return features

        
    def extract_features(self, img):
        
        feature_image = self.convert_img(img, color_space=self.cspace)
        features = []
                
        if self.spatial_feat:
            # Apply bin_spatial() to get spatial color features
            spatial_features = self.bin_spatial(feature_image, self.size)
            features.append(spatial_features)

        if self.hist_feat:
            # Apply color_hist() to get histogram of colors
            _, _, _, _, hist_features = self.color_hist(feature_image, self.hist_bins, self.hist_range)
            features.append(hist_features)
            
        if self.hog_feat:
            # Call get_hog_features() with vis=False, feature_vec=True
            if self.hog_channel == 'ALL':
                hog_features = []
                for channel in range(feature_image.shape[2]):
                    hog_features.append(self.get_hog_features(feature_image[:,:,channel], 
                                        orient=self.orient, pix_per_cell=self.pix_per_cell, cell_per_block=self.cell_per_block, 
                                        vis=False, feature_vec=True))
                hog_features = np.ravel(hog_features)        
            else:
                hog_features = self.get_hog_features(feature_image[:,:,self.hog_channel],
                                                     orient=self.orient, pix_per_cell=self.pix_per_cell, cell_per_block=self.cell_per_block,
                                                     vis=False, feature_vec=True)
            features.append(hog_features)

        return np.concatenate(features)