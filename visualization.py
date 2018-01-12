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
import matplotlib.pyplot as plt
import matplotlib.image as mpimg


class Visualize():

	def compare_images(self, two_images, two_labels, figsize=(10, 10)):

	    fig, axes = plt.subplots(ncols=2, figsize=figsize)
	    axes[0].imshow(two_images[0])
	    axes[0].set_title(two_labels[0])
	    axes[1].imshow(two_images[1])
	    axes[1].set_title(two_labels[1])



	def plot_figures(self, series, n_row=1, n_col=2, figsize=(15, 5), cmap=None, save=False, filename=''):

	    fig, ax = plt.subplots(n_row, n_col, figsize=figsize)
	    n_images = n_row*n_col
	    series = series[:n_images]
	    
	    for i, serie in enumerate(series):
	        plt.subplot(n_row, n_col, i+1)
	        plt.plot(serie)
	        plt.xticks([])
	        plt.yticks([])   
	        if save:
	            plt.savefig(os.path.join(output_path, filename + '_' + str(i) + '.png'))
	    plt.tight_layout(pad=0, h_pad=0, w_pad=0)
	    plt.show()


	def show_images(self, images, n_row=1, n_col=2, figsize=(12, 3), cmap=None, save=False, filename=''):

	    fig, ax = plt.subplots(n_row, n_col, figsize=figsize)
	    n_images = n_row*n_col
	    images = images[:n_images]
	    
	    for i, image in enumerate(images):
	        plt.subplot(n_row, n_col, i+1)
	        plt.imshow(image) if cmap is None else plt.imshow(image, cmap=cmap)
	        plt.xticks([])
	        plt.yticks([])   
	        if save:
	            plt.savefig(os.path.join(output_path, filename + '_' + str(i) + '.png'))
	    plt.tight_layout(pad=0, h_pad=0, w_pad=0)
	    plt.show()




	def plot_hist(self, channel1_hist, channel2_hist, channel3_hist, bin_centers, figsize=(15, 5)):
		
	    fig = plt.figure(figsize=figsize)
	    plt.subplot(131)
	    plt.bar(bin_centers, channel1_hist[0])
	    plt.xlim(0, 256)
	    plt.title('Channel 1 Histogram')
	    plt.subplot(132)
	    plt.bar(bin_centers, channel2_hist[0])
	    plt.xlim(0, 256)
	    plt.title('Channel 2 Histogram')
	    plt.subplot(133)
	    plt.bar(bin_centers, channel3_hist[0])
	    plt.xlim(0, 256)
	    plt.title('Channel 3 Histogram')
	    fig.tight_layout()