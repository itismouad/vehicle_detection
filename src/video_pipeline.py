# -*- coding: utf-8 -*-
# encoding=utf8


"""Identification and tracking of vehicles. Written by Mouad Hadji (@itismouad)

 Usage:
  video_pipeline.py [-i <file>] [-o <file>]
  video_pipeline.py -h | --help

Options:
  -h --help
  -i <file> --input <file>   Input text file [default: ../videos/project_video.mp4]
  -o <file> --output <file>  Output generated file [default: ../videos/project_video_output.mp4]

"""

import os, sys
import glob
import cv2
import json
import time
from docopt import docopt
from tqdm import tqdm
from moviepy.editor import VideoFileClip
import pandas as pd
import numpy as np

from skimage.feature import hog
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC, SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split, GridSearchCV

from feature_engineering import FeatureExtraction
from slider import Slider
from heater import Heater
from visualization import Visualize


class Train():

    def __init__(self, FeatureExtraction, data_path):
        self.FE = FeatureExtraction
        self.vehicles_path = glob.glob(os.path.join(data_path, "vehicles/**/*.png"))
        self.non_vehicles_path = glob.glob(os.path.join(data_path, "non-vehicles/**/*.png"))


    def load_data(self):
        vehicles = [cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB) for img_path in self.vehicles_path]
        non_vehicles = [cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB) for img_path in self.non_vehicles_path]
        return vehicles, non_vehicles

    def preprocess_data(self, vehicles, non_vehicles):
        vehicles_features = [self.FE.extract_features(img) for img in vehicles]
        nonvehicles_features = [self.FE.extract_features(img) for img in non_vehicles]

        y_vehicles = list(np.ones(len(vehicles_features)))
        y_non_vehicles = list(np.zeros(len(nonvehicles_features)))

        X = np.vstack((vehicles_features, nonvehicles_features)).astype(np.float64)
        y = np.hstack((y_vehicles, y_non_vehicles))

        # Fit a per-column scaler
        X_scaler = StandardScaler().fit(X)
        # Apply the scaler to X
        scaled_X = X_scaler.transform(X)

        return scaled_X, y, X_scaler

    def train_model(self, X_train, X_test, y_train, y_test, sklearn_model):
        """
        Trains the classifier `sklearn_model`. 
        """
        sklearn_model.fit(X_train, y_train)
        accuracy = sklearn_model.score(X_test, y_test)
        
        return sklearn_model, accuracy

    def run(self):
        print('Training model and scaler...')
        rand_state = np.random.randint(0, 100)
        vehicles, non_vehicles = self.load_data()
        scaled_X, y, X_scaler = self.preprocess_data(vehicles, non_vehicles)
        X_train, X_test, y_train, y_test = train_test_split(scaled_X, y, test_size=0.2, random_state=rand_state)
        rf_raw = RandomForestClassifier(n_estimators=50, min_samples_split=7, min_samples_leaf=1)
        rf, accuracy_rf = self.train_model(X_train, X_test, y_train, y_test, rf_raw)
        print("Accuracy of model : ", accuracy_rf)
        print('End of training...')
        return rf, X_scaler



class videoPipeline():

    def __init__(self, data_path, params, search_params, threshold):
        # initialize helpers class
        self.FE = FeatureExtraction(params)
        self.Tr = Train(self.FE, data_path)
        self.SL = Slider(self.FE, search_params)
        self.Hr = Heater(threshold)

        # initialize the model and scaler
        self.clf, self.scaler = self.Tr.run()

    def process_image(self, img, draw=True):
        current_windows = self.SL.run_efficient(img, self.clf, self.scaler)
        draw_img, heatmap = self.Hr.run(img, current_windows)
        result = draw_img if draw else heatmap
        return draw_img

    def run(self, input_video, output_video):
        print('Starting video processing...')
        raw_clip = VideoFileClip(input_video)
        processed_clip = raw_clip.fl_image(self.process_image)
        processed_clip.write_videofile(output_video, audio=False)
        print('End of video processing...')



if __name__ == "__main__":

    parameters = docopt(__doc__)
    input_file, output_file = parameters['--input'], parameters['--output']
    print("Current input file: " , input_file)
    print("Current output file: " , output_file)

    data_path = "../data"
    config_path = "../config"

    with open(os.path.join(config_path, 'config.json')) as json_data_file:
        config_data = json.load(json_data_file)
        params = config_data["feature"]
        search_params = config_data["search"]

    vP = videoPipeline(data_path, params, search_params, threshold=4)
    vP.run(input_file, output_file)
