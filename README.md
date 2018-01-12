# Vehicle Detection and Tracking

## Task

[//]: # (Image References)

[compare_start_end]: ./output_images/compare_start_end.png "compare_start_end"

In this project, my main goal is to write a software pipeline to **identify and track vehicles** in a video from a front-facing camera on a car (without any false positives).

You will find the exploration code for this project is in the [IPython Notebook](https://github.com/itismouad/vehicle_detection/blob/master/Vehicle%20Detection%20and%20Tracking.ipynb) and a [video](https://github.com/itismouad/vehicle_detection/blob/master/project_video_output.mp4) displaying how my pipeline can allow to detect and track vehicles on the road. A more detailed report of the project is available [here]().

The steps of this project are the following:

* Build a feature engineering pipeline to create a dependent variable for our classification model.
* Train a classifier using the selected features (HOG features and color features in our case).
* Implement a sliding window search.
* Create a video pipeline that create bounding boxes and identify vehicles most of the time with minimal false positives.

## Usage

`video_pipeline.py path_to_input_video path_to_output_video`

Input video needs to be a feed from centered onboard camera.

NB: You will need to download the training data ([vehicle](https://s3.amazonaws.com/udacity-sdc/Vehicle_Tracking/vehicles.zip) and [non-vehicle(https://s3.amazonaws.com/udacity-sdc/Vehicle_Tracking/non-vehicles.zip) images) and put in a folder named `data`. It has been ignored for sizing issues.

## Example

![alt text][compare_start_end]
