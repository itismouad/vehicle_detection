# Vehicle Detection and Tracking

Identification and tracking of vehicles.

## Installation

```
conda env create -f environment.yml
source activate environment
```

## Usage

```
 Usage:
  video_pipeline.py [-i <file>] [-o <file>]
  video_pipeline.py -h | --help

Options:
  -h --help
  -i <file> --input <file>   Input text file [default: ../videos/project_video.mp4]
  -o <file> --output <file>  Output generated file [default: ../videos/project_video_output.mp4]
```


NB:
* Input video needs to be a feed from centered onboard camera.
* You will need to download the training data ([vehicle](https://s3.amazonaws.com/udacity-sdc/Vehicle_Tracking/vehicles.zip) and [non-vehicle](https://s3.amazonaws.com/udacity-sdc/Vehicle_Tracking/non-vehicles.zip) images) and put in a folder named `data`. It has been ignored for sizing issues.

## Example

```
python video_pipeline.py-i ../videos/project_video.mp4 -o ../videos/project_video_output.mp4
```

![alt text][compare_start_end]

## Detailed description

[//]: # (Image References)

[compare_start_end]: ./output_images/compare_start_end.png "compare_start_end"

In this project, my main goal is to write a software pipeline to **identify and track vehicles** in a video from a front-facing camera on a car (without any false positives).

You will find the exploration code for this project is in the [IPython Notebook](https://github.com/itismouad/vehicle_detection/blob/master/notebooks/Vehicle%20Detection%20and%20Tracking.ipynb) and a [video](https://github.com/itismouad/vehicle_detection/blob/master/videos/project_video_output.mp4) displaying how my pipeline can allow to detect and track vehicles on the road. A more detailed report of the project is available [here]().

The steps of this project are the following:

* Build a feature engineering pipeline to create a dependent variable for our classification model.
* Train a classifier using the selected features (HOG features and color features in our case).
* Implement a sliding window search and use the trained classifier to search for vehicles in images
* Create a video pipeline that create bounding boxes and identify vehicles most of the time with minimal false positives.
