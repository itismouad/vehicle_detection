# Vehicle Detection

[//]: # (Image References)

[compare_start_end]: ./output_images/compare_start_end.png "compare_start_end"
[vehicles_raw]: ./output_images/vehicles_raw.png "vehicles_raw"
[non_vehicles_raw]: ./output_images/non_vehicles_raw.png "non_vehicles_raw"
[vehicles_spatbin]: ./output_images/vehicles_spatbin.png "vehicles_spatbin"
[non_vehicles_spatbin]: ./output_images/non_vehicles_spatbin.png "non_vehicles_spatbin"
[vehicles_hist]: ./output_images/vehicles_hist.png "vehicles_hist"
[non_vehicles_hist]: ./output_images/non_vehicles_hist.png "non_vehicles_hist"
[vehicles_hog]: ./output_images/vehicles_hog.png "vehicles_hog"
[non_vehicles_hog]: ./output_images/non_vehicles_hog.png "non_vehicles_hog"
[test_images_tosearch]: ./output_images/test_images_tosearch.png "test_images_tosearch"
[test_images_tosearch_boxed]: ./output_images/test_images_tosearch_boxed.png "test_images_tosearch_boxed"
[test_images_tosearch_hot]: ./output_images/test_images_tosearch_hot.png "test_images_tosearch_hot"
[test_images_tosearch_unique]: ./output_images/test_images_tosearch_unique.png "test_images_tosearch_unique"
[]: ./output_images/.png ""
[]: ./output_images/.png ""
[]: ./output_images/.png ""
[]: ./output_images/.png ""
[]: ./output_images/.png ""



## Introduction

In this project, my main goal is to write a software pipeline to **identify and track vehicles** in a video from a front-facing camera on a car (without any false positives).

You will find the exploration code for this project is in the [IPython Notebook](https://github.com/itismouad/vehicle_detection/blob/master/Vehicle%20Detection%20and%20Tracking.ipynb) and a [video](https://github.com/itismouad/vehicle_detection/blob/master/project_video_output.mp4) displaying how my pipeline can allow to detect and track vehicles on the road.

![alt text][compare_start_end]

For this purpose, I will aim at achieving the following steps :

* Build a feature engineering pipeline to create a dependent variable for our classification model.
* Train a classifier using the selected features (HOG features and color features in our case).
* Implement a sliding window search.
* Create a video pipeline that create bounding boxes and identify vehicles most of the time with minimal false positives.


## Feature Engineering

* code : [feature_engineering.py](https://github.com/itismouad/vehicle_detection/blob/master/feature_engineering.py)
* name of the python class :  **FeatureExtraction()**

The images I receive as an input are coming from the forward facing camera. The main goal of this software pipeline is to detect vehicles hence we need to build a feature vectors that will allow to perform this task. There are many features that we can consider extracting from an image.

Let's first take a look at the training images we have in our possession. The images are compiled from both the [GTI vehicle image database](http://www.gti.ssr.upm.es/data/Vehicle_database.html) and the [KITTI vision benchmark suite](http://www.cvlibs.net/datasets/kitti/).

Images of `vehicles` :

![alt text][vehicles_raw]

Images of `non_vehicles` :

![alt text][non_vehicles_raw]

* Vehicle train images count: `8792`
* Non-vehicle train image count: `8968`


Let's see how we can train a machine learning algorithm to distinguish the feaures of vehicles versus non-vehicles.

### Spatial Color Binning

Raw pixel values can be quite useful to include in our feature vector in searching for cars. While it could be cumbersome to include three color channels of a full resolution image, one can perform spatial binning on an image and still retain enough information to help in finding vehicles.

We use the `bin_spatial` function for this purpose :

```Python
def bin_spatial(self, img, size=(32, 32)):
	features = cv2.resize(img, size).ravel()
	return features
```


Result for a sample of vehicle images : 

![alt text][vehicles_spatbin]

Result for a sample of non-vehicle images :

![alt text][non_vehicles_spatbin]

Even if it is difficult to notice the difference between the two samples on these graphs, the upcoming result will comfort our choice to keep this simple feature vector when creating the final feature vector.


### Histograms of Colors

Another way to use color features that is more flexible than template matching or spatial binning is to use histogram of colors. Indeed, this technique is more robust to changes and appearances. Objects athat appear on different aspects and orientations can be matched.

We use the `color_hist` function in this case, and especially the `hist_features` output that contains the information about the 3 color channels.

```Python
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
```

Histograms of colors for a sample of vehicle images : 

![alt text][vehicles_hist]

Histograms of colors for a sample of non-vehicle images :

![alt text][non_vehicles_hist]

The difference becomes more striking. The vehicles images are usually more saturated when compared to a pale background.


### Histogram of Gradients (HOG)

Eventually, the gradients of an image can definitely help to identify the structures within the image like I demonstrated it in other projects. Unless color feature extraction technique, HOG features extraction is robust to change of colors which can obviously happen with vehicles. The gradient is able to capture the edges of the shape of the car. We use a modified version that averages the gradients across multiple cells to account for some possible noise in the image.

The `scikit-image` package has a built in function to extract Histogram of Oriented Gradient features. There are a couple of parameters that we can adjust to get meaningful features:

* `ORIENT`: represents the number of orientation bins that the gradient information will be split up into in the histogram
* `PIX_PER_CELL`: specifies the cell size over which each gradient histogram is computed
* `CELL_PER_BLOCK`: specifies the local area over which the histogram counts in a given cell will be normalized

Here is the initial function we used :

```Python
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
```

To make it more visual, we can plot a representation that shows the dominant gradient direction within each cell with brightness corresponding to the strength of gradients in that cell.

Result for a sample of vehicle images : 

![alt text][vehicles_hog]

Result for a sample of non-vehicle images :

![alt text][non_vehicles_hog]

The HOG features capture the shape of the vehicles extremely well !

In the end we **concatenate all those features** to create our final feature vector. The set of parameters has been set after testing several options. We could choose the final parameters with a cross-validation approach but the classifier accuracy is so high (see below) that I did not find necessary to spend to much time on this task. They can be found in the [config file](https://github.com/itismouad/vehicle_detection/blob/master/config/config.json) storing all the hyperparamters.

|Parameter|Value|
|:--------|----:|
|Color Space|YCrCb|
|HOG Orient|8|
|HOG Pixels per cell|8|
|HOG Cell per block|2|
|HOG Channels|0|
|Spatial bin size| (16,16)|
|Histogram bins|32|
|Histogram range|(0,256)|


## Train a classifier

* code : [video_pipeline.py](https://github.com/itismouad/vehicle_detection/blob/master/video_pipeline.py)
* name of the python class :  **Train()**

Now that we have our training dataset, we can start to train our classifier. We first normalize our vectors and train both SVM and random forest classifiers. 

|Classifier|Accuracy|
|:--------:|:------:|
|Random Forest|0.993806306306|
|SVM|0.98536036036|

Accuracies are extremely high. We will use the random forest moving forward.


## Implement a sliding window search.

* code : [slider.py](https://github.com/itismouad/vehicle_detection/blob/master/slider.py)
* name of the python class :  **Slider()**

Now that the classifier can identify if an image is a car, we need to be able to search through the whole image for potential car matches. For this purpose we implement a sliding window search that will be an input to our classifier.

Test images to search :

![alt text][test_images_tosearch]

Test images after running each of the windows through a classifier, we are able to identify the windows where a vehicle is present :

![alt text][test_images_tosearch_boxed]

#### Heatmap creation

* code : [heater.py](https://github.com/itismouad/vehicle_detection/blob/master/heater.py)
* name of the python class :  **Heater()**

Now, to avoid false positives we can create a heatmap and apply a threshold of posiive predictions to make sure we are identifying a car :

![alt text][test_images_tosearch_hot]

The heatmap strategy works well to control for the noise in our images. The last step allows us to create a unique box per car :

![alt text][test_images_tosearch_unique]

**NB** : The performance of the method calculating HOG on each particular window was slow. To improve the processing performance, a HOG sub-sampling was implemented with the `run_efficient` function [here](https://github.com/itismouad/vehicle_detection/blob/master/slider.py)

![alt text][compare_start_end]

The image above was displayed thanks to the **videoPipeline()** python class located in [video_pipeline.py](https://github.com/itismouad/vehicle_detection/blob/master/video_pipeline.py) (see `process_image` and `run`).


## Pipeline (video)

* code : [video_pipeline.py](https://github.com/itismouad/vehicle_detection/blob/master/video_pipeline.py)
* name of the python class :  **videoPipeline()**

The final video can be found here : [project_video_output.mp4](https://github.com/itismouad/vehicle_detection/blob/master/project_video_output.mp4). There are some glitches in the current pipeline but overall, it has a strong performance.


## Discussion

- There a few improvements that need to be done on the glitches. The algorithm still shows a few false positives which we would need to completely remove in production.
- More information could be use from frame to frame to improve the robustness of the process.