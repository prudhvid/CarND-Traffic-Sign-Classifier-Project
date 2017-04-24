# **Traffic Sign Recognition** 

---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set, downloaded from [here](https://d17h27t6h515a5.cloudfront.net/topher/2017/February/5898cd6f_traffic-signs-data/traffic-signs-data.zip)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./imgs/graph.png "Visualization"
[image2]: ./pics/grayscale.jpg "Grayscaling"
[image3]: ./pics/.png "Random Noise"
[image4]: ./pics/0.png "Traffic Sign 1"
[image5]: ./pics/1.png "Traffic Sign 2"
[image6]: ./pics/2.png "Traffic Sign 3"
[image7]: ./pics/3.png "Traffic Sign 4"
[image8]: ./pics/4.png "Traffic Sign 5"
[image9]: ./pics/5.png "Traffic Sign 6"
[image10]: ./pics/6.png "Traffic Sign 7"
[image11]: ./pics/7.png "Traffic Sign 8"
[image12]: ./pics/8.png "Traffic Sign 9"
[image13]: ./imgs/graph2.png "Visualization2"
[image14]: ./imgs/pt.png "orig"
[image15]: ./imgs/pt2.png "perspective"
[image16]: ./imgs/g.png "gamma"
[image17]: ./imgs/gn.png "guassian"
[image18]: ./imgs/r.png "rotation"
[image19]: ./imgs/f.png "Traffic Sign 10"
[image20]: ./imgs/pred.png "Predictions"
[image21]: ./imgs/right.png "right"
[image22]: ./imgs/right-n.png "right-neural"
[image23]: ./imgs/right-n2.png "right-neural2"



## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---

You're reading it! and here is a link to my [project code](https://github.com/prudhvid/CarND-Traffic-Sign-Classifier-Project)

### Data Set Summary & Exploration

#### 1. A basic summary of the data set. 

I used the python numpy library to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799
* The size of the validation set is 12630
* The size of test set is 4410
* The shape of a traffic sign image is 32x32x3
* The number of unique classes/labels in the data set is 43

#### 2. An exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. It is a bar chart showing how the frequency of training examples varies across classes

![Frequncy vs class graph][image1]

### Design and Test a Model Architecture

#### 1. Preprocessing the image data.

As a first step, I decided to use histogram equalization using equalize_adapthist. This will help in spreading the intensities across all the pixels. I initally tried
testing with rbg images and hsv images. But the results are not soo good. So finally I've changes to grayscale image after equalize_adapthist.

Here is an example of a traffic sign image before and after preprocessing.

![alt text][image19]

As a last step, I normalized the image data because the SGD algorithm works best when the data is normalized around mean 0.
By dividing it within 255.0 initially, all the images were brought within the range of `[0-1.0]`, later I did `(x-0.5)/0.5` to normalize the data between
-1 to 1

#### I decided to generate additional data because
1. The initial dataset has non-uniform distribution which will lead to bias of the classes with more freqency
2. From the failure cases, I figured that for many of the images which the model was unable to predict the correct class, usually it was due to the lighening
condtion, hence generated lots of images with different gamma
3. The other augmentations applied also helps in balancing the dataset

#### Here are some examples of an original image and an augmented image:

**Perpective Transform**

![alt text][image14] ![alt text][image15]


**Rotation Transform**

![alt text][image14] ![alt text][image18]

**Random Gamma correction**

![alt text][image14] ![alt text][image16]

**Guassian Noise Correction**

![alt text][image14] ![alt text][image17]

After all the augmentations, the frequency vs classes graph looks like this

![alt text][image13]

#### 2. Final model architecture

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x1 GRAYSCALE    							|
| Convolution layer  with RELU   	| 5x5 stride, valid padding, outputs 28x28x20 	|
| Max pooling	      	| 2x2 stride,  outputs 14x14x20 				|
| Convolution layer with RELU	    | 5x5 stride, valid padding, outputs 10x10x40	|
| Max pooling | 2x2 stride, outputs 5x5x40 |
| Flatten		| output 1000        									|
| Fully connected				| output 250        					|
| Fully Connected				| output 125					        |
| Fully Connected        		| output 84									|
| Fully Connected        		| output 43									|


**Between every fully connected layer, I've added dropout layers to aoid over fitting**
 


#### 3. Training the model.

To train the model, I used Adam Optimizer with batch size of 128 and with 10 epochs. The training occurs in two passes, first pass with learning rate - 0.001 and second with learning rate - 0.0001. The keep probabilty for the dropouts is assigned to be 0.6. Adam optimizer optimizes to reduce the cross entropy of the model. 

#### 4. The approach taken for finding a solution and getting the validation set accuracy to be at least 0.93.

My final model results were:
* training set accuracy of 99.5
* validation set accuracy of 96.5
* test set accuracy of 96.4

If a well known architecture was chosen:
* Started with LeNet model as it is good for generic classification of image data, and had acheived a very high accuracy with handwriting data sets
* But it could only achieve a maximum of 93% on validation set which is the baseline of the project submission. So it was underfitting the datset. 
* To add more complexity to the model, added a fully connected layer and increased the width of each layer within the model
* Also added dropout layers between fully connected to avoid overfitting
* To over overfitting with the new complex model, augmented dataset to add more training examples
* This enabled the model to achieve a validation accuracy of 96.5%, test accuracy of 96.4%, training accuracy of 99.5%

### Test a Model on New Images

#### 1. Classifying images from the web

Here are nine German traffic signs that I found on the web:

![alt text][image4] ![alt text][image5] ![alt text][image6] 
![alt text][image7] ![alt text][image8] ![alt text][image9]
![alt text][image10] ![alt text][image11] ![alt text][image12]

All these images are classified correctly with high probabilty for the classsification. 

#### 2. Discuss the model's predictions on these new traffic signs.
Below on these new images, of the type never seen before by the model during training, and it had struggled to give correct answers

![alt text][image19]

Here are the results of the prediction:

![alt text][image20]


The model was able to correctly guess correctly on all images **similar** to the ones it has seen before. But when given traffic signs of other countries, like for eg, here in one of the examples, **the wild animals crossing sign is flipped when compared to the german dataset** and the model failed to predict it. **The accuracy for all images taken from the web turns out to be 66.7% and that of those images similar to german dataset is 100%.** 

####3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 11th cell of the Ipython notebook.

For all the german data set images, the model is pretty sure(close to 95-100%) evidently from the bar graph. But for new datasets, here are the results

**Pedestrian crossing**
The model predicted the new sign as children crossing, probably because of new horizontal lines which is not present in German dataset pedestrian crossing images

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .85         			|   Children crossing 									| 
| .12     				| bicycle crossing 										|
| .08					| Pedestrian crossing							|


**Stop sign**
The model predicted the new sign as stop sign with a very high probabilty, which is pretty amazing!! 

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .99         			|  Stop sign									| 

**Bumpy Road**
The model predicted the new sign as end of all speed and passing limits, probably because the traffic sign is not exactly cropped to fit entirely with the image

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .65         			|  End of all speed and passing limits									| 
| .09     				| Road work 										|
| .08					| Bumpy road							|

### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
#### 1.Visual output of your trained network's feature maps.

![alt text][image21]
![alt text][image22]

* This shows the first layer visualization of the cnn.
* If we look closely, 
    * Feature 16 represents the edges of the input image
    * Feature 12 represents the inverse image
    * Other features which are not very obvious from here

![alt text][image23]

At layer 2, it has become more complex, to analyze the image especially because of the relu units.