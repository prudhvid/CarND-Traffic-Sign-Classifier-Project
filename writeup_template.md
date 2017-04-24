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

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

As a first step, I decided to convert the images to grayscale because ...

Here is an example of a traffic sign image before and after grayscaling.

![alt text][image2]

As a last step, I normalized the image data because the SGD algorithm works best when the data is normalized around mean 0.
By dividing it within 255.0 initially, all the images were brought within the range of `[0-1.0]`, later I did `(x-0.5)/0.5` to normalize the data between
-1 to 1

#### I decided to generate additional data because
1. The initial dataset has non-uniform distribution which will lead to bias of the classes with more freqency
2. From the failure cases, I figured that for many of the images which the model was unable to predict the correct class, usually it was due to the lighening
condtion, hence generated lots of images with different gamma
3. The other augmentations applied also helps in balancing the dataset

Here is an example of an original image and an augmented image:

![alt text][image3]

The difference between the original data set and the augmented data set is the following ... 

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
* test set accuracy of 98.1

If a well known architecture was chosen:
* Started with LeNet model as it is good for generic classification of image data, and had acheived a very high accuracy with handwriting data sets
* But it could only achieve a maximum of 93% on validation set which is the baseline of the project submission. So it was underfitting the datset. 
* To add more complexity to the model, added a fully connected layer and increased the width of each layer within the model
* Also added dropout layers between fully connected to avoid overfitting
* To over overfitting with the new complex model, augmented dataset to add more training examples
* This enabled the model to achieve a validation accuracy of 96.5%, test accuracy of 98.1%, training accuracy of 99.5%

### Test a Model on New Images

#### 1. Classifying images from the web

Here are nine German traffic signs that I found on the web:

![alt text][image4] ![alt text][image5] ![alt text][image6] 
![alt text][image7] ![alt text][image8] ![alt text][image9]
![alt text][image10] ![alt text][image11] ![alt text][image12]

The first image might be difficult to classify because ...

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Stop Sign      		| Stop sign   									| 
| U-turn     			| U-turn 										|
| Yield					| Yield											|
| 100 km/h	      		| Bumpy Road					 				|
| Slippery Road			| Slippery Road      							|


The model was able to correctly guess 4 of the 5 traffic signs, which gives an accuracy of 80%. This compares favorably to the accuracy on the test set of ...

####3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 11th cell of the Ipython notebook.

For the first image, the model is relatively sure that this is a stop sign (probability of 0.6), and the image does contain a stop sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .60         			| Stop sign   									| 
| .20     				| U-turn 										|
| .05					| Yield											|
| .04	      			| Bumpy Road					 				|
| .01				    | Slippery Road      							|


For the second image ... 

### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
####1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?


