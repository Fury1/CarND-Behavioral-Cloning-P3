# **Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: writeup_images/cnn-architecture.png "Nvidia Model"
[image2]: writeup_images/figure_1.png "Training and Test Data"
[image3]: writeup_images/left_camera.jpg "Left Car Camera"
[image4]: writeup_images/center_camera.jpg "Center Car Camera"
[image5]: writeup_images/right_camera.jpg "Right Car Camera"
[image6]: writeup_images/left_crop.jpg "Left Car Camera Crop"
[image7]: writeup_images/center_crop.jpg "Center Car Camera Crop"
[image8]: writeup_images/right_crop.jpg "Right Car Camera Crop"

### Rubric Points
#### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted

* Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a final fully trained convolution neural network
* writeup_report.md summarizing the results

---

#### Quality of Code
* Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing

```sh
python drive.py model.h5
```

* The model.py file contains the code for training and saving the convolution neural network in a checkpoint fashion. There is a function that can be set for either re-training of an already trained model on new data as well as just fitting a whole new model from scratch. The current file and its configuration shows the pipeline used for training and validating the final model.

---

### Model Architecture and Training Strategy

#### Model Architecture
* My model's initial architecture was NVIDIA's published example that they used on a real self driving car (Below).
I thought this would be a good initial starting point for the simulator as NIVIDIA seemed to have good success with it.
I read [End to End Deep Learning](https://devblogs.nvidia.com/parallelforall/deep-learning-self-driving-cars/) and tried to understand what it was going to take to get this model to work with the Udacity data. After taking some time to decide on key data augmentation points. I thought I would begin by using just center images/steering angles, introducing some random image/steering angle flips, training the model with an Adam optimizer to keep things simple, using a mean squared error function to calculate the loss, and 5 epochs. From there I figured I could get a baseline to see what happens and adjust accordingly.   

![Nvidia Model][image1]

#### Overfitting
* The model was trained and validated on different data sets to ensure that the model was not overfitting (model.py code line 142). In addition, the model contains dropout layers in order to reduce overfitting (model.py lines 48 and 52). Lastly, the final trained model was tested by running it through the simulator and ensuring that the vehicle could stay on the track and drive rationally.

#### Model Parameter Tuning
* The model used an Adam optimizer, so the learning rate was not tuned manually (model.py line 167-168).
However, batch size, steering correction values, drop out rates, random flip probabilities, and epochs were tuned manually to get the right model to be produced during training sessions.

#### Appropriate Training Data
* The initial training data used was what was included in the Udacity dataset. After initial training and testing, I found that there were similar problems in each of the models tested despite parameter tuning. At that point I concluded that additional data should be collected to help the model train better on the areas of concern. Additional data was then recorded in the problem spots and added to the original Udacity dataset. The final training dataset was a combination of the Udacity data as well as some hand picked parts of the track that needed to be improved upon based on the initially tested models.

---

### Architecture and Training Documentation

#### Solution Design Approach
* The overall strategy for deriving a model architecture was to start with a known good architecture like Nvidia and fine tune it to better suit the test track and the dataset. After initial testing and hyperparameter tuning I found that the model was lacking something. Initially, I knew from previous experience that the pixels should start by being normalized between -1 and 1. I manually added in a Keras lambda pixel normalization and retested to find some small improvements. After reading some further Keras documentation I found that Keras had their own layer normalization that could be added to a model. Once I understood what it was actually doing I thought it wouldn't be a bad idea to normalize every layer as inputs passed through the network, my thoughts were that this would help get the model trained more accurately (and quickly) because values could not get out of control between layers. Right away by simply just adding in a normalization between all my layers I noticed a significant improvement. At that point I was able to cover the majority of the track with the  exception of a few problem areas.

	I further tried tuning parameters and adding in different data augmentation techniques like dropping data that could make my model biased (particularly 0 steering angle measurements), flipping measurements/images, and implementing additional camera images with respective corrections to steering angles, etc. After trying all these different techniques and testing the model in the simulation repeatedly I concluded that I did not have enough data in specific areas of the track, and that was the root cause of my problem areas.

	At that point I went back into the simulator and collected some of my own data in the areas that I felt needed it and added that in to the Udacity data set to better balance the training examples. From there I also had to further adjust how many of the samples with 0 steering angle measurements made it into my training data that got fed into the model, this helped make sure the data was not biased towards not turning. At the end of every training session I plotted the training loss vs validation loss as well as a histogram of all the steering angle samples to ensure that my samples were spread out evenly and that I was not overfitting. Plotting that data gave me confidence that my model and data was in fact what I wanted it to be.

	Lastly, I found it hard to find the right number of epochs to train for, to many and I was wasting time or missing out on a good model, to few and I didn't know if there was anything better ahead. To fix this I implemented the Keras checkpoint feature and saved a copy of each model every epoch. At the end of each training session I ran the models one by one to see which one performed the best. Interestingly enough, the model that performed the best was not the epoch that had the lowest training or validation loss. That made me really value the checkpoint feature because I was able to try many different hyperparameters for any number of epochs without thinking to much about it or having to worry.

	At this point hyperparameters were tuned, tested, re-tuned, and re-tested until a successful model emerged that allowed me to get a full clean lap on the test track.

	![Training and Test Data][image2]

#### Final Model Architecture
* The final model architecture (model.py lines 35-56)

Final Architecture:

| Layer         		|     Description	        							|
|:---------------------:|:-----------------------------------------------------:|
| Input      			|160x320x3 RGB image   									|
| Cropping2D			|65px off the top, 25px off the bottom					|
| Pixel Normalization	|All image pixels get normalized to a mean of 0, +-1	|
| Convolution 5x5   	|2x2 stride, VALID padding, Elu activation, Output = 24	|
| Batch Normalization	|Normalize the layer									|
| Convolution 5x5   	|2x2 stride, VALID padding, Elu activation, Output = 36	|
| Batch Normalization	|Normalize the layer									|
| Convolution 5x5   	|2x2 stride, VALID padding, Elu activation, Output = 48	|
| Batch Normalization	|Normalize the layer									|
| Convolution 3x3   	|1x1 stride, VALID padding, Elu activation, Output = 64	|
| Batch Normalization	|Normalize the layer									|
| Convolution 3x3   	|1x1 stride, VALID padding, Elu activation, Output = 64	|
| Batch Normalization	|Normalize the layer									|
| Dropout				|0.50 Dropout rate during training						|
| Flatten				|Flatten the CL to prepare for FC layer					|
| Fully Connected #1 	| Output = 100, Elu activation							|
| Batch Normalization	|Normalize the layer									|
| Dropout				|0.50 Dropout rate during training						|
| Fully Connected #2 	| Output = 50, Elu activation							|
| Batch Normalization	|Normalize the layer									|
| Fully Connected #3 	| Output = 10, Elu activation							|
| Fully Connected #4 	| Output = 1, Final Model Steering Angle Prediction		|


#### Creation of the Training Set & Training Process
* The majority of the data I used was from the included Udacity dataset. Initially
like I have previously stated I tried to train the model off of just that but I found that
it was lacking in certain areas of the track. After making this conclusion I collected my own
data around the problem areas of the track that I found when testing my model in the simulator.
From there I added that data into the Udacity dataset.

	Overall this helped the model generalize the track better but there was still more room for
	improvement in my opinion. Knowing that there was 3x as much information available to me in
	the current dataset if I utilized all 3 car cameras, I elected to try and utilize the left and right cameras
	to further help my model generalize the track. I already knew I had a good balanced dataset, so rather then risking collecting
	bad data by accident I thought that this was a safer move that would help me create an even better model. It also saved me
	time trying to drive around track as the data was already there but not being utilized. I figured it was a win across the board.

	For reference, here are some sample images before and after preprocessing/augmentation, the final examples are what were fed into the network.

	**Intial Images:**

	![Left][image3]
	![Center][image4]
	![Right][image5]

	**Cropped Images:**
	*After cropping, associated left and right camera steering angles were adjusted by +.125 for left and -.125 for right*

	![Left][image6]
	![Center][image7]
	![Right][image8]

	*Additionally all pixels were normalized to a mean of 0 with a deviation of +-1. IE: (pixel - 128/128)*

---

### Simulation
*Note: Model was run in the simulator at "fastest" graphics quality and "full" screen settings.*

* Ultimately I had a few models that could get around the track without taking any undesired shortcuts. There were however subtle differences between those models so I ended up choosing the model that drove the smoothest to represent the final product (*Checkpoint model 23*). My final model could successfully navigate the track forwards as well as backwards, but at times there was some slight swaying of the vehicle going around the track. Because of this, I think it is worth mentioning that when I was adding my own training data to the Udacity dataset I did sway from left to right with the car when recording data using the mouse. In retrospect, I might have been able to gather better data just steering with the keyboard.

 	The most important part of this project that I learned was that good data is key, in fact, it is worth the time to hand pick the data that you use to train your model. Something as simple as me recording data using a mouse that I admittedly was not the best at caused my model to reproduce the same type of behavior. For the purposes of this project and my model this was not the end of the world, but in real life I could see some very undesired consequences.


---

### Conclusion
* I think this was one of the coolest projects to date and I thought that there was some very fustrating but important lessons to learn. My take away from all of this is that good data and sound preprocessing/augmentation is key no matter how good your model architecture is.
