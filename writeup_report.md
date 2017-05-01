## Behavioral Cloning Project

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/placeholder.png "Model Visualization"
[image2]: ./examples/placeholder.png "Grayscaling"
[image3]: ./examples/placeholder_small.png "Recovery Image"
[image4]: ./examples/placeholder_small.png "Recovery Image"
[image5]: ./examples/placeholder_small.png "Recovery Image"
[image6]: ./examples/placeholder_small.png "Normal Image"
[image7]: ./examples/placeholder_small.png "Flipped Image"

** Rubric Points 
Each of the points in the project [rubric](https://review.udacity.com/#!/rubrics/432/view) are addressed below.

---
** Files Submitted & Code Quality 

** 1. Submission includes all required files and can be used to run the simulator in autonomous mode

Project submission includes the following files:
* model.py contains the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 contains a trained convolution neural network 
* writeup_report.md summarizing the results
* video.mp4 shows the car autonomously driving around track 1

** 2. Submission includes functional code **
Using the Udacity-provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```
and running the simulator in autonomous mode.

** 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline used for training and validating the model, and it contains comments to explain how the code works.

** Model Architecture and Training Strategy

** 1. An appropriate model architecture has been employed

The model consists of a convolutional neural network based on the NVIDIA architecture (model.py lines 57-70) 

### SHOW LAYERS
model.add(Lambda(lambda x: x / 255.0 - 0.5, input_shape=(160,320,3)))
model.add(Cropping2D(cropping=((70,25),(0,0))))
model.add(Convolution2D(24,5,5,subsample=(2,2),activation="relu"))
model.add(Convolution2D(36,5,5,subsample=(2,2),activation="relu"))
model.add(Convolution2D(48,5,5,subsample=(2,2),activation="relu"))
model.add(Convolution2D(64,3,3,activation="relu"))
model.add(Convolution2D(64,3,3,activation="relu"))
model.add(Flatten())
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(1))

The data is normalized in the model using a lambda layer. Cropping is using to crop the top 70 pixels and the bottom 20 pixels from the frame, since these are very similar from image to image and have little bearing on effective training of the network. A series of convolution filters is applied, with varying filter size from 24 to 64. A series of flattening and densifcation layers were used to reduce to a single output (steering angle). 


** 2. Attempts to reduce overfitting in the model

Overfitting did not appear to be a problem based on the training performance. A sufficient size data set was used and validation data loss was increasing over the 4 epochs used. The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

** 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 73).

** 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. The training data included center lane driving, one reverse lap, and recovering from the left and right sides of the road.

** Model Architecture and Training Strategy

**1. Solution Design Approach

An evolutionary approach was used to develop the solution. An initial solution was developed based on the NVIDIA-derived neural network along with the provided data set. The NVIDIA network was observed to be superior to LeNet, so it was selected. In autonomous mode, the vehicle was able to drive straight sections of the course but had trouble with sharp turns, the bridge, and areas of transition between types of outside lane markers. 

From this initial baseline, new ideas were added one by one and tested to see if they offered any improvement. The most promising ideas involved pre-processing of the image data set. First, all the image data from all three cameras (left, center, right) was included. The steering angle for the left and right cameras was adjusted by 25 deg. Second, every image was flipped and these images were added to the data set (along with the associated steering angle, which was given a sign change). These two changes had the combined effect of a 6-fold increase in the size of the data set. It did increase the training time significantly, but the effect was that the autonomous car was able to successfully stay on the road throughout the course. 

** 2. Final Model Architecture

The final model architecture consisted of a convolutional neural network and is depicted above.

** 3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two laps on track one using center lane driving. Here is an example image of center lane driving:

![alt text][image1]

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to .... These images show what a recovery looks like starting from ... :

![alt text][image2]
![alt text][image3]
![alt text][image4]


To augment the data sat, I also flipped images and angles thinking that this would ... For example, here is an image that has then been flipped:

![alt text][image5]
![alt text][image6]

After the collection process, I had X number of data points. I then preprocessed this data by ...

I finally randomly shuffled the data set and put Y% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was Z as evidenced by ... I used an adam optimizer so that manually training the learning rate wasn't necessary.
