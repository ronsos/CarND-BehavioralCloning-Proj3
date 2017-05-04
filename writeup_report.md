## Behavioral Cloning Project

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/center.jpg "Center Camera"
[image2]: ./examples/right.jpg "Right Camera"
[image3]: ./examples/left.jpg "Left Camera"
[image4]: ./examples/flipped.jpg "Flipped Image"


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

The final model architecture consisted of a convolutional neural network and is depicted below:

____________________________________________________________________________________________________
Layer (type)                     Output Shape          Param #     Connected to                     
====================================================================================================
lambda_1 (Lambda)                (None, 160, 320, 3)   0           lambda_input_1[0][0]             
____________________________________________________________________________________________________
cropping2d_1 (Cropping2D)        (None, 65, 320, 3)    0           lambda_1[0][0]                   
____________________________________________________________________________________________________
convolution2d_1 (Convolution2D)  (None, 31, 158, 24)   1824        cropping2d_1[0][0]               
____________________________________________________________________________________________________
convolution2d_2 (Convolution2D)  (None, 14, 77, 36)    21636       convolution2d_1[0][0]            
____________________________________________________________________________________________________
convolution2d_3 (Convolution2D)  (None, 5, 37, 48)     43248       convolution2d_2[0][0]            
____________________________________________________________________________________________________
convolution2d_4 (Convolution2D)  (None, 3, 35, 64)     27712       convolution2d_3[0][0]            
____________________________________________________________________________________________________
convolution2d_5 (Convolution2D)  (None, 1, 33, 64)     36928       convolution2d_4[0][0]            
____________________________________________________________________________________________________
flatten_1 (Flatten)              (None, 2112)          0           convolution2d_5[0][0]            
____________________________________________________________________________________________________
dense_1 (Dense)                  (None, 100)           211300      flatten_1[0][0]                  
____________________________________________________________________________________________________
dense_2 (Dense)                  (None, 50)            5050        dense_1[0][0]                    
____________________________________________________________________________________________________
dense_3 (Dense)                  (None, 10)            510         dense_2[0][0]                    
____________________________________________________________________________________________________
dense_4 (Dense)                  (None, 1)             11          dense_3[0][0]                    
====================================================================================================
Total params: 348,219
Trainable params: 348,219
Non-trainable params: 0

** 3. Creation of the Training Set & Training Process

Here is an example image of center lane driving from the center camera:

![alt text][image1]

The left and right camera frames were also added to the data set. Here are examples of the left and right camera images from the same time as the center image:

![alt text][image2]
![alt text][image3]

To augment the data set, I flipped all of the images and changed the sign on the associated steering angles. The rationale was to add more right turn data to help offset all the left turns on the track. Here is the flipped version of center image at the same time stamp:

![alt text][image4]

After the collection process, I had X number of data points. I then preprocessed this data by normalizing.

20% of the data was sliced off into a validation set. 

I used the training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 4 as evidenced by the observed loss. I used an adam optimizer so that manually training the learning rate wasn't necessary.
