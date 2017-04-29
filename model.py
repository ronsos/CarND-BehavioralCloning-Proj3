import csv 
import cv2
import numpy as np

# create adjusted steering measurements for the side camera images
correction = 0.25 # this is a parameter to tune

# Load driving log data
lines = [] 
with open('data/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader: 
        lines.append(line)

# Load images and steering measurements from all 3 cameras
images = []
measurements = []
for line in lines:
    
    path_center = 'data/IMG/' + line[0].split('/')[-1]
    path_left = 'data/IMG/' + line[1].split('/')[-1]
    path_right = 'data/IMG/' + line[2].split('/')[-1]
    
    img_center = cv2.imread(path_center)
    img_left = cv2.imread(path_left)
    img_right = cv2.imread(path_right)  
    
    steering_center = float(line[3])
    steering_left = steering_center + correction
    steering_right = steering_center - correction
    
    # add images and angles to data set
    images.append(img_center)
    images.append(img_left)
    images.append(img_right)
    measurements.append(steering_center)
    measurements.append(steering_left)
    measurements.append(steering_right)

# Include flipped images    
augmented_images, augmented_measurements = [],[]
for image,measurement in zip(images, measurements):
    augmented_images.append(image)
    augmented_measurements.append(measurement)
    augmented_images.append(cv2.flip(image,1))
    augmented_measurements.append(measurement*-1.0)

# Create training data arrays    
X_train = np.array(augmented_images)
y_train = np.array(augmented_measurements)

# Keras input/header
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda
from keras.layers.convolutional import Convolution2D, Cropping2D

# NVIDIA architecture with cropping
model = Sequential()
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

# Train model
model.compile(loss='mse', optimizer='adam')
model.fit(X_train, y_train, validation_split = 0.2, shuffle=True, nb_epoch=4)

# Save model
model.save('model.h5')









    