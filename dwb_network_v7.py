import csv
import cv2
from scipy import ndimage
import numpy as np
import math

#may want to use this instead of cv2 because  cv2.imread will get images in BGR format, while drive.py uses RGB.
#image = ndimage.imread(current_path)

# Commenting out the method which loads ALL of the images into memory at once, going with generator method
# lines = []
# #with open('/opt/provided_training_data/driving_log.csv') as csvfile:
# with open('/home/workspace/CarND-Behavioral-Cloning-P3/data/driving_log.csv') as csvfile:
#     reader = csv.reader(csvfile)
#     next(reader) #skip the first line since it is a header row
#     for line in reader:
#         lines.append(line)

# images = []
# measurements = []

# image_count = 0
# for line in lines:
#     source_path = line[0]					  #The 0th "token" (i.e. column in the file)
#     filename = source_path.split('/')[-1]     #splits off the filename after the /

#     #current_path = '/opt/provided_training_data/IMG/' + filename  #Resue only the filename, but with the new path on the server
#     current_path = '/home/workspace/CarND-Behavioral-Cloning-P3/data/IMG/' + filename  #Resue only the filename, but with the new path on the server
	
#     current_image = ndimage.imread(current_path)	  #Used instead of (due to RGB/BRG)- cv2.imread(current_path)
#     image_count+=1                                    #Increment the Image Counter
	
#     current_measurement = float(line[3])			  #grab the 4th column in the file which is the steering wheel angle cooresponding to that image
	
# 	#Now build a cumulative set of these images and associated measurements by appending it as we go
#     images.append(current_image)
#     measurements.append(current_measurement)


# #Now I have my set of Training Images (X_train) and Labels (y_train) for these images
# #Making them numpy arrays because this is what Keras wants as an input
# print("Total Number of Images: ", image_count)
# X_train = np.array(images)
# y_train = np.array(measurements)

import os
import csv

samples = []
with open('/home/workspace/dwb_record_v5/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    next(reader) #skip the first line since it is a header row
    for line in reader:
        samples.append(line)

from sklearn.model_selection import train_test_split
train_samples, validation_samples = train_test_split(samples, test_size=0.2)

import cv2
import numpy as np
import sklearn

def generator(samples, batch_size=32):
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        sklearn.utils.shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            angles = []
            for batch_sample in batch_samples:
                name = '/home/workspace/dwb_record_v5/IMG/'+batch_sample[0].split('/')[-1]
                center_image = cv2.imread(name)
                center_angle = float(batch_sample[3])
                images.append(center_image)
                angles.append(center_angle)

            # trim image to only see section with road
            X_train = np.array(images)
            y_train = np.array(angles)
            yield sklearn.utils.shuffle(X_train, y_train)

# Set our batch size
batch_size=32

# compile and train the model using the generator function
train_generator = generator(train_samples, batch_size=batch_size)
validation_generator = generator(validation_samples, batch_size=batch_size)

#ch, row, col = 3, 80, 320  # Trimmed image format



#Build most basic neural network just to test things out
# This most-basic network consists of flattened image connected to single output layer

#Some differences from the last project where we were building a classification network
# For the classification network we wanted to apply a softmax function to the output of the network because we 
# wanted to treat the output of the network as probabilities that the image was associated with each possible class
# then we would calculate a cross-entropy between this output and the ground-truth labels across all the predictions

#In this case we are trying to "directly" predict the steering wheel angle as a number - 
#so we just observe the output of the model directly.  We don't apply a softmax at the output.
#Also, we aren't using cross-entropy here - we are going to use Mean-Squared-Error (mse) - error
# between the ground-truth label and the predicted label across the whole set - and we seek to minimize this.


from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Activation, Dropout, Cropping2D
#from keras.layers import Convolution2D
#from keras.layers.pooling import MaxPooling2D
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPooling2D

# dwb_model = Sequential()   #Type of model
# dwb_model.add(Lambda(lambda x : x/255.0-0.5,input_shape = (160,320,3))) #using a keras "Lamdba layer" to pre-process the images
# dwb_model.add(Flatten()) # don't need this because now in lamda step - (Flatten(input_shape = (160,320,3)))
# dwb_model.add(Dense (1))


dwb_model = Sequential()                                                #Setting up model type
#dwb_model.add(Lambda(lambda x : x/255.0-0.5,input_shape = (160,320,3), output_shape=(ch,row,col))) #Keep same pre-processing step
dwb_model.add(Lambda(lambda x : x/255.0-0.5,input_shape = (160,320,3))) #Keep same pre-processing step
dwb_model.add(Cropping2D(cropping=((70,25),(0,0))))                       #cropping off top 70 and lower 25 pixels of each image
#dwb_model.add(Flatten()) #Don't need to flatten now - sending image stright into convolution layer
dwb_model.add(Conv2D(24,5,5,subsample=(2,2),activation='relu'))               #Layer1 - LeNet Convolutional Layer1 is 6 filters at size of 5x5
#dwb_model.add(Activation('relu'))
#dwb_model.add(MaxPooling2D())                  #After the first layer ther eis 2x2 pooling, 2x2 is default?
dwb_model.add(Dropout(0.1))                    #This is not in original LeNet - adding a Dropout layer after each pooling layer
dwb_model.add(Conv2D(36,5,5,subsample=(2,2),activation='relu'))              #Layer2 - LeNet Convolutional Layer2 is 16 output filters at size of 5x5
#dwb_model.add(Activation('relu'))
#dwb_model.add(MaxPooling2D())                  #After the this layer there is 2x2 pooling, 2x2 is default?
dwb_model.add(Dropout(0.1))                    #This is not in original LeNet - adding a Dropout layer after each pooling layer

dwb_model.add(Conv2D(48,5,5,subsample=(2,2),activation='relu'))
dwb_model.add(Conv2D(64,3,3,activation='relu'))
dwb_model.add(Conv2D(64,3,3,activation='relu'))

dwb_model.add(Flatten())                       #Then perform flatten before going into fully-connected layers
#Note with the fully connected layers I am not using Activation because we want to directly predict a value, instead of performing classification
dwb_model.add(Dense(120))                      #Layer3 - LeNet Fully Connected with output of 120
dwb_model.add(Dropout(0.2))  
dwb_model.add(Dense(84))                       #Layer4 - LeNet Fully Connected with output of 84
dwb_model.add(Dropout(0.05))  
dwb_model.add(Dense(1))                        #Layer5 - LeNet Fully Connected - final output is only 1 value - steering wheel angle



#Compile the model and set the training pipeline to minimize mean squared error
dwb_model.compile(loss='mse', optimizer='adam')

#Now perform the training with the data that you put into the X_train and y_train arrays
# We are splitting 20% of the data off and also performing a shuffle of the data
# The default number of epochs is set to 10 by Keras
#history_object = dwb_model.fit(X_train, y_train, validation_split=0.2,shuffle=True,nb_epoch=5)

history_object = dwb_model.fit_generator(train_generator, \
            steps_per_epoch=math.ceil(len(train_samples)/batch_size), \
            validation_data=validation_generator, \
            validation_steps=math.ceil(len(validation_samples)/batch_size), \
            epochs=8, verbose=1)


#from keras.models import Model
import matplotlib.pyplot as plt

### print the keys contained in the history object
print(history_object.history.keys())

# See this link for more info on save feature: https://keras.io/getting-started/faq/#how-can-i-save-a-keras-model
dwb_model.save('dwb_model_v6-4.h5')  #Creates an HDF5 file 'model.h5'

### plot the training and validation loss for each epoch
#fig = plt.figure()
plt.plot(history_object.history['loss'])
plt.plot(history_object.history['val_loss'])
plt.title('model mean squared error loss')
plt.ylabel('mean squared error loss')
plt.xlabel('epoch')
plt.legend(['training set', 'validation set'], loc='upper right')
#fig.savefig('Training_Results.png')
plt.show()

exit()  #adding this since I set my generator to go forever and never end :-)


# Since we're trying to predict steering wheel angle this is a regression network not a classification network so we don't need activation


#Considering the LeNet Architecture that was previously built in TensorFlow for the last project:
# def LeNet(x):    
#     # Hyperparameters
#     # Arguments used for tf.truncated_normal, randomly defines variables for the weights and biases for each layer
#     mu = 0
#     sigma = 0.148  #started out as 0.1, then I tried 0.15
    
#     #Input feature sizes being pased in as conv1_input (x) are 32x32x1
      
    
#     # DWB: Layer 1: Convolutional. 
#     # Actual Input Size (for each element in len(X_train) is = 32x32x1
#     # Desired Output Size for this Architecture is Output = 28x28x6
#     # We choose a filter size of 5x5 with Stride of 1 to achieve this.  (32-5+1)/1 = 28
    
#     # Fundamental Parameters Weight and Bias are sized correctly and initialized to random values
#     # Weight dimensions are in the form: (height, width, input_depth, output_depth)
#     # Since my desired output shape is (....by-6) I want 6 biases to be added
#     conv1_W = tf.Variable(tf.truncated_normal(shape=(5, 5, 1, 6), mean = mu, stddev = sigma))
#     conv1_b = tf.Variable(tf.zeros(6))
    
#     #inserted 'conv1_input' for 'x' in original lab answer
#     # Note that strids are all set to 1 here - strieds are of following dimensions (batch_size, height, width, depth)
#     conv1   = tf.nn.conv2d(x, conv1_W, strides=[1, 1, 1, 1], padding='VALID') + conv1_b

#     # DWB: Activation.
#     conv1 = tf.nn.relu(conv1)

#     # DWB: Pooling. Input = 28x28x6. Output = 14x14x6.
#     conv1 = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')
    
#     # DWB: Layer 2: Convolutional. Output = 10x10x16.
#     conv2_W = tf.Variable(tf.truncated_normal(shape=(5, 5, 6, 16), mean = mu, stddev = sigma))
#     conv2_b = tf.Variable(tf.zeros(16))
#     conv2   = tf.nn.conv2d(conv1, conv2_W, strides=[1, 1, 1, 1], padding='VALID') + conv2_b
    
#     # DWB: Activation.
#     conv2 = tf.nn.relu(conv2)

#     # DWB: Pooling. Input = 10x10x16. Output = 5x5x16.
#     conv2 = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')

#     # DWB Flatten. Input = 5x5x16. Output = 400.
#     fc0   = flatten(conv2)
    
#     # DWB: Layer 3: Fully Connected. Input = 400. Output = 120.
#     fc1_W = tf.Variable(tf.truncated_normal(shape=(400, 120), mean = mu, stddev = sigma))
#     fc1_b = tf.Variable(tf.zeros(120))
#     fc1   = tf.matmul(fc0, fc1_W) + fc1_b
    
#     # DWB: Activation.
#     fc1    = tf.nn.relu(fc1)

#     # DWB: Layer 4: Fully Connected. Input = 120. Output = 84.
#     fc2_W  = tf.Variable(tf.truncated_normal(shape=(120, 84), mean = mu, stddev = sigma))
#     fc2_b  = tf.Variable(tf.zeros(84))
#     fc2    = tf.matmul(fc1, fc2_W) + fc2_b
    
#     # DWB: Activation.
#     fc2    = tf.nn.relu(fc2)
    
#     #changed output from 10 to 43 since we have 43 possible signs in this set
#     # DWB: Layer 5: Fully Connected. Input = 84. Output = 43.
#     fc3_W  = tf.Variable(tf.truncated_normal(shape=(84, 43), mean = mu, stddev = sigma))
#     fc3_b  = tf.Variable(tf.zeros(43))
#     logits = tf.matmul(fc2, fc3_W) + fc3_b
