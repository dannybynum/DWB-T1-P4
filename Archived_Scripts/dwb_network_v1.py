import csv
import cv2
from scipy import ndimage
import numpy as np

#may want to use this instead of cv2 because  cv2.imread will get images in BGR format, while drive.py uses RGB.
#image = ndimage.imread(current_path)

lines = []
with open('/opt/provided_training_data/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    next(reader) #skip the first line since it is a header row
    for line in reader:
        lines.append(line)

images = []
measurements = []

i_tshoot = 0
for line in lines:
    source_path = line[0]					  #The 0th "token" (i.e. column in the file)
    filename = source_path.split('/')[-1]     #splits off the filename after the /
    if i_tshoot == 0:
        print("source path is: ",source_path)
    i_tshoot=1
    current_path = '/opt/provided_training_data/IMG/' + filename  #Resue only the filename, but with the new path on the server
	
    current_image = ndimage.imread(current_path)	  #Used instead of (due to RGB/BRG)- cv2.imread(current_path)
	
    current_measurement = float(line[3])			  #grab the 4th column in the file which is the steering wheel angle cooresponding to that image
	
	#Now build a cumulative set of these images and associated measurements by appending it as we go
    images.append(current_image)
    measurements.append(current_measurement)


#Now I have my set of Training Images (X_train) and Labels (y_train) for these images
#Making them numpy arrays because this is what Keras wants as an input

X_train = np.array(images)
y_train = np.array(measurements)

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
from keras.layers import Flatten, Dense

MostBasicModel = Sequential()   #Type of model
MostBasicModel.add(Flatten(input_shape = (160,320,3)))
MostBasicModel.add(Dense (1))

#Compile the model and set the training pipeline to minimize mean squared error
MostBasicModel.compile(loss='mse', optimizer='adam')

#Now perform the training with the data that you put into the X_train and y_train arrays
# We are splitting 20% of the data off and also performing a shuffle of the data
# The default number of epochs is set to 10 by Keras
MostBasicModel.fit(X_train, y_train, validation_split=0.2,shuffle=True,nb_epoch=5)

# See this link for more info on save feature: https://keras.io/getting-started/faq/#how-can-i-save-a-keras-model
MostBasicModel.save('dwb_model.h5')  #Creates an HDF5 file 'model.h5'

# Since we're trying to predict steering wheel angle this is a regression network not a classification network so we don't need activation