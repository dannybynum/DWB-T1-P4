#Keras - front end interface to TensorFlow

############THE VERY BASICS#####################################################
# The keras.models.Sequential class is a wrapper for the neural network model. 
# It provides common functions like fit(), evaluate(), and compile().

from keras.models import Sequential

# Create the Sequential model
model = Sequential()

#1st Layer - Add a flatten layer
model.add(Flatten(input_shape=(32, 32, 3)))

#2nd Layer - Add a fully connected layer (the '100' is the output shape)
model.add(Dense(100))

#3rd Layer - Add a ReLU activation layer
model.add(Activation('relu'))

#4th Layer - Add a fully connected layer - output of last layer should be equal to the number of
#"classes" for a classification problem ... for example MNST (0-9 digits - set to 10), traffic sign classifer was 43 classes
#This example shows '60'
model.add(Dense(60))

#5th Layer - Add a ReLU activation layer
model.add(Activation('relu'))

# Keras will automatically infer the shape of all layers after the first layer. 
# This means you only have to set the input dimensions for the first layer.
# The first layer from above, model.add(Flatten(input_shape=(32, 32, 3))), sets the input dimension to (32, 32, 3) 
# and output dimension to (3072=32 x 32 x 3). The second layer takes in the output of the first layer and sets the 
# output dimensions to (100). This chain of passing output to the next layer continues until the last layer, 
#which is the output of the model.


#####################################################################################
import pickle
import numpy as np
import tensorflow as tf

# Load pickled data
with open('small_train_traffic.p', mode='rb') as f:
    data = pickle.load(f)

# split data
X_train, y_train = data['features'], data['labels']

# Setup Keras
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Flatten, Dropout
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPooling2D

# ONE EXAMPLE OF A FULLY CONNECTED NETWORK with the following characteristics:
# Set the first layer to a Flatten() layer with the input_shape set to (32, 32, 3).
# Set the second layer to a Dense() layer with an output width of 128.
# Use a ReLU activation function after the second layer.
# Set the output layer width to 5, because for this data set there are only 5 classes.
# Use a softmax activation function after the output layer.
# Train the model for 3 epochs. You should be able to get over 50% training accuracy.
my_model = Sequential()
my_model.add(Flatten(input_shape=(32, 32, 3)))
my_model.add(Dense((128), activation = 'relu'))
#my_model.add(Activation('relu'))
my_model.add(Dense((5), activation = 'softmax'))
#my_model.add(Activation('softmax'))

#ONE EXAMPLE OF CONVOLUTIONAL
# Building from the previous network...
# Add a convolutional layer with 32 filters, a 3x3 kernel, and valid padding before the flatten layer.
#Add a ReLU activation after the convolutional layer.
#Add a 2x2 max pooling layer immediately following your convolutional layer.
#Add a dropout layer after the pooling layer. Set the dropout rate to 50%. (Note in Keras this is the prob to drop)

model = Sequential()
model.add(Conv2D(32, (3, 3), input_shape=(32, 32, 3)))
model.add(MaxPooling2D((2, 2)))
model.add(Dropout(0.5))
model.add(Activation('relu'))
model.add(Flatten())
model.add(Dense(128))
model.add(Activation('relu'))
model.add(Dense(5))
model.add(Activation('softmax'))


# preprocess data
X_normalized = np.array(X_train / 255.0 - 0.5 )

from sklearn.preprocessing import LabelBinarizer
label_binarizer = LabelBinarizer()
y_one_hot = label_binarizer.fit_transform(y_train)

my_model.compile('adam', 'categorical_crossentropy', ['accuracy'])
# Can change the number of epochs here
history = my_model.fit(X_normalized, y_one_hot, epochs=3, validation_split=0.2)



# evaluate model against the test data
with open('small_test_traffic.p', 'rb') as f:
    data_test = pickle.load(f)

X_test = data_test['features']
y_test = data_test['labels']

# preprocess data
X_normalized_test = np.array(X_test / 255.0 - 0.5 )
y_one_hot_test = label_binarizer.fit_transform(y_test)

print("Testing")

metrics = model.evaluate(X_normalized_test, y_one_hot_test)
for metric_i in range(len(model.metrics_names)):
    metric_name = model.metrics_names[metric_i]
    metric_value = metrics[metric_i]
    print('{}: {}'.format(metric_name, metric_value))    



###########################From Lesson 18 Transfer Learning, Section 9. VGG##########################
# Load our images first, and we'll check what we have
from glob import glob
import matplotlib.image as mpimg
import matplotlib.pyplot as plt

image_paths = glob('images/*.jpg')

# Print out the image paths
print(image_paths)

# View an example of an image
example = mpimg.imread(image_paths[0])
plt.imshow(example)
plt.show()

# Here, we'll load an image and pre-process it
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input
import numpy as np

i = 1 # Can change this to your desired image to test
img_path = image_paths[i]
img = image.load_img(img_path, target_size=(224, 224))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
x = preprocess_input(x)

# Note - this will likely need to download a new version of VGG16
from keras.applications.vgg16 import VGG16, decode_predictions

# Load the pre-trained model
model = VGG16(weights='imagenet')

# Perform inference on our pre-processed image
predictions = model.predict(x)

# Check the top 3 predictions of the model
print('Predicted:', decode_predictions(predictions, top=3)[0])


###########################From Lesson 18 Transfer Learning, Section 14. Lab on Transfer Learning##########################

# Welcome to the lab on Transfer Learning! Here, you'll get a chance to try out training a network with ImageNet 
# pre-trained weights as a base, but with additional network layers of your own added on. You'll also get to see 
# the difference between using frozen weights and training on all layers.


# Set a couple flags for training - you can ignore these for now
freeze_flag = True  # `True` to freeze layers, `False` for full training
weights_flag = 'imagenet' # 'imagenet' or None
preprocess_flag = True # Should be true for ImageNet pre-trained typically

# Loads in InceptionV3
from keras.applications.inception_v3 import InceptionV3

# We can use smaller than the default 299x299x3 input for InceptionV3
# which will speed up training. Keras v2.0.9 supports down to 139x139x3
input_size = 139

# Using Inception with ImageNet pre-trained weights
inception = InceptionV3(weights=weights_flag, include_top=False,
                        input_shape=(input_size,input_size,3))

if freeze_flag == True:
    ## TODO: Iterate through the layers of the Inception model
    ##       loaded above and set all of them to have trainable = False
    for layer in inception.layers:
        layer.trainable = False
#print("Model Layers", inception.layers)

## TODO: Use the model summary function to see all layers in the
##       loaded Inception model
inception.summary()

from keras.layers import Input, Lambda
import tensorflow as tf

# Makes the input placeholder layer 32x32x3 for CIFAR-10
cifar_input = Input(shape=(32,32,3))

# Re-sizes the input with Kera's Lambda layer & attach to cifar_input
resized_input = Lambda(lambda image: tf.image.resize_images( 
    image, (input_size, input_size)))(cifar_input)

# Feeds the re-sized input into Inception model
# You will need to update the model name if you changed it earlier!
inp = inception(resized_input)

# Imports fully-connected "Dense" layers & Global Average Pooling
from keras.layers import Dense, GlobalAveragePooling2D

## TODO: Setting `include_top` to False earlier also removed the
##       GlobalAveragePooling2D layer, but we still want it.
##       Add it here, and make sure to connect it to the end of Inception
x1 = GlobalAveragePooling2D()(inp)

## TODO: Create two new fully-connected layers using the Model API
##       format discussed above. The first layer should use `out`
##       as its input, along with ReLU activation. You can choose
##       how many nodes it has, although 512 or less is a good idea.
##       The second layer should take this first layer as input, and
##       be named "predictions", with Softmax activation and 
##       10 nodes, as we'll be using the CIFAR10 dataset.
x2 = Dense(512, activation = 'relu')(x1)
predictions = Dense(10, activation = 'softmax')(x2)

# Imports the Model API
from keras.models import Model

# Creates the model, assuming your final layer is named "predictions"
model = Model(inputs=cifar_input, outputs=predictions)

# Compile the model
model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Check the summary of this new model to confirm the architecture
model.summary()

#GPU TIME!!
from sklearn.utils import shuffle
from sklearn.preprocessing import LabelBinarizer
from keras.datasets import cifar10

(X_train, y_train), (X_val, y_val) = cifar10.load_data()

# One-hot encode the labels
label_binarizer = LabelBinarizer()
y_one_hot_train = label_binarizer.fit_transform(y_train)
y_one_hot_val = label_binarizer.fit_transform(y_val)

# Shuffle the training & test data
X_train, y_one_hot_train = shuffle(X_train, y_one_hot_train)
X_val, y_one_hot_val = shuffle(X_val, y_one_hot_val)

# We are only going to use the first 10,000 images for speed reasons
# And only the first 2,000 images from the test set
X_train = X_train[:10000]
y_one_hot_train = y_one_hot_train[:10000]
X_val = X_val[:2000]
y_one_hot_val = y_one_hot_val[:2000]


# Use a generator to pre-process our images for ImageNet
from keras.preprocessing.image import ImageDataGenerator
from keras.applications.inception_v3 import preprocess_input

if preprocess_flag == True:
    datagen = ImageDataGenerator(preprocessing_function=preprocess_input)
    val_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)
else:
    datagen = ImageDataGenerator()
    val_datagen = ImageDataGenerator()



# Train the model
batch_size = 32
epochs = 5
# Note: we aren't using callbacks here since we only are using 5 epochs to conserve GPU time
model.fit_generator(datagen.flow(X_train, y_one_hot_train, batch_size=batch_size), 
                    steps_per_epoch=len(X_train)/batch_size, epochs=epochs, verbose=1, 
                    validation_data=val_datagen.flow(X_val, y_one_hot_val, batch_size=batch_size),
                    validation_steps=len(X_val)/batch_size)