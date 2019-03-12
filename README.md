# Behavioral Cloning Project

[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)

Note: I have included a copy of this README and associated files that I created on my GitHub page.  [*Dannys_Project*](https://github.com/dannybynum/DWB-T1-P4)

## Background / Reference
---
The overall goal of this project is for students to create and train a neural network to predict a steering wheel angle for a given input image.  The images are gathered with a driving simulator with manual steering & throttle control and then the output of the trained neural network is tested on the same simulator (placed into Autonomous mode).  The Udacity team has provided some cool tools to help with this process. Here is a short desription of the tools that were provided as a part of the project (See the assoicated github page for more details - [*LINK*](https://github.com/udacity/CarND-Behavioral-Cloning-P3)):

* Driving Simulator - via a Workspace with connection to remote GPU-Enabled Linux Machine.  Source Files also available at: [*LINK*](https://github.com/udacity/self-driving-car-sim)
* Driving Simulator Training Mode - by manually driving the car (using arrow keys and mouse) and hitting the "record" button on the simulator you can record all of the training images + labels/ground-truth (steering angles) associated with these images --- the program creates a .csv file that shows the file name of each image and the associated steering wheel angle.  ALL of this is provided.
* `drive.py`  - program that sets up connection to feed real-time images from the Simulator in Autonomous mode directly into your trained neural network and send associated steering wheel angle prediction back to the Simulator to steer the car.
* `video.py` - program that stitches together the single images gathered by the simulator into a video - very helpful for creating output video required to be submitted.

**The files that I generated are as follows:**
* `dwb_network_v7.py` (aka model.py) -- This is the main work content I created.  This script has the Keras implemenation of the Neural Network that I choose and it loads the captured images (via a generator) into the this network and performs training and then saves the assoicated model.
* `dwb_model_v7.h5` (aka model.h5) -- This is the trained Keras model - given an input image it can successfully predict the steering wheel angle that keeps the car moving and near the center of the road for the entire track provided in the Simulator.
* `vid.mp4` -- PROOF :-).  This video of the vehicle driving in autonomous mode was created using the model that I trained and saved with the script that I created for this project - the car drives around the whole track successfully and meets project 'rubric' criteria:
_No tire may leave the drivable portion of the track surface. The car may not pop up onto ledges or roll over any surfaces that would otherwise be considered unsafe (if humans were in the vehicle)._



**Steps Followed To Complete the Project**
* Use the simulator to collect data of good driving behavior (had to practice some before being profficient enough to capture "good" ground truth behavior)
* Design, train and validate a model that predicts a steering angle from image data (I did ~7 iterations of models training and testing before I had a solution that worked around the whole track)
* Use the model to drive the vehicle autonomously around the first track in the simulator. The vehicle should remain on the road for an entire loop around the track.
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./report_images/Training_Loss_V7.PNG "Training Results for Version 7"
[image2]: ./report_images/IMG_0480.JPG "Bridge Recovery Training Image"
[image3]: ./report_images/IMG_0481.JPG "Tight Turn Recovery Training Image"



##Model/Architecture Strategy and Documentation (7 steps to success!)
---

The course material walked through several examples and even some code and discussed it line-by-line which was very cool to have as a starting place.  There were still quite a few decisions to make and here is the process that I followed in order to end up with a working model for this project.

**Version 1** `dwb_network_v1.py`
---
_Purpose/Content:_  Followed the suggestion given to create a very basic model just to test the whole _pipeline_ end-to-end.
_Training Data Used:_  Just used the training data that was provided - I didn't realize it copied into the `/home/workspaces/` directory as soon as you started GPU mode so I actually downloaded and copied the data over to the `/opt` folder
_Main Features added:_ Basic Implementation of reading the files and assoicated labels (from .csv file) and creating `X_train` and `y_train` sets.
_Testing Results:_ Just like in lesson video the car basically just sat in middle and steering wheel went back and forth.
_Relevant Code Snipets:_  The whole model fits on 8 lines (comment lines removed):
```python
from keras.models import Sequential
from keras.layers import Flatten, Dense
MostBasicModel = Sequential()   #Type of model
MostBasicModel.add(Flatten(input_shape = (160,320,3)))
MostBasicModel.add(Dense (1))
MostBasicModel.compile(loss='mse', optimizer='adam')
MostBasicModel.fit(X_train, y_train, validation_split=0.2,shuffle=True,nb_epoch=5)
MostBasicModel.save('dwb_model.h5')  #Creates an HDF5 file 'model.h5'
```


**Version 2** `dwb_network_v2.py`
---
_Purpose/Content:_  Try to make a big jump and see if its enough for early success. 
_Training Data Used:_  Still using the example set provided, but this time using it directly from workspace where it already exists when enabling GPU mode.
_Main Features added:_ (a) Implemented LeNet Architecture, (b) Added Preprocessing, (c) Added Cropping of upper 70 and lower 25pixels, (d) mixed in a little Dropout after EVERY layer (used values of 0.1, 0.2 and 0.05 - somewhat randomly chosen but stuck with low values because didn't want big effect here yet)
_Testing Results:_ It drove for a little bit but then ended up at the bottom of the lake.  (at least it applied the brakes some as it was going in!)
_Relevant Code Snipets:_  Here is a portion of the model build-up for version 2:
```python
dwb_model.add(Conv2D(6, (5, 5)))               #Layer1 - LeNet Convolutional Layer1 is 6 filters at size of 5x5
dwb_model.add(Activation('relu'))
dwb_model.add(MaxPooling2D())                  #After the first layer ther eis 2x2 pooling, 2x2 is default?
dwb_model.add(Dropout(0.1))                    #This is not in original LeNet - adding a Dropout layer after each pooling layer
```



**Version 3** `dwb_network_v3.py` --- unsuccessful, had bugs, moved on to version 4
---
_Purpose/Content:_  Implement the generator because I may want to collect a large training set and I don't want to run into memory limitations.
_Training Data Used:_  No Change.  Still using Example Set Provided (about ~8000 images)
_Main Features added:_ ONLY change was to add the generator - this was done intentionally to make sure this didn't mess anything up
_Testing Results:_ Results regressed from what I had, saved this and moved into version 4 to fix the bugs
_Relevant Code Snipets:_  
```python

def generator(samples, batch_size=32):
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]
[....]
```


**Version 4** `dwb_network_v4.py`  Same as Version 3 except fixed bugs
---
_Bug Fixes From Version 3_
_Testing Results:_ Same results with car as expected - drove for a while and then drove into lake.


**Version 5** `dwb_network_v5.py`
---
_Purpose/Content:_  Start using the training data that I collected
_Training Data Used:_  Changed to my initial collection of training data.
_Main Features added:_ No major changes, model identical --- just changed to using the training data that I collected with the Simulator
_Testing Results:_ Results got worse if anything from the initial training set that I created.  This signaled to me that I needed to spend more time getting profficient with using the simulator.
_Relevant Code Snipets:_  Only change was the pointer to the data now saved in a directory I created for my training data recordings
```python
samples = []
with open('/home/workspace/dwb_record_v5/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
[......]

```


**Version 6** `dwb_network_v6.py`
---
_Purpose/Content:_  Between version 5 and version 6 I tried a LOT of different approaches for training (capturing new sets, etc) - this will be described more in the training strategy description / writeup.
_Training Data Used:_  Various and Many :-).  
_Main Features added:_  Tried different training sets and iterations (adding more scenes/scenarios to main set) and also tired different numbers of ephocs
_Testing Results:_ Very frustratingly the results were largely the same -- one or two areas would improve with new data but something else would get worse and eventually the car would always drive off the track
_Relevant Code Snipets:_  Just changed the save name to keep track of different models in case I wanted to replay one, for example: `dwb_model.save('dwb_model_v6-4.h5')`


**Version 7** `dwb_network_v7.py`
---
_Purpose/Content:_  Switch to a more powerful architecture - mostly using the one suggested in the course from Nvidia.
_Training Data Used:_  I put my best foot forward here and followed my best method to come up with what I thought would be a good set - more notes on this in the writeup on training strategy.
_Main Features added:_ Switched to Architecture published by Nvidia.  (kept the fully connected layers same size as LeNet)
_Testing Results:_ It worked the first time!  Car drove all the way around the track.
_Relevant Code Snipets:_  The full model that worked is here:
```python
dwb_model = Sequential()                                                #Setting up model type
dwb_model.add(Lambda(lambda x : x/255.0-0.5,input_shape = (160,320,3))) #Keep same pre-processing step
dwb_model.add(Cropping2D(cropping=((70,25),(0,0))))                       
dwb_model.add(Conv2D(24,5,5,subsample=(2,2),activation='relu'))               
dwb_model.add(Dropout(0.1))                    							#Kept dropout after 1st Conv layer like I had been using
dwb_model.add(Conv2D(36,5,5,subsample=(2,2),activation='relu'))              
dwb_model.add(Dropout(0.1))                    							#Kept dropout after 2nd Conv layer like I had been using
dwb_model.add(Conv2D(48,5,5,subsample=(2,2),activation='relu'))			#Nvidia has 5 Convolution layers instead of just 2 like LeNet
dwb_model.add(Conv2D(64,3,3,activation='relu'))							#Nvidia has 5 Convolution layers instead of just 2 like LeNet
dwb_model.add(Conv2D(64,3,3,activation='relu'))							#Nvidia has 5 Convolution layers instead of just 2 like LeNet
dwb_model.add(Flatten())                      
dwb_model.add(Dense(120))                      							#Kept LeNet size of 120 vs Nvidia 100
dwb_model.add(Dropout(0.2))  											#Kept dropout after first fully connected layer
dwb_model.add(Dense(84))                      						    #Kept LeNet size of 84 vs Nvidia 50
dwb_model.add(Dropout(0.05))  											#Kept dropout after second fully connected layer
dwb_model.add(Dense(1))                      							#Final output is only 1 value - steering wheel angle

```

The training performance achieved with this last model (Version 7) and assoicated training data (20% validation set split off after shuffle):
![alt text][image1]



##Training Strategy

The build up of the training data is shown in the table below:

| Sequence   |     Short Description 						     |           Rationale						  |
|:----------:|:---------------------------------------------:    |:-------------------------------------------| 
|     1      | Base set - smooth driving with __mouse__ ~0.75laps | I didn't want to overdo the smooth driving - I was finding that the car was sometimes hugging the rails and not reacting so was thinking it might be "overpowered" with too much "small angle" data| 
|     2      | Drive opposite direction around track with __mouse__ - smooth driving for ~0.5laps| I didn't want to have a left turn bias in the data so I found some relative small angle sections and drove them in reverse direction| 
|     3      | Bridge data - straight drive multiple runs| Lesson mentioned the bridge may look different so I wanted to have a decent representation of the bridge in my data | 
|     4      | Bridge Data part II - recovery from side of bridge and repeated ~5-6 times the transition from bridge to road| I was seeing in some of my testing that the car would have a problem specifically trying to cover off of the bridge and get back to the normal road so I captured data specifically to deal with this| 
|     5      | Recovery driving - on side of road | Wanted the model to know what to do if the car was actually pointed off the road while still on it so I spent some time capturing this by stopping the car at an angle and cutting the wheel all the way to the correction position and recording some images with the car moving very slowly back to the correct position - lots of images captured with extreme steering wheel so this should help keep the car from maintaining a small angle and veering off the road if it is pointed in the wrong direction| 
|     6      | Tight Turns (red/white stripes)- 3-4 runs on 2 sets of curves| The red/white marked sections are clearly different looking than the rest of the track so spent some focused time here and was very careful to have a correct steering wheel angle - no undercorrection or overcorrection - at least in most cases | 

Below are some images of "recovery" training:
![alt text][image2]
![alt text][image3]




















## Details About Files In This Directory

### `drive.py`

Usage of `drive.py` requires you have saved the trained model as an h5 file, i.e. `model.h5`. See the [Keras documentation](https://keras.io/getting-started/faq/#how-can-i-save-a-keras-model) for how to create this file using the following command:
```sh
model.save(filepath)
```

Once the model has been saved, it can be used with drive.py using this command:

```sh
python drive.py model.h5
```

The above command will load the trained model and use the model to make predictions on individual images in real-time and send the predicted angle back to the server via a websocket connection.

Note: There is known local system's setting issue with replacing "," with "." when using drive.py. When this happens it can make predicted steering values clipped to max/min values. If this occurs, a known fix for this is to add "export LANG=en_US.utf8" to the bashrc file.

#### Saving a video of the autonomous agent

```sh
python drive.py model.h5 run1
```

The fourth argument, `run1`, is the directory in which to save the images seen by the agent. If the directory already exists, it'll be overwritten.

```sh
ls run1

[2017-01-09 16:10:23 EST]  12KiB 2017_01_09_21_10_23_424.jpg
[2017-01-09 16:10:23 EST]  12KiB 2017_01_09_21_10_23_451.jpg
[2017-01-09 16:10:23 EST]  12KiB 2017_01_09_21_10_23_477.jpg
[2017-01-09 16:10:23 EST]  12KiB 2017_01_09_21_10_23_528.jpg
[2017-01-09 16:10:23 EST]  12KiB 2017_01_09_21_10_23_573.jpg
[2017-01-09 16:10:23 EST]  12KiB 2017_01_09_21_10_23_618.jpg
[2017-01-09 16:10:23 EST]  12KiB 2017_01_09_21_10_23_697.jpg
[2017-01-09 16:10:23 EST]  12KiB 2017_01_09_21_10_23_723.jpg
[2017-01-09 16:10:23 EST]  12KiB 2017_01_09_21_10_23_749.jpg
[2017-01-09 16:10:23 EST]  12KiB 2017_01_09_21_10_23_817.jpg
...
```

The image file name is a timestamp of when the image was seen. This information is used by `video.py` to create a chronological video of the agent driving.

### `video.py`

```sh
python video.py run1
```

Creates a video based on images found in the `run1` directory. The name of the video will be the name of the directory followed by `'.mp4'`, so, in this case the video will be `run1.mp4`.

Optionally, one can specify the FPS (frames per second) of the video:

```sh
python video.py run1 --fps 48
```

Will run the video at 48 FPS. The default FPS is 60.
