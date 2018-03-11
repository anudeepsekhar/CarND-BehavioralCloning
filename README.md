# CarND-BehavioralCloning

# Project

## Goals

The goals / steps of this project are the following:

- Use the simulator to collect data of good driving behavior
- Build, a convolution neural network in Keras that predicts steering angles from images
- Train and validate the model with a training and validation set
- Test that the model successfully drives around track one without leaving the road
- Summarize the results with a written report

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model_final.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model_final.h5 containing a trained convolution neural network
* writeup_report.md or writeup_report.pdf summarizing the results

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing
```sh
python drive.py model_final.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

I started off by trial and error, experimenting with different model architectures by adding different layers and measuring how the model performed.
Few of them were able to make the car head in the right direction but failed at sharp turns. Later  I decided to try the nVidia Autonomous Car Group model, and the car drove the complete first track after just 5 training epochs.

The summary of the model is given below:
```
    ____________________________________________________________________________________________________
    Layer (type)                     Output Shape          Param #     Connected to
    ====================================================================================================
    lambda_1 (Lambda)                (None, 160, 320, 3)   0           lambda_input_1[0][0]
    ____________________________________________________________________________________________________
    cropping2d_1 (Cropping2D)        (None, 90, 320, 3)    0           lambda_1[0][0]
    ____________________________________________________________________________________________________
    convolution2d_1 (Convolution2D)  (None, 43, 158, 24)   1824        cropping2d_1[0][0]
    ____________________________________________________________________________________________________
    convolution2d_2 (Convolution2D)  (None, 20, 77, 36)    21636       convolution2d_1[0][0]
    ____________________________________________________________________________________________________
    convolution2d_3 (Convolution2D)  (None, 8, 37, 48)     43248       convolution2d_2[0][0]
    ____________________________________________________________________________________________________
    convolution2d_4 (Convolution2D)  (None, 6, 35, 64)     27712       convolution2d_3[0][0]
    ____________________________________________________________________________________________________
    convolution2d_5 (Convolution2D)  (None, 4, 33, 64)     36928       convolution2d_4[0][0]
    ____________________________________________________________________________________________________
    flatten_1 (Flatten)              (None, 8448)          0           convolution2d_5[0][0]
    ____________________________________________________________________________________________________
    dense_1 (Dense)                  (None, 100)           844900      flatten_1[0][0]
    ____________________________________________________________________________________________________
    dense_2 (Dense)                  (None, 50)            5050        dense_1[0][0]
    ____________________________________________________________________________________________________
    dense_3 (Dense)                  (None, 10)            510         dense_2[0][0]
    ____________________________________________________________________________________________________
    dense_4 (Dense)                  (None, 1)             11          dense_3[0][0]
    ====================================================================================================
    Total params: 981,819
    Trainable params: 981,819
    Non-trainable params: 0
    ____________________________________________________________________________________________________
```

#### 2. Attempts to reduce overfitting in the model

I decided not to modify the model by applying regularization techniques like Dropout or Max pooling as it made the models performance worse when I tried it with my dataset. Instead, I decided to keep the training epochs low. In addition
the model was trained and validated on different data sets to ensure that the model was not overfitting.  I split my sample data into training and validation data. Using 80% as training and 20% as validation. The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road and also driving anti-clockwise for one lap, as the track is turning clockwise, driving the car anti-clockwised help generalize the data.

For details about how I created the training data, see the next section.

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was to drive the car in the center of the lane smoothly.

My first step was to start with the most basic neural network possible and then went about adding new layers to it and gauging it performance. I thought this approach might be appropriate because because it would give me an insight into how each layer is effecting the performance of the network.
- First layer I added was an normalization layer to mean center the training data.
- Then my next observation was that the upper half of the frame was mostly trees and mountains which harm the networks performance more than improving it so i added a cropping layer to crop it off.
- In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I found that my first model had a low mean squared error on the training set but a high mean squared error on the validation set. This implied that the model was overfitting.
- Then I expiremented with adding more convolution layer and fully connected layers. I observed slight increase in performance but the car steered offcourse majority of the times.
- Then I decided to try the nVidia Autonomous Car Group model, and the car drove the complete first track after just 5 training epochs.
- Then I tried adding dropout layer to avoid overfitting but it made the models performance worse when I tried it with my dataset.

So to combat the overfitting, I decided not to modify the model rather I reduced the training epochs.

To improve the performance of the model I also used few preprocessing and data augmentation techniques which are discussed in the later sections,

The final step was to run the simulator to see how well the car was driving around track one.
At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.



![png](output_11_0.png)


#### 2. Final Model Architecture

The final model architecture consisted of a convolution neural network with the following layers and layer sizes:

Layer|Properties
------|----------
**Lambda**|**Input**: (160,320,3)
**Cropping2D**|**Cropping:** ([50,20],[0,0])
**Convolution 0**|**Convolution2D**
||**Kernel:** (5,5)
||**Activation:** Relu
||**Stride:** (2,2)
||**Filters:** (24)
**Convolution 1**|**Convolution2D**
||**Kernel:** (5,5)
||**Activation:** Relu
||**Stride:** (2,2)
||**Filters:** (36)
**Convolution2**|**Convolution2D**
||**Kernel:** (5,5)
||**Activation:** Relu
||**Stride:** (2,2)
||**Filters:** (48)
**Convolution3**|**Convolution2D**
||**Kernel:** (3,3)
||**Activation:** Relu
||**Stride:** (1,1)
||**Filters:** (64)
**Convolution4**|**Convolution2D**
||**Convolution2D**
||**Kernel:** (3,3)
||**Activation:** Relu
||**Stride:** (1,1)
||**Filters:** (64)
**Flatten**|
**Fully Connected 0**|**Dense**
||**Input:** 8448
||**Output:** 100
||**Activation:** Linear
**Fully Connected 1**|**Dense**
||**Input:** 100
||**Output:** 50
||**Activation:** Linear
**Fully Connected 2**|**Dense**
||**Input:** 50
||**Output:** 10
||**Activation:** Linear
**Fully Connected 3**|**Dense**
||**Input:** 10
||**Output:** 1
||**Activation:** Linear

#### 3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two laps on track one using center lane driving. Here is an example image of center lane driving:

![alt text][image2]

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to recove from the left side and right side.

Then I repeated this process on track two in order to get more data points.

To augment the data sat, I also flipped images and angles thinking that this would help generalize the data aswell as provides more data.

After the collection process, I had 12486 number of data points. I then preprocessed this data by converting all images to RGB using `cv2.cvtColor`

I finally randomly shuffled the data set and put 20% of the data into a validation set.

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 5. I used an adam optimizer so that manually training the learning rate wasn't necessary.

 
