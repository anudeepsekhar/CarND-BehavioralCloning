import cv2
import csv
import numpy as np
import os


def getCsvLines():
    lines = []
    with open('/home/carnd/data1/driving_log.csv') as csvfile:
        reader = csv.reader(csvfile)
        for line in reader:
            lines.append(line)
#    print(len(lines))
    return lines


def getdata(lines):
    left_images = []
    center_images = []
    right_images = []
    
    m_tot = []
    for line in lines:
        image_center = []
        image_left = []
        image_right = []
        measurements = []
        source_path = '/home/carnd/data1/IMG/'
        c_path = line[0].split("\\")[-1]
#        print(c_path)
        l_path = line[1].split("\\")[-1]
        r_path = line[2].split("\\")[-1]
        measurement=line[3]
        image_left.append(source_path + l_path)
        image_center.append(source_path + c_path)
        image_right.append(source_path + r_path)
        measurements.append(measurement)
        left_images.extend(image_left)
        right_images.extend(image_right)
        center_images.extend(image_center)
        m_tot.extend(measurements)
    print(len(m_tot))
    return left_images,right_images,center_images,m_tot


def combine_data(left_images,right_images,center_images,measurement):
    image_paths = []
    image_paths.extend(center_images)
    image_paths.extend(left_images)
    image_paths.extend(right_images)
    measurements = []
    measurements.extend(measurement)
    measurements.extend([float(measurement[x]) + 0.2 for x in range(len(measurement))])
    measurements.extend([float(measurement[x]) - 0.2 for x in range(len(measurement))])
    print(len(measurements))
    return image_paths,measurements 

import sklearn


def generator(samples, batch_size=32):
    num_samples = len(samples)
    while 1:  # Loop forever so the generator never terminates
        samples = sklearn.utils.shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset: offset + batch_size]

            images = []
            angles = []
            for imagePath, measurement in batch_samples:
                originalImage = cv2.imread(imagePath)
                image = cv2.cvtColor(originalImage, cv2.COLOR_BGR2RGB)
                images.append(image)
                angles.append(float(measurement))
                # Flipping
                images.append(cv2.flip(image, 1))
                angles.append(float(measurement) * -1.0)

            # trim image to only see section with road
            inputs = np.array(images)
            outputs = np.array(angles)
            #(inputs,outputs)=sklearn.utils.shuffle(inputs, outputs)
            yield (inputs,outputs)


from keras.models import Sequential, Model
from keras.layers import Flatten, Dense, Lambda, Convolution2D, Cropping2D
from keras.layers.pooling import MaxPooling2D
import matplotlib.pyplot as plt


def model():
    model = Sequential()
    model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=(160,320,3)))
    model.add(Cropping2D(cropping=((50,20), (0,0))))
    model.add(Convolution2D(24,5,5, subsample=(2,2), activation='relu'))
    model.add(Convolution2D(36,5,5, subsample=(2,2), activation='relu'))
    model.add(Convolution2D(48,5,5, subsample=(2,2), activation='relu'))
    model.add(Convolution2D(64,3,3, activation='relu'))
    model.add(Convolution2D(64,3,3, activation='relu'))
    model.add(Flatten())
    model.add(Dense(100))
    model.add(Dense(50))
    model.add(Dense(10))
    model.add(Dense(1))
    return model

lines = getCsvLines()
left_images,right_images,center_images,measurements = getdata(lines)
image_paths,measurements = combine_data(left_images,right_images,center_images,measurements)
print(len(image_paths))
print(len(measurements))
from sklearn.model_selection import train_test_split
samples = list(zip(image_paths, measurements))
train_samples, validation_samples = train_test_split(samples, test_size=0.2)

train_generator = generator(train_samples, batch_size=32)
validation_generator = generator(validation_samples, batch_size=32)
print(len(samples))
model = model()

model.compile(loss='mse', optimizer='adam')
#history_object = model.fit_generator(train_generator, samples_per_epoch=len(train_samples), validation_data=validation_generator, nb_val_samples=len(validation_samples), nb_epoch=3, verbose=1)
model.fit_generator(train_generator, samples_per_epoch= len(train_samples),nb_epoch =5,validation_data=validation_generator, nb_val_samples=len(validation_samples), verbose = 1)

model.save('model3.h5')
