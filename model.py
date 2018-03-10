import csv
import cv2
import numpy as np
from keras.models import Sequential
from keras.layers import Lambda, Conv2D, MaxPooling2D, Dropout, Dense, Flatten, Cropping2D
from keras.optimizers import Adam
from utils import INPUT_SHAPE, batch_generator

def path(i):
        source_path = line[i]
        filename = source_path.split('/')[-1]
        current_path = '/home/carnd/data/IMG' + filename
        return current_path


lines = []
with open('/home/carnd/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        lines.append(line)


images = []
measurements = []
images_center = []
images_left = []
images_right = []
for line in lines:
    for i in range(3):
        # create adjusted steering measurements for the side camera images
        steering_center = float(line[3])
        correction = 0.2 # this is a parameter to tune
        steering_left = steering_center + correction
        steering_right = steering_center - correction


        image_center = cv2.imread(path(0))
        image_left = cv2.imread(path(1))
        image_right = cv2.imread(path(2))
        images_center.append(image_center)
        images_left.append(image_center)
        images_right.append(image_center)

        images.extend(img_center, img_left, img_right)
        measurements.extend(steering_center, steering_left, steering_right)

augmented_images, augmented_measurements = [], []
for image,measurement in zip(images, measurements):
    augmented_images.append(image)
    augmented_measurements.append(measurement)
    augmented_images.append(cv2.flip(image,1))
    augmented_measurements.append(measurement * -1.0)

X_train = np.array(augmented_images)
y_train = np.array(augmented_measurements)

# model
model = Sequential()
model.add(Lambda(lambda x: x/255.0 - 0.5, input_shape(160,320,3)))
model.add(Cropping2D(cropping = ((70,25),(0,0))))
model.add(Conv2D(24, 5, 5, activation='elu', subsample=(2, 2)))
model.add(Conv2D(36, 5, 5, activation='elu', subsample=(2, 2)))
model.add(Conv2D(48, 5, 5, activation='elu', subsample=(2, 2)))
model.add(Conv2D(64, 3, 3, activation='elu'))
model.add(Conv2D(64, 3, 3, activation='elu'))
model.add(Dropout(keep_prob = 0.5))
model.add(Flatten())
model.add(Dense(100, activation='elu'))
model.add(Dense(50, activation='elu'))
model.add(Dense(10, activation='elu'))
model.add(Dense(1))
model.summary()

# train model
model.compile(loss='mse', optimizer = 'adam')
model.fit(X_train, y_train, validation_split = 0.2, shuffle = True, nb_epoch = 7)

model.save('model.h5')
