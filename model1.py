import csv
import cv2
import numpy as np
from keras.models import Sequential
from keras.layers import Lambda, Conv2D, MaxPooling2D, Dropout, Dense, Flatten, Cropping2D
from keras.optimizers import Adam

images = []
measurements = []

lines = []
with open('/home/carnd/data/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        lines.append(line)
print(len(lines))

for line in lines:
    for i in range(3):
        correction = 0.2
        source_path = line[i]
        filename = source_path.split("\\")[-1]
#       print(filename)
        current_path = '/home/carnd/data/IMG/' + filename
        image_center = cv2.imread(current_path)
#       print(image_center.shape)
        images.append(image_center)
        if(i==0):
            measurements.append(float(line[3]))
        elif(i==1):
            measurements.append(float(line[3])+correction)
        elif(i==2):
            measurements.append(float(line[3])-correction)

augmented_images, augmented_measurements = [], []
#for image,measurement in zip(images, measurements):
#    augmented_images.append(image)
#    augmented_measurements.append(measurement)
#    augmented_images.append(cv2.flip(image,1))
#    augmented_measurements.append(measurement * -1.0)

#X_train = np.array(augmented_images)
#y_train = np.array(augmented_measurements)
X_train = np.array(images)
y_train = np.array(measurements)
print(X_train.shape)
img_rows, img_cols = 160, 320
#x_train =X_train.reshape(X_train.shape[0], img_rows, img_cols, 3)
#x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)

# model
model = Sequential()
model.add(Lambda(lambda x: x/255.0 - 0.5, input_shape=(160,320,3)))
model.add(Cropping2D(cropping = ((70,25),(0,0))))
model.add(Conv2D(24, 5, 5, activation='elu', subsample=(2, 2)))
model.add(Conv2D(36, 5, 5, activation='elu', subsample=(2, 2)))
model.add(Conv2D(48, 5, 5, activation='elu', subsample=(2, 2)))
model.add(Conv2D(64, 3, 3, activation='elu'))
model.add(Conv2D(64, 3, 3, activation='elu'))
model.add(Dropout(0.5))
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

