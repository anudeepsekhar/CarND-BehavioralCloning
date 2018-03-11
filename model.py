import csv
import cv2
import numpy as np
from keras.models import Sequential
from keras.layers import Lambda, Conv2D, MaxPooling2D, Dropout, Dense, Flatten, Cropping2D
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle


samples = []
with open('/home/carnd/data/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for sample in reader:
        samples.append(sample)
print(len(samples))
train_samples, validation_samples = train_test_split(samples, test_size=0.2)

def generator(samples, batch_size=32):
    num_samples = len(samples)
    while 1:
        sample = shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]
            images = []
            measurements = []
            for batch_sample in batch_samples:
                #print(batch_sample)
                for i in range(3):
                    correction = 0.2
                    source_path = batch_sample[i]
                    filename = source_path.split("\\")[-1]
                #   print(filename)
                    current_path = '/home/carnd/data/IMG/' + filename
                    image_center = cv2.imread(current_path)
                #   print(image_center.shape)
                    images.append(image_center)
                    if(i==0):
                        measurements.append(float(batch_sample[3]))
                    elif(i==1):
                        measurements.append(float(batch_sample[3])+correction)
                    elif(i==2):
                        measurements.append(float(batch_sample[3])-correction)

                augmented_images, augmented_measurements = [], []
                for image,measurement in zip(images, measurements):
                    augmented_images.append(image)
                    augmented_measurements.append(measurement)
                    augmented_images.append(cv2.flip(image,1))
                    augmented_measurements.append(measurement * -1.0)
                X_train = np.array(augmented_images)
                y_train = np.array(augmented_measurements)
                #print(X_train.shape)
                yield X_train, y_train


train_generator = generator(train_samples, batch_size=50)
validation_generator = generator(validation_samples, batch_size=50)

# model
model = Sequential()
model.add(Lambda(lambda x: x/255.0 - 0.5, input_shape=(160,320,3)))
model.add(Cropping2D(cropping = ((70,25),(0,0))))
model.add(Conv2D(24, 5, 5, activation='elu', subsample=(2, 2)))
model.add(Conv2D(36, 5, 5, activation='elu', subsample=(2, 2)))
model.add(Dropout(0.5))
model.add(Conv2D(48, 5, 5, activation='elu', subsample=(2, 2)))
model.add(Conv2D(64, 3, 3, activation='elu'))
model.add(Dropout(0.5))
model.add(Conv2D(64, 3, 3, activation='elu'))
model.add(Dropout(0.5))
model.add(Flatten())
model.add(Dense(100, activation='elu'))
model.add(Dense(50, activation='elu'))
model.add(Dropout(0.5))
model.add(Dense(10,activation='elu'))
model.add(Dense(1))
model.summary()

# train model
model.compile(loss='mse', optimizer='adam')
model.fit_generator(train_generator, samples_per_epoch= 80000,validation_data=validation_generator, nb_val_samples=len(validation_samples), nb_epoch=5)

model.save('model1.h5')

