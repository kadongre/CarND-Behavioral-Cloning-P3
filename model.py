import cv2
import h5py
import math
import json
import random
import numpy as np
import pandas as pd

import scipy
from scipy import misc

import keras
import keras.models as models

from sklearn.utils import shuffle
from keras.models import Sequential
from keras.layers import BatchNormalization
from keras.regularizers import l2, activity_l2
from keras.layers.convolutional import Convolution2D
from keras.layers import ELU, Lambda, Dense, Dropout, Flatten

train_dir = "./train/"
train_csv_file = train_dir + "driving_log.csv"
train_dataset = pd.read_csv(train_csv_file, usecols=['center', 'left', 'right', 'steering'])

test_dir = "./test/"
test_csv_file = test_dir + "driving_log.csv"
test_dataset = pd.read_csv(test_csv_file, usecols=['center', 'left', 'right', 'steering'])

def translate_image(image, steering_angle):
    rows, cols, channels = image.shape

    # shift image randomly to the left or right for upto 10 px
    shift_range = 50
    tr_x = shift_range*np.random.uniform()-shift_range/2
    
    # no shifts in vertical directions
    tr_y = 0

    translationM = np.float32([[1,0,tr_x],[0,1,tr_y]])
    new_image = cv2.warpAffine(image,translationM,(cols,rows))

    # compute the new steering angle applying a correction factor or 0.2 per pixel
    new_steering_angle = steering_angle + tr_x/shift_range*.4    
    return new_image,new_steering_angle


def flip_image(image, steering_angle):
    if (random.choice([True, False])):
        new_image = cv2.flip(image, 1)
        new_steering_angle = -steering_angle
    else:
        new_image = image
        new_steering_angle = steering_angle
        
    return new_image,new_steering_angle


def augment_brightness(image):
    new_image = cv2.cvtColor(image,cv2.COLOR_RGB2HSV)
    random_bright = .25+np.random.uniform()
    
    new_image[:,:,2] = new_image[:,:,2]*random_bright
    new_image = cv2.cvtColor(new_image,cv2.COLOR_HSV2RGB)

    return new_image


def crop_resize_image(image):
    rows, cols, channels = image.shape

    new_image = image[60:-20, : ]
    new_image = cv2.resize(image,(200, 66))    

    return new_image


def process_image(image_file, steering_angle):
    image = cv2.imread(image_file)
    image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
    translated_image, translated_steering_angle = translate_image(image, steering_angle)
    flipped_image, flipped_steering_angle = flip_image(translated_image, translated_steering_angle)
    brightness_adjusted_image = augment_brightness(flipped_image)
    cropped_resized_image = crop_resize_image(brightness_adjusted_image)

    return cropped_resized_image, flipped_steering_angle


def train_batch_generator(batch_size=128):
    while True:
        batch_train_X, batch_train_y = [], []

        for index, row in train_dataset.iterrows():
            steering_angle = row['steering']
            camera_offset = 0.0
            sel_cam_view = np.random.choice(['center', 'left', 'right'])
            if sel_cam_view == 'left':
                image_file = train_dir + row['left']
                camera_offset = 0.20
            elif sel_cam_view == 'right':
                image_file = train_dir + row['right']
                camera_offset = -0.30
            else:
                image_file = train_dir + row['center']

            processed_image, processed_steering_angle = process_image(image_file, steering_angle+camera_offset)
            batch_train_X.append(np.reshape(processed_image, (1, 66, 200, 3)))
            batch_train_y.append(np.array([[processed_steering_angle]]))

            if len(batch_train_X) == batch_size:
                # shuffle batch
                batch_train_X, batch_train_y, = shuffle(batch_train_X, batch_train_y, random_state=0)
                yield (np.vstack(batch_train_X), np.vstack(batch_train_y))
                batch_train_X, batch_train_y = [], []


def test_batch_generator(batch_size=128):
    while True:
        batch_test_X, batch_test_y = [], []

        for index, row in test_dataset.iterrows():
            steering_angle = row['steering']
            image_file = test_dir + row['center']

            image = cv2.imread(image_file)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = crop_resize_image(image)

            batch_test_X.append(np.reshape(image, (1, 66, 200, 3)))
            batch_test_y.append(np.array([[steering_angle]]))

            if len(batch_test_X) == batch_size:
                yield (np.vstack(batch_test_X), np.vstack(batch_test_y))
                batch_test_X, batch_test_y = [], []


def nvidia_model_v1():
    channels, rows, cols = 3, 66, 200

    model = Sequential()

    model.add(BatchNormalization(epsilon=0.001, mode=2, axis=1, input_shape=(rows, cols, channels)))

    model.add(Convolution2D(24, 5, 5, border_mode='valid', activation='relu', subsample=(2, 2)))
    model.add(Convolution2D(36, 5, 5, border_mode='valid', activation='relu', subsample=(2, 2)))
    model.add(Convolution2D(48, 5, 5, border_mode='valid', activation='relu', subsample=(2, 2)))
    model.add(Convolution2D(64, 3, 3, border_mode='valid', activation='relu', subsample=(1, 1)))
    model.add(Convolution2D(64, 3, 3, border_mode='valid', activation='relu', subsample=(1, 1)))
    model.add(Flatten())
    model.add(Dense(1164, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(100, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(50, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(10, activation='relu'))
    model.add(Dense(1, activation='tanh'))

    model.compile(optimizer="adam", loss="mse")

    return  model


def nvidia_model_v2():
    channels, rows, cols = 3, 66, 200

    model = Sequential()

    model.add(Lambda(lambda x: x/127.5 - 1.0,
                     input_shape = (rows, cols, channels),
                     output_shape = (rows, cols, channels)))

    model.add(Convolution2D(24, 5, 5, border_mode='valid', init='glorot_uniform', subsample=(2, 2), W_regularizer=l2(0.01)))
    model.add(ELU())
    model.add(Dropout(0.2))

    model.add(Convolution2D(36, 5, 5, border_mode='valid', init='glorot_uniform', subsample=(2, 2)))
    model.add(ELU())
    model.add(Dropout(0.2))

    model.add(Convolution2D(48, 5, 5, border_mode='valid', init='glorot_uniform', subsample=(2, 2)))
    model.add(ELU())
    model.add(Dropout(0.2))

    model.add(Convolution2D(64, 3, 3, border_mode='valid', init='glorot_uniform', subsample=(1, 1)))
    model.add(ELU())
    model.add(Dropout(0.2))

    model.add(Convolution2D(64, 3, 3, border_mode='valid', init='glorot_uniform', subsample=(1, 1)))
    model.add(ELU())
    model.add(Dropout(0.2))

    model.add(Flatten())

    model.add(Dense(100))
    model.add(ELU())
    model.add(Dropout(0.2))

    model.add(Dense(50))
    model.add(ELU())
    model.add(Dropout(0.2))

    model.add(Dense(10))
    model.add(ELU())

    model.add(Dense(1))

    model.compile(optimizer="adam", loss="mse")

    return  model


def comma_ai_model():
    channels, rows, cols = 3, 66, 200

    model = Sequential()
    model.add(Lambda(lambda x: x/127.5 - 1.,
                   input_shape=(rows, cols, channels),
                   output_shape=(rows, cols, channels)))
    model.add(Convolution2D(16, 8, 8, subsample=(4, 4), border_mode="same"))
    model.add(ELU())
    model.add(Convolution2D(32, 5, 5, subsample=(2, 2), border_mode="same"))
    model.add(ELU())
    model.add(Convolution2D(64, 5, 5, subsample=(2, 2), border_mode="same"))
    model.add(Flatten())
    model.add(Dropout(.2))
    model.add(ELU())
    model.add(Dense(512))
    model.add(Dropout(.5))
    model.add(ELU())
    model.add(Dense(1))

    model.compile(optimizer="adam", loss="mse")

    return model



# create the model
# model_file= "comma_ai_model"
# model = comma_ai_model()

# model_file = "nvidia_model_v1"
# model = nvidia_model_v1()

model_file = "nvidia_model_v2"
model = nvidia_model_v2()

# callbacks
from keras.callbacks import ModelCheckpoint, EarlyStopping

early_stopping = EarlyStopping(monitor = 'val_loss',
                               patience = 3,
                               verbose = 1,
                               mode = 'min')
model_checkpoint = ModelCheckpoint('best_' + model_file + '.h5',
                                   monitor = 'val_loss',
                                   verbose = 1,
                                   save_best_only = True
                                   )

# Training and validation
EPOCHS = 5
BATCH_SIZE = 128

#shuffle train and test dataset
# train_dataset = shuffle(train_dataset)
# test_dataset = shuffle(test_dataset)

# initialize generators
train_samples_generator = train_batch_generator(BATCH_SIZE)
test_samples_generator = test_batch_generator(BATCH_SIZE)

# train model
history = model.fit_generator(
    train_samples_generator,
    nb_epoch = EPOCHS,
    samples_per_epoch = BATCH_SIZE * (BATCH_SIZE + 16),
    validation_data = test_samples_generator,
    nb_val_samples = len(test_dataset.index),
    callbacks = [model_checkpoint, early_stopping]
)

# save model
print("Saving model weights and configuration file...")
model.save(model_file+'.h5', True)
with open(model_file+'.json', 'w') as outfile:
	json.dump(model.to_json(), outfile)

lastest_model_result = model.evaluate_generator(test_samples_generator, len(test_dataset.index))
print ('Latest Model Results: ', lastest_model_result)

model.load_weights('best_' + model_file+'.h5')
best_model_result = model.evaluate_generator(test_samples_generator, len(test_dataset.index))
print ('Best Model Results: ',best_model_result)

# summarize history for accuracy
import matplotlib.pyplot as plt

# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()