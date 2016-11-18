from __future__ import print_function
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.optimizers import SGD
from keras.utils import np_utils
from keras.regularizers import l2, activity_l2

from six.moves import cPickle as pickle
import numpy as np
import os
import scipy as sp
import matplotlib.pyplot as plt

# Add path where the train images are unzipped and stored
paths = "/home/animesh/Documents/Kaggle/Fisheries"
os.chdir(paths)

# Load pickled dataset - For future runs. This step can be skipped
pickle_file = 'all_fish_gray.pickle'

with open(pickle_file, 'rb') as f:
    save = pickle.load(f)
    X_valid = save['X_valid']
    y_valid = save['y_valid']
    X_train = save['X_train']
    y_train = save['y_train']
    del save  # hint to help gc free up memory
    print('Train set', X_train.shape, y_train.shape)
    print('Validation set', X_valid.shape, y_valid.shape)

# Define some initial parameters
batch_size = 32
nb_classes = 8
nb_epoch = 50
data_augmentation = True # Set to False initially

# input image dimensions
img_rows, img_cols = 128, 128
img_channels = 1

# Resize dataset into a 4D array as the proper input format into tensor
image_size = 128
num_channels = 1 # grayscale
def reshape(dataset):
  dataset = dataset.reshape(
    (-1, image_size, image_size, num_channels)).astype(np.float32)
  return dataset
X_train = reshape(X_train)
X_valid = reshape(X_valid)

# convert class vectors to binary class matrices
Y_train = np_utils.to_categorical(y_train, nb_classes)
Y_valid = np_utils.to_categorical(y_valid, nb_classes)

print('Train set', X_train.shape, Y_train.shape)
print('Validation set', X_valid.shape, Y_valid.shape)


## Start programming layers
model = Sequential()
# Step 1: Convolution Layer with patch size = 3, stride = 1, same padding an depth = 16
# Activation function - RELU. Added L2 regularization with weight decay of 0.01
model.add(Convolution2D(16, 3, 3, border_mode='same',
                        input_shape=X_train.shape[1:], W_regularizer=l2(0.01)))
model.add(Activation('relu'))

# Step 2: Convolution Layer with patch size = 3, stride = 1, same padding an depth = 16
# Activation function - RELU. Added L2 regularization with weight decay of 0.01
model.add(Convolution2D(16, 3, 3, W_regularizer=l2(0.01)))
model.add(Activation('relu'))

# Step3 : First Maxpool with kernel size = 2 and stride = 2
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

# Step 4: Convolution Layer with patch size = 3, stride = 1, same padding an depth = 32
# Activation function - RELU
model.add(Convolution2D(32, 3, 3, border_mode='same'))
model.add(Activation('relu'))
# Step 5: Convolution Layer with patch size = 3, stride = 1, same padding an depth = 32
# Activation function - RELU
model.add(Convolution2D(32, 3, 3))
model.add(Activation('relu'))
# Step6 : Second Maxpool with kernel size = 2 and stride = 2
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

#Step 7: Dense layer with 128 neurons, RELU activation and L2 regulization with weight decay = 0.01
model.add(Flatten())
model.add(Dense(128, W_regularizer=l2(0.01)))
model.add(Activation('relu'))
model.add(Dropout(0.5))

#Step 12: Output Layer for number of classes and Softmax activation
model.add(Dense(nb_classes))
model.add(Activation('softmax'))

# let's train the model using SGD + momentum (how original).
sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy',
              optimizer=sgd,
              metrics=['accuracy'])


if not data_augmentation:
    print('Not using data augmentation.')
    model.fit(X_train, Y_train,
              batch_size=batch_size,
              nb_epoch=nb_epoch,
              validation_data=(X_valid, Y_valid),
              shuffle=True)
else:
    print('Using real-time data augmentation.')

    # this will do preprocessing and realtime data augmentation
    # For now image augmentation has been turned off. Only doing preprocessing
    datagen = ImageDataGenerator(
        featurewise_center=True,  # set input mean to 0 over the dataset
        samplewise_center=False,  # set each sample mean to 0
        featurewise_std_normalization=True,  # divide inputs by std of the dataset
        samplewise_std_normalization=False,  # divide each input by its std
        zca_whitening=False,  # apply ZCA whitening
        rotation_range=0,  # randomly rotate images in the range (degrees, 0 to 180)
        width_shift_range=0,  # randomly shift images horizontally (fraction of total width)
        height_shift_range=0,  # randomly shift images vertically (fraction of total height)
        horizontal_flip=False,  # randomly flip images
        vertical_flip=False)  # randomly flip images

    # compute quantities required for featurewise normalization
    # (std, mean, and principal components if ZCA whitening is applied)
    datagen.fit(X_train)

    # fit the model on the batches generated by datagen.flow()
    model.fit_generator(datagen.flow(X_train, Y_train,
                        batch_size=batch_size),
                        samples_per_epoch=X_train.shape[0],
                        nb_epoch=nb_epoch,
                        validation_data=(X_valid, Y_valid),
                        verbose=2)