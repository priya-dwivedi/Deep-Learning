

from __future__ import print_function
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
#from hyperas import optim
#from optim import *
#from distributions import choice, uniform, conditional
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.optimizers import SGD, Adam, RMSprop
from keras.utils import np_utils
from keras.regularizers import l2, activity_l2


from six.moves import cPickle as pickle
import numpy as np
import os
import sys
import scipy as sp
import matplotlib.pyplot as plt


# Add path where the train images are unzipped and stored
paths = "/home/animesh/Documents/Kaggle/fisheries"
os.chdir(paths)
print(paths)

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

#Load test dataset
pickle_file = 'kaggle_test_gray.pickle'

with open(pickle_file, 'rb') as f:
    save = pickle.load(f)
    dataset_test = save['dataset_test']
    img_nm = save['img_nm']
    del save  # hint to help gc free up memory


## Reshuffle train set
## Concatenate into one master dataset
dataset_train = np.concatenate((X_train, X_valid))
target_train  = np.concatenate((y_train, y_valid))

##Image preprocessing
datagen = ImageDataGenerator(
    featurewise_center=False,  # set input mean to 0 over the dataset
    samplewise_center=True,  # set each sample mean to 0
    featurewise_std_normalization=False,  # divide inputs by std of the dataset
    samplewise_std_normalization=True,  # divide each input by its std
    zca_whitening=False,  # apply ZCA whitening
    rotation_range=0,  # randomly rotate images in the range (degrees, 0 to 180)
    width_shift_range=0,  # randomly shift images horizontally (fraction of total width)
    height_shift_range=0,  # randomly shift images vertically (fraction of total height)
    horizontal_flip=False,  # randomly flip images
    vertical_flip=False)  # randomly flip images

datagen.fit(dataset_train)
datagen.fit(dataset_test)

# split the full dataset into 80% test and 20% validation
from sklearn import cross_validation
X_train, X_valid, y_train, y_valid = cross_validation.train_test_split(
dataset_train, target_train, test_size=0.25, random_state=2865)

print("train dataset", X_train.shape, y_train.shape)
print("Validation dataset", X_valid.shape, y_valid.shape)

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
X_test = reshape(dataset_test)

# convert class vectors to binary class matrices
Y_train = np_utils.to_categorical(y_train, nb_classes)
Y_valid = np_utils.to_categorical(y_valid, nb_classes)

print('Train set', X_train.shape, Y_train.shape)
print('Validation set', X_valid.shape, Y_valid.shape)

space = {

            'depth1': hp.choice('depth1',[8,16,32]),
            'depth2': hp.choice('depth2',[16,32,64]),

            'dense1': hp.choice('dense1',[56,128,256]),
            'dense2': hp.choice('dense2',[28,56,128]),

            'dropout1': hp.uniform('dropout1', .25,.75),
            'dropout2': hp.uniform('dropout2',  .25,.75),

            'nb_epochs' :  hp.choice('nb_epochs',[30, 50,70,90]),
            'batch_size' :  hp.choice('batch_size',[32,64]),
            'optimizer': hp.choice('optimizer',['sgd','adam','rmsprop']),
        }

def f_nn(params):

    print ('Params testing: ', params)
    model = Sequential()
    model.add(ZeroPadding2D((1, 1), input_shape=X_train.shape[1:]))
    # Step 1: Convolution Layer with patch size = 3, stride = 1, same padding an depth = 16
    # Activation function - RELU. Added L2 regularization with weight decay of 0.01
    model.add(Convolution2D(params['depth1'], 3, 3, W_regularizer=l2(0.001), init='he_normal'))
    model.add(Activation('relu'))

    # Step 2: Convolution Layer with patch size = 3, stride = 1, same padding an depth = 16
    # Activation function - RELU. Added L2 regularization with weight decay of 0.01
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(params['depth1'], 3, 3, W_regularizer=l2(0.001), init='he_normal'))
    model.add(Activation('relu'))

    # Step3 : First Maxpool with kernel size = 2 and stride = 2
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

    # Step 4: Convolution Layer with patch size = 3, stride = 1, same padding an depth = 32
    # Activation function - RELU
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(params['depth2'], 3, 3, W_regularizer=l2(0.001), init='he_normal'))
    model.add(Activation('relu'))

    # Step 5: Convolution Layer with patch size = 3, stride = 1, same padding an depth = 32
    # Activation function - RELU
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(params['depth2'], 3, 3, W_regularizer=l2(0.001), init='he_normal'))
    model.add(Activation('relu'))

    # Step6 : Second Maxpool with kernel size = 2 and stride = 2
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    # model.add(Dropout(0.25))

    # Step 7: Dense layer with 256 neurons, RELU activation and L2 regulization with weight decay = 0.01
    model.add(Flatten())
    model.add(Dense(params['dense1'], W_regularizer=l2(0.001), init='he_normal'))
    model.add(Activation('relu'))
    model.add(Dropout(params['dropout1']))

    # Step 8: Dense layer with 128 neurons, RELU activation and L2 regulization with weight decay = 0.01
    model.add(Dense(params['dense2'], W_regularizer=l2(0.01), init='he_normal'))
    model.add(Activation('relu'))
    model.add(Dropout(params['dropout2']))

    # Step 9: Output Layer for number of classes and Softmax activation
    model.add(Dense(nb_classes))
    model.add(Activation('softmax'))

    model.compile(loss='categorical_crossentropy', optimizer=params['optimizer'])

    model.fit(X_train, Y_train, nb_epoch=params['nb_epochs'], batch_size= params['batch_size'],show_accuracy=True,
              verbos======e=2,
              validation_data=(X_valid, Y_valid))

    res = model.evaluate(X_valid,Y_valid, batch_size=32)
    print('Result:', res)
    sys.stdout.flush()
    return {'loss': res, 'status': STATUS_OK}


trials = Trials()
best = fmin(f_nn, space, algo=tpe.suggest, max_evals=20, trials=trials)
print('best: ')
print(best)


