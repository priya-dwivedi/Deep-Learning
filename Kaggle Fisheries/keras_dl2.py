

from __future__ import print_function
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.optimizers import SGD, Adam, RMSprop
from keras.utils import np_utils
from keras.regularizers import l2, activity_l2
from sklearn.cross_validation import KFold

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

#Load test dataset
pickle_file = 'kaggle_test_gray1.pickle'

with open(pickle_file, 'rb') as f:
    save = pickle.load(f)
    dataset_test = save['dataset_test']
    img_nm = save['img_nm']
    del save  # hint to help gc free up memory

print(img_nm[:50])
## Reshuffle train set
## Concatenate into one master dataset
dataset_train = np.concatenate((X_train, X_valid))
target_train  = np.concatenate((y_train, y_valid))

print(np.unique(target_train, return_counts= True))
##Image preprocessing
##Image preprocessing
datagen = ImageDataGenerator(
    featurewise_center=True,  # set input mean to 0 over the dataset
    samplewise_center=True,  # set each sample mean to 0
    featurewise_std_normalization=True,  # divide inputs by std of the dataset
    samplewise_std_normalization=True,  # divide each input by its std
    zca_whitening=False,  # apply ZCA whitening
    rotation_range=0,  # randomly rotate images in the range (degrees, 0 to 180)
    width_shift_range=0,  # randomly shift images horizontally (fraction of total width)
    height_shift_range=0,  # randomly shift images vertically (fraction of total height)
    horizontal_flip=False,  # randomly flip images
    vertical_flip=False)  # randomly flip images

datagen.fit(dataset_train)
datagen.fit(dataset_test)

# Define some initial parameters
batch_size = 32
nb_classes = 8


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
dataset_train= reshape(dataset_train)
X_test = reshape(dataset_test)

print(target_train.shape)
# convert class vectors to binary class matrices
target_train = np_utils.to_categorical(target_train, nb_classes)

print('Data set', dataset_train.shape, target_train.shape)

img_aug = True
nb_epoch = 50
## Start programming layers
def create_model():
    model = Sequential()
    model.add(ZeroPadding2D((1, 1), input_shape=dataset_train.shape[1:]))
    # Step 1: Convolution Layer with patch size = 3, stride = 1, same padding an depth = 16
    # Activation function - RELU. Added L2 regularization with weight decay of 0.01
    model.add(Convolution2D(8, 3, 3, W_regularizer=l2(0.001), init='he_normal'))
    model.add(Activation('relu'))

    # Step 2: Convolution Layer with patch size = 3, stride = 1, same padding an depth = 16
    # Activation function - RELU. Added L2 regularization with weight decay of 0.01
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(8, 3, 3, W_regularizer=l2(0.001), init='he_normal'))
    model.add(Activation('relu'))

    # Step3 : First Maxpool with kernel size = 2 and stride = 2
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Dropout(0.25))

    # Step 4: Convolution Layer with patch size = 3, stride = 1, same padding an depth = 32
    # Activation function - RELU
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(32, 3, 3, W_regularizer=l2(0.001), init='he_normal'))
    model.add(Activation('relu'))
    # Step 5: Convolution Layer with patch size = 3, stride = 1, same padding an depth = 32
    # Activation function - RELU
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(32, 3, 3, W_regularizer=l2(0.001), init='he_normal'))
    model.add(Activation('relu'))
    # Step6 : Second Maxpool with kernel size = 2 and stride = 2
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Dropout(0.25))

    #Step 7: Dense layer with 256 neurons, RELU activation and L2 regulization with weight decay = 0.01
    model.add(Flatten())
    model.add(Dense(256, W_regularizer=l2(0.001), init='he_normal'))
    model.add(Activation('relu'))
    model.add(Dropout(0.5273))

    #Step 8: Dense layer with 56 neurons, RELU activation and L2 regulization with weight decay = 0.01
    model.add(Dense(56, W_regularizer=l2(0.001), init='he_normal'))
    model.add(Activation('relu'))
    model.add(Dropout(0.4057))

    #Step 9: Output Layer for number of classes and Softmax activation
    model.add(Dense(nb_classes))
    model.add(Activation('softmax'))

    # let's train the model using SGD + momentum (how original).
    sgd = SGD(lr=0.001, decay=1e-6, momentum=0.8, nesterov=True)
    adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
    rmsprop = RMSprop(lr=0.0005, rho=0.9, epsilon=1e-08, decay=0.0)
    model.compile(loss='categorical_crossentropy',
                  optimizer=rmsprop,
                  metrics=['accuracy'])
    return model

def train_model(model, train_data, labels_train, test_data, labels_test):
    datagen_im = ImageDataGenerator(
        featurewise_center=False,  # set input mean to 0 over the dataset
        samplewise_center=False,  # set each sample mean to 0
        featurewise_std_normalization=False,  # divide inputs by std of the dataset
        samplewise_std_normalization=False,  # divide each input by its std
        zca_whitening=False,  # apply ZCA whitening
        rotation_range=10,  # randomly rotate images in the range (degrees, 0 to 180)
        width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
        height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
        shear_range=0.0,
        zoom_range=0.2,  # Zooms images
        horizontal_flip=False,  # randomly flip images
        fill_mode='nearest',  # Fill any gaps in pixels
        vertical_flip=False)  # randomly flip images

    if not img_aug:
        print('Not using data augmentation.')
        history = model.fit(train_data, labels_train,
                            batch_size=batch_size,
                            nb_epoch=nb_epoch,
                            validation_data=(test_data, labels_test),
                            shuffle=True,
                            verbose=2)
    else:
        print('Using real-time data augmentation.')
        datagen_im.fit(X_train)

        # fits the model on batches with real-time data augmentation:
        model.fit_generator(datagen_im.flow(train_data, labels_train, batch_size=32),
                                      samples_per_epoch=len(train_data), nb_epoch=nb_epoch,
                                      validation_data=(test_data, labels_test),
                                      verbose=2)


#if __name__ == "__main__":
n_folds = 5
i=0
kf = KFold(len(target_train), n_folds=n_folds, shuffle=True)

for train_index, test_index in kf:
            print("Running Fold", i+1, "/", n_folds)
            ## Create datasets
            train_data = dataset_train[train_index]
            labels_train = target_train[train_index]
            test_data = dataset_train[test_index]
            labels_test = target_train[test_index]
            model = None # Clearing the NN.
            model = create_model()
            train_model(model, train_data, labels_train, test_data, labels_test)
            i += 1

# To comment - Ctrl + / ; To uncomment: ctrl +  /

paths = "/home/animesh/Documents/Kaggle/Fisheries"
os.chdir(paths)
from keras.models import model_from_json

# serialize model to JSON
model_json = model.to_json()
with open("model.json1", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5

model.save_weights("dl1_sgd.h5")
print("Saved model to disk")


test_pred = model.predict(X_test, batch_size=32)
print(test_pred[:5])
print(len(test_pred))


# Export results to CSV file
import csv
with open('test2.csv', 'wb') as f:
    writer = csv.writer(f, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL, lineterminator='\n')
    i=0
    for val in test_pred:
        # Write item to outcsv
        writer.writerow([img_nm[i], val[0], val[1], val[2], val[3], val[4], val[5], val[6], val[7]])
        i+= 1


#LB score: 1.40708
