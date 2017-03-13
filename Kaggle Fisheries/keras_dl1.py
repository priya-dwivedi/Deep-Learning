

from __future__ import print_function
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
import scipy as sp
import matplotlib.pyplot as plt

# Add path where the train images are unzipped and stored
paths = "/home/animesh/Documents/Kaggle/Fisheries"
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
pickle_file = 'kaggle_test_gray1.pickle'

with open(pickle_file, 'rb') as f:
    save = pickle.load(f)
    dataset_test = save['dataset_test']
    img_nm = save['img_nm']
    del save  # hint to help gc free up memory


## Reshuffle train set
## Concatenate into one master dataset
dataset_train = np.concatenate((X_train, X_valid))
target_train  = np.concatenate((y_train, y_valid))

img_aug = True

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

# split the full dataset into 80% test and 20% validation
from sklearn import cross_validation
X_train, X_valid, y_train, y_valid = cross_validation.train_test_split(
dataset_train, target_train, test_size=0.20, random_state=2278)

print("train dataset", X_train.shape, y_train.shape)
print("Validation dataset", X_valid.shape, y_valid.shape)


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
X_train = reshape(X_train)
X_valid = reshape(X_valid)
X_test = reshape(dataset_test)

# convert class vectors to binary class matrices
Y_train = np_utils.to_categorical(y_train, nb_classes)
Y_valid = np_utils.to_categorical(y_valid, nb_classes)

print('Train set', X_train.shape, Y_train.shape)
print('Validation set', X_valid.shape, Y_valid.shape)

nb_epoch = 90
## Start programming layers
model = Sequential()
model.add(ZeroPadding2D((1, 1), input_shape=X_train.shape[1:]))
# Step 1: Convolution Layer with patch size = 3, stride = 1, same padding an depth = 16
# Activation function - RELU. Added L2 regularization with weight decay of 0.01
model.add(Convolution2D(8, 3, 3, W_regularizer=l2(0.001), init = 'he_normal'))
model.add(Activation('relu'))

# Step 2: Convolution Layer with patch size = 3, stride = 1, same padding an depth = 16
# Activation function - RELU. Added L2 regularization with weight decay of 0.01
model.add(ZeroPadding2D((1, 1)))
model.add(Convolution2D(8, 3, 3, W_regularizer=l2(0.001),init = 'he_normal'))
model.add(Activation('relu'))

# Step3 : First Maxpool with kernel size = 2 and stride = 2
model.add(MaxPooling2D(pool_size=(2, 2),strides=(2, 2)))
model.add(Dropout(0.25))


# Step 4: Convolution Layer with patch size = 3, stride = 1, same padding an depth = 32
# Activation function - RELU
model.add(ZeroPadding2D((1, 1)))
model.add(Convolution2D(32, 3, 3, W_regularizer=l2(0.001),init = 'he_normal'))
model.add(Activation('relu'))
# Step 5: Convolution Layer with patch size = 3, stride = 1, same padding an depth = 32
# Activation function - RELU
model.add(ZeroPadding2D((1, 1)))
model.add(Convolution2D(32, 3, 3, W_regularizer=l2(0.001),init = 'he_normal'))
model.add(Activation('relu'))
# Step6 : Second Maxpool with kernel size = 2 and stride = 2
model.add(MaxPooling2D(pool_size=(2, 2),strides=(2, 2)))
model.add(Dropout(0.25))

### Adding one more convolution layer
# Step 4: Convolution Layer with patch size = 3, stride = 1, same padding an depth = 32
# Activation function - RELU
model.add(ZeroPadding2D((1, 1)))
model.add(Convolution2D(32, 3, 3, W_regularizer=l2(0.001),init = 'he_normal'))
model.add(Activation('relu'))
# Step 5: Convolution Layer with patch size = 3, stride = 1, same padding an depth = 32
# Activation function - RELU
model.add(ZeroPadding2D((1, 1)))
model.add(Convolution2D(32, 3, 3, W_regularizer=l2(0.001),init = 'he_normal'))
model.add(Activation('relu'))
# Step6 : Second Maxpool with kernel size = 2 and stride = 2
model.add(MaxPooling2D(pool_size=(2, 2),strides=(2, 2)))
model.add(Dropout(0.25))

#Step 7: Dense layer with 256 neurons, RELU activation and L2 regulization with weight decay = 0.01
model.add(Flatten())
model.add(Dense(256, W_regularizer=l2(0.001), init = 'he_normal'))
model.add(Activation('relu'))
model.add(Dropout(0.5273))

#Step 8: Dense layer with 128 neurons, RELU activation and L2 regulization with weight decay = 0.01
model.add(Dense(56, W_regularizer=l2(0.001), init = 'he_normal'))
model.add(Activation('relu'))
model.add(Dropout(0.4057))

#Step 9: Output Layer for number of classes and Softmax activation
model.add(Dense(nb_classes))
model.add(Activation('softmax'))

# let's train the model using SGD + momentum (how original).
sgd = SGD(lr=0.001, decay=1e-6, momentum=0.8, nesterov=True)
adam = Adam(lr=0.0004, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
rmsprop = RMSprop(lr=0.0004, rho=0.9, epsilon=1e-08, decay=0.0)
model.compile(loss='categorical_crossentropy',
              optimizer=rmsprop,
              metrics=['accuracy'])

##Image preprocessing
datagen_im = ImageDataGenerator(
    featurewise_center=False,  # set input mean to 0 over the dataset
    samplewise_center=False,  # set each sample mean to 0
    featurewise_std_normalization=False,  # divide inputs by std of the dataset
    samplewise_std_normalization=False,  # divide each input by its std
    zca_whitening=False,  # apply ZCA whitening
    rotation_range=12,  # randomly rotate images in the range (degrees, 0 to 180)
    width_shift_range=0.15,  # randomly shift images horizontally (fraction of total width)
    height_shift_range=0.15,  # randomly shift images vertically (fraction of total height)
    shear_range=0.0,
    zoom_range=0.2, # Zooms images
    horizontal_flip=False, #randomly flip images
    fill_mode='nearest',  # Fill any gaps in pixels
    vertical_flip=False)  # randomly flip images

if not img_aug:
    print('Not using data augmentation.')
    history = model.fit(X_train, Y_train,
                        batch_size=batch_size,
                        nb_epoch=nb_epoch,
                        validation_data=(X_valid, Y_valid),
                        shuffle=True,
                        verbose=2)
else:
    print('Using real-time data augmentation.')
    datagen_im.fit(X_train)

    # fits the model on batches with real-time data augmentation:
    history = model.fit_generator(datagen_im.flow(X_train, Y_train, batch_size=32),
                        samples_per_epoch=len(X_train), nb_epoch=nb_epoch,
                        validation_data=(X_valid, Y_valid),
                        verbose=2)


# fit the model on the batches generated by datagen.flow()
# history = model.fit(X_train, Y_train,
#                     batch_size=batch_size,
#                     nb_epoch=nb_epoch,
#                     validation_data=(X_valid, Y_valid),
#                     shuffle= True,
#                     verbose=2)

# To comment - Ctrl + / ; To uncomment: ctrl +  /
# # summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

res = model.evaluate(X_valid, Y_valid)
print(res)

Y_valid_pred = model.predict(X_valid)
print(Y_valid_pred[:5])
print(Y_valid_pred.shape)

print(Y_valid[:5])

from sklearn.metrics import log_loss
print(log_loss(Y_valid, Y_valid_pred))

from keras.models import model_from_json

# serialize model to JSON
model_json = model.to_json()
with open("model.json2", "w") as json_file:
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


# Test set with Adam: 1.33785
#Epoch 90/90
#2s - loss: 1.0940 - acc: 0.6925 - val_loss: 0.5889 - val_acc: 0.8003


# Test set with Adam: 1.34791
#Epoch 90/90
#2s - loss: 1.2457 - acc: 0.6001 - val_loss: 0.9036 - val_acc: 0.7116
