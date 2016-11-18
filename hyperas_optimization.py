from __future__ import print_function
from hyperopt import Trials, STATUS_OK, tpe
from hyperas import optim
from hyperas.distributions import choice, uniform, conditional
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.optimizers import SGD, Adam
from keras.utils import np_utils
from keras.regularizers import l2, activity_l2


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
pickle_file = '/home/animesh/Documents/Kaggle/Fisheries/test_stg1/kaggle_test_gray.pickle'

with open(pickle_file, 'rb') as f:
    save = pickle.load(f)
    dataset_test = save['dataset_test']
    del save  # hint to help gc free up memory

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

def data():
    '''
    Data providing function:

    This function is separated from model() so that hyperopt
    won't reload data for each evaluation run.
    '''
    (X_train, y_train), (X_test, y_test) = mnist.load_data()
    X_train = X_train.reshape(60000, 784)
    X_test = X_test.reshape(10000, 784)
    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')
    X_train /= 255
    X_test /= 255
    nb_classes = 10
    Y_train = np_utils.to_categorical(y_train, nb_classes)
    Y_test = np_utils.to_categorical(y_test, nb_classes)
    return X_train, Y_train, X_test, Y_test


def model(X_train, Y_train, X_valid, Y_valid):
    '''
    Model providing function:

    Create Keras model with double curly brackets dropped-in as needed.
    Return value has to be a valid python dictionary with two customary keys:
        - loss: Specify a numeric evaluation metric to be minimized
        - status: Just use STATUS_OK and see hyperopt documentation if not feasible
    The last one is optional, though recommended, namely:
        - model: specify the model just created so that we can later use it again.
    '''
    model = Sequential()
    # First Conv pair
    model.add(Convolution2D({{choice([8, 16, 32])}}, 3, 3, border_mode='same',
                            input_shape=X_train.shape[1:], W_regularizer=l2(0.01), init='he_normal'))
    model.add(Activation('relu'))

    model.add(Convolution2D({{choice([8, 16, 32])}}, 3, 3, W_regularizer=l2(0.01), init='he_normal'))
    model.add(Activation('relu'))


    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout({{uniform(0, 1)}})


    model.add(Convolution2D({{choice([16, 32, 64])}}, 3, 3, border_mode='same',
                            input_shape=X_train.shape[1:], W_regularizer=l2(0.01), init='he_normal'))
    model.add(Activation('relu'))

    model.add(Convolution2D({{choice([16, 32, 64])}}, 3, 3, W_regularizer=l2(0.01), init='he_normal'))
    model.add(Activation('relu'))


    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout({{uniform(0, 1)}})

    # Dense Layer
    model.add(Flatten())
    model.add(Dense({{choice([128, 256, 512])}}, W_regularizer=l2(0.01), init='he_normal'))
    model.add(Activation('relu'))
    model.add(Dropout({{uniform(0, 1)}})

    # Step 8: Dense layer with 128 neurons, RELU activation and L2 regulization with weight decay = 0.01
    model.add(Dense({{choice([64, 128, 256])}}, W_regularizer=l2(0.01), init='he_normal'))
    model.add(Activation('relu'))
    model.add(Dropout({{uniform(0, 1)}})

    # Step 9: Output Layer for number of classes and Softmax activation
    model.add(Dense(nb_classes))
    model.add(Activation('softmax'))


    model.compile(loss='categorical_crossentropy', metrics=['accuracy'],
                  optimizer={{choice(['rmsprop', 'adam', 'sgd'])}})

    model.fit(X_train, Y_train,
              batch_size={{choice([32, 64])}},
              nb_epoch=1,
              show_accuracy=True,
              verbose=2,
              validation_data=(X_test, Y_valid))
    score, acc = model.evaluate(X_test, Y_test, verbose=0)
    print('Test accuracy:', acc)
    return {'loss': -acc, 'status': STATUS_OK, 'model': model}

if __name__ == '__main__':
    best_run, best_model = optim.minimize(model=model,
                                          data=data,
                                          algo=tpe.suggest,
                                          max_evals=5,
                                          trials=Trials())
    X_train, Y_train, X_test, Y_test = data()
    print("Evalutation of best performing model:")
    print(best_model.evaluate(X_test, Y_test))