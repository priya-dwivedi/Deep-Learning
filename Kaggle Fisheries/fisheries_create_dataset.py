## Kaggle Project



from __future__ import print_function
from keras.datasets import cifar10
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.optimizers import SGD
from keras.utils import np_utils


from six.moves import cPickle as pickle
import glob
from scipy import ndimage
import numpy as np
import os
from PIL import Image
import scipy as sp
import matplotlib.pyplot as plt

# TODO: Add path where the train images are unzipped and stored
paths = "/home/animesh/Documents/Kaggle/Fisheries"
os.chdir(paths)

alb = os.path.join(paths,'ALB')
print(alb)
img_alb = os.path.join(alb,'*.jpg')
width = []
height = []
for infile in glob.glob(img_alb):
    #file, ext = os.path.splitext(infile)
    im = Image.open(infile)
    width.append(im.size[0])
    height.append(im.size[1])

print(np.average(width), np.average(height))
# 1272, 730
plt.scatter(width,height)
plt.show()

## Lets reduce to 128 x 128 images
img_alb = os.path.join(alb,'*.jpg')
size = 128, 128
for infile in glob.glob(img_alb):
    outfile = os.path.splitext(infile)[0] + ".small"
    file, ext = os.path.splitext(infile)
    im = Image.open(infile).convert('L')
    out = im.resize((size))
    out.save(outfile, "JPEG")


# Display reduced images
os.chdir(alb)
im = Image.open("img_00015.small")
print(im.format, im.size, im.mode)
# Works

##Flatten to dataset - grayscale
alb_small = os.path.join(alb, "*.small")
image_size = 128
pixel_depth = 255
image_files = 1719
num_channels = 1
dataset_alb = np.ndarray(shape=(image_files, image_size, image_size), dtype=np.float32)
target_alb = np.ndarray(shape=(image_files), dtype=np.int_) # set to 0
num_images = 0
for filename in glob.glob(alb_small):

    if num_images % 500 == 0: print(num_images)
    try:
        image_data = (ndimage.imread(filename, flatten=True).astype(float)) / pixel_depth
        if image_data.shape != (image_size, image_size):
            raise Exception('Unexpected image shape: %s' % str(image_data.shape))
        dataset_alb[num_images, :, :] = image_data
        name = os.path.basename(filename)
        target_alb[num_images] = 0
        num_images = num_images + 1
    except IOError as e:
        print('Could not read:', filename, ':', e, '- it\'s ok, skipping.')


print('Dataset shape:', dataset_alb.shape)
print('Target shape:', target_alb.shape)
print('Dataset Mean:', np.mean(dataset_alb))
print('Target Mean:', np.mean(target_alb))


# Repeat for BET

newloc = os.path.join(paths,'BET')  #Update
image_files = 200 #Update
dataset_bet = np.ndarray(shape=(image_files, image_size, image_size), dtype=np.float32) #Update
target_bet = np.ndarray(shape=(image_files), dtype=np.int_) # set to 1 - Update
newpath1 = os.path.join(newloc,'*.jpg')
for infile in glob.glob(newpath1):
    outfile = os.path.splitext(infile)[0] + ".small"
    file, ext = os.path.splitext(infile)
    im = Image.open(infile).convert('L')
    out = im.resize((size))
    out.save(outfile, "JPEG")

newpath2 = os.path.join(newloc, "*.small")
image_size = 128
pixel_depth = 255
num_images = 0
for filename in glob.glob(newpath2):

    if num_images % 500 == 0: print(num_images)
    try:
        image_data = (ndimage.imread(filename, flatten=True).astype(float)) / pixel_depth
        if image_data.shape != (image_size, image_size):
            raise Exception('Unexpected image shape: %s' % str(image_data.shape))
        dataset_bet[num_images, :, :] = image_data #Update
        name = os.path.basename(filename)
        target_bet[num_images] = 1 #Update
        num_images = num_images + 1
    except IOError as e:
        print('Could not read:', filename, ':', e, '- it\'s ok, skipping.')

print('Dataset shape:', dataset_bet.shape)
print('Target shape:', target_bet.shape)
print('Dataset Mean:', np.mean(dataset_bet))
print('Target Mean:', np.mean(target_bet))


# Repeat for DOL

newloc = os.path.join(paths,'DOL')  #Update
image_files = 117 #Update
dataset_dol = np.ndarray(shape=(image_files, image_size, image_size), dtype=np.float32) #Update
target_dol = np.ndarray(shape=(image_files), dtype=np.int_) # set to 2 - Update
newpath1 = os.path.join(newloc,'*.jpg')
for infile in glob.glob(newpath1):
    outfile = os.path.splitext(infile)[0] + ".small"
    file, ext = os.path.splitext(infile)
    im = Image.open(infile).convert('L')
    out = im.resize((size))
    out.save(outfile, "JPEG")

newpath2 = os.path.join(newloc, "*.small")
image_size = 128
pixel_depth = 255
num_images = 0
for filename in glob.glob(newpath2):

    if num_images % 500 == 0: print(num_images)
    try:
        image_data = (ndimage.imread(filename, flatten=True).astype(float)) / pixel_depth
        if image_data.shape != (image_size, image_size):
            raise Exception('Unexpected image shape: %s' % str(image_data.shape))
        dataset_dol[num_images, :, :] = image_data #Update
        name = os.path.basename(filename)
        target_dol[num_images] = 2 #Update
        num_images = num_images + 1
    except IOError as e:
        print('Could not read:', filename, ':', e, '- it\'s ok, skipping.')

print('Dataset shape:', dataset_dol.shape)
print('Target shape:', target_dol.shape)
print('Dataset Mean:', np.mean(dataset_dol))
print('Target Mean:', np.mean(target_dol))

# Repeat for LAG

newloc = os.path.join(paths,'LAG')  #Update
image_files = 67 #Update
dataset_lag = np.ndarray(shape=(image_files, image_size, image_size), dtype=np.float32) #Update
target_lag = np.ndarray(shape=(image_files), dtype=np.int_) # set to 3 - Update
newpath1 = os.path.join(newloc,'*.jpg')
for infile in glob.glob(newpath1):
    outfile = os.path.splitext(infile)[0] + ".small"
    file, ext = os.path.splitext(infile)
    im = Image.open(infile).convert('L')
    out = im.resize((size))
    out.save(outfile, "JPEG")

newpath2 = os.path.join(newloc, "*.small")
image_size = 128
pixel_depth = 255
num_images = 0
for filename in glob.glob(newpath2):

    if num_images % 500 == 0: print(num_images)
    try:
        image_data = (ndimage.imread(filename, flatten=True).astype(float)) / pixel_depth
        if image_data.shape != (image_size, image_size):
            raise Exception('Unexpected image shape: %s' % str(image_data.shape))
        dataset_lag[num_images, :, :] = image_data #Update
        name = os.path.basename(filename)
        target_lag[num_images] = 3 #Update
        num_images = num_images + 1
    except IOError as e:
        print('Could not read:', filename, ':', e, '- it\'s ok, skipping.')

print('Dataset shape:', dataset_lag.shape)
print('Target shape:', target_lag.shape)
print('Dataset Mean:', np.mean(dataset_lag))
print('Target Mean:', np.mean(target_lag))


# Repeat for NoF

newloc = os.path.join(paths,'NoF')  #Update
image_files = 465 #Update
dataset_nof = np.ndarray(shape=(image_files, image_size, image_size), dtype=np.float32) #Update
target_nof = np.ndarray(shape=(image_files), dtype=np.int_) # set to 4 - Update
newpath1 = os.path.join(newloc,'*.jpg')
for infile in glob.glob(newpath1):
    outfile = os.path.splitext(infile)[0] + ".small"
    file, ext = os.path.splitext(infile)
    im = Image.open(infile).convert('L')
    out = im.resize((size))
    out.save(outfile, "JPEG")

newpath2 = os.path.join(newloc, "*.small")
image_size = 128
pixel_depth = 255
num_images = 0
for filename in glob.glob(newpath2):

    if num_images % 500 == 0: print(num_images)
    try:
        image_data = (ndimage.imread(filename, flatten=True).astype(float)) / pixel_depth
        if image_data.shape != (image_size, image_size):
            raise Exception('Unexpected image shape: %s' % str(image_data.shape))
        dataset_nof[num_images, :, :] = image_data #Update
        name = os.path.basename(filename)
        target_nof[num_images] = 4 #Update
        num_images = num_images + 1
    except IOError as e:
        print('Could not read:', filename, ':', e, '- it\'s ok, skipping.')

print('Dataset shape:', dataset_nof.shape)
print('Target shape:', target_nof.shape)
print('Dataset Mean:', np.mean(dataset_nof))
print('Target Mean:', np.mean(target_nof))


# Repeat for Other

newloc = os.path.join(paths,'OTHER')  #Update
image_files = 299 #Update
dataset_other = np.ndarray(shape=(image_files, image_size, image_size), dtype=np.float32) #Update
target_other = np.ndarray(shape=(image_files), dtype=np.int_) # set to 5 - Update
newpath1 = os.path.join(newloc,'*.jpg')
for infile in glob.glob(newpath1):
    outfile = os.path.splitext(infile)[0] + ".small"
    file, ext = os.path.splitext(infile)
    im = Image.open(infile).convert('L')
    out = im.resize((size))
    out.save(outfile, "JPEG")

newpath2 = os.path.join(newloc, "*.small")
image_size = 128
pixel_depth = 255
num_images = 0
for filename in glob.glob(newpath2):

    if num_images % 500 == 0: print(num_images)
    try:
        image_data = (ndimage.imread(filename, flatten=True).astype(float)) / pixel_depth
        if image_data.shape != (image_size, image_size):
            raise Exception('Unexpected image shape: %s' % str(image_data.shape))
        dataset_other[num_images, :, :] = image_data #Update
        name = os.path.basename(filename)
        target_other[num_images] = 5 #Update
        num_images = num_images + 1
    except IOError as e:
        print('Could not read:', filename, ':', e, '- it\'s ok, skipping.')

print('Dataset shape:', dataset_other.shape)
print('Target shape:', target_other.shape)
print('Dataset Mean:', np.mean(dataset_other))
print('Target Mean:', np.mean(target_other))


# Repeat for Shark

newloc = os.path.join(paths,'SHARK')  #Update
image_files = 176 #Update
dataset_sh = np.ndarray(shape=(image_files, image_size, image_size), dtype=np.float32) #Update
target_sh = np.ndarray(shape=(image_files), dtype=np.int_) # set to 6 - Update
newpath1 = os.path.join(newloc,'*.jpg')
for infile in glob.glob(newpath1):
    outfile = os.path.splitext(infile)[0] + ".small"
    file, ext = os.path.splitext(infile)
    im = Image.open(infile).convert('L')
    out = im.resize((size))
    out.save(outfile, "JPEG")

newpath2 = os.path.join(newloc, "*.small")
image_size = 128
pixel_depth = 255
num_images = 0
for filename in glob.glob(newpath2):

    if num_images % 500 == 0: print(num_images)
    try:
        image_data = (ndimage.imread(filename, flatten=True).astype(float)) / pixel_depth
        if image_data.shape != (image_size, image_size):
            raise Exception('Unexpected image shape: %s' % str(image_data.shape))
        dataset_sh[num_images, :, :] = image_data #Update
        name = os.path.basename(filename)
        target_sh[num_images] = 6 #Update
        num_images = num_images + 1
    except IOError as e:
        print('Could not read:', filename, ':', e, '- it\'s ok, skipping.')

print('Dataset shape:', dataset_sh.shape)
print('Target shape:', target_sh.shape)
print('Dataset Mean:', np.mean(dataset_sh))
print('Target Mean:', np.mean(target_sh))

# Repeat for YFT

newloc = os.path.join(paths,'YFT')  #Update
image_files = 734 #Update
dataset_yft = np.ndarray(shape=(image_files, image_size, image_size), dtype=np.float32) #Update
target_yft = np.ndarray(shape=(image_files), dtype=np.int_) # set to 7 - Update
newpath1 = os.path.join(newloc,'*.jpg')
for infile in glob.glob(newpath1):
    outfile = os.path.splitext(infile)[0] + ".small"
    file, ext = os.path.splitext(infile)
    im = Image.open(infile).convert('L')
    out = im.resize((size))
    out.save(outfile, "JPEG")

newpath2 = os.path.join(newloc, "*.small")
image_size = 128
pixel_depth = 255
num_images = 0
for filename in glob.glob(newpath2):

    if num_images % 500 == 0: print(num_images)
    try:
        image_data = (ndimage.imread(filename, flatten=True).astype(float)) / pixel_depth
        if image_data.shape != (image_size, image_size):
            raise Exception('Unexpected image shape: %s' % str(image_data.shape))
        dataset_yft[num_images, :, :] = image_data #Update
        name = os.path.basename(filename)
        target_yft[num_images] = 7 #Update
        num_images = num_images + 1
    except IOError as e:
        print('Could not read:', filename, ':', e, '- it\'s ok, skipping.')

print('Dataset shape:', dataset_yft.shape)
print('Target shape:', target_yft.shape)
print('Dataset Mean:', np.mean(dataset_yft))
print('Target Mean:', np.mean(target_yft))

## Concatenate into one master dataset
dataset = np.concatenate((dataset_alb,dataset_bet,dataset_dol,dataset_lag,dataset_nof,dataset_other,dataset_sh,dataset_yft))
target = np.concatenate((target_alb,target_bet,target_dol,target_lag,target_nof,target_other,target_sh,target_yft))

# Check Stats on the dataset
print('Dataset shape:', dataset.shape)
print('Target shape:', target.shape)
print('Dataset Mean:', np.mean(dataset))
print('Dataset Standard deviation:', np.std(dataset))
print('Dataset Max:', np.amax(dataset))
print('Dataset Min:', np.amin(dataset))
print('Target shape:', target.shape)
print('Target Mean:', np.mean(target))
print('Target Standard deviation:', np.std(target))
print('Target Max:', np.amax(target))
print('Target Min:', np.amin(target))

# Randomize dataset and target
def randomize(dataset, labels):
  permutation = np.random.permutation(labels.shape[0])
  shuffled_dataset = dataset[permutation,:,:]
  shuffled_labels = labels[permutation]
  return shuffled_dataset, shuffled_labels
all_dataset, all_labels = randomize(dataset, target)

print(all_dataset.shape, all_labels.shape)
print(all_labels[:10])

# split the full dataset into 80% test and 20% validation
from sklearn import cross_validation
X_train, X_valid, y_train, y_valid = cross_validation.train_test_split(
all_dataset, all_labels, test_size=0.2, random_state=2275)

print("train dataset", X_train.shape, y_train.shape)
print("Validation dataset", X_valid.shape, y_valid.shape)


# Pickle this dataset for future use if required. This step can be skipped
os.chdir(paths)
pickle_file = 'all_fish_gray.pickle'

try:
  f = open(pickle_file, 'wb')
  save = {
    'X_valid': X_valid,
    'y_valid': y_valid,
    'X_train': X_train,
    'y_train': y_train,
    }
  pickle.dump(save, f, pickle.HIGHEST_PROTOCOL)
  f.close()
except Exception as e:
  print('Unable to save data to', pickle_file, ':', e)
  raise

