
from six.moves import cPickle as pickle
import glob
from scipy import ndimage
import numpy as np
import os
from PIL import Image
import scipy as sp
import matplotlib.pyplot as plt
import numpy as np

test_path = "/home/animesh/Documents/Kaggle/Fisheries/test_stg1"
os.chdir(test_path)

image_files = 1000 #Update
image_size = 128
pixel_depth = 255
size = 128, 128
dataset_test = np.ndarray(shape=(image_files, image_size, image_size), dtype=np.float32)#Update
img_nm = []
newpath1 = os.path.join(test_path,'*.jpg')
for infile in sorted(glob.glob(newpath1)):
    outfile = os.path.splitext(infile)[0] + ".small"
    file, ext = os.path.splitext(infile)
    im = Image.open(infile).convert('L')
    out = im.resize((size))
    out.save(outfile, "JPEG")

newpath2 = os.path.join(test_path, "*.small")
image_size = 128
pixel_depth = 255
num_images = 0
for filename in sorted(glob.glob(newpath2)):

    if num_images % 500 == 0: print(num_images)
    try:
        image_data = (ndimage.imread(filename, flatten=True).astype(float)) / pixel_depth
        if image_data.shape != (image_size, image_size):
            raise Exception('Unexpected image shape: %s' % str(image_data.shape))
        dataset_test[num_images, :, :] = image_data #Update
        name = os.path.basename(filename)
        img_nm.append(name.split(".")[0])
        num_images = num_images + 1
    except IOError as e:
        print('Could not read:', filename, ':', e, '- it\'s ok, skipping.')

print('Dataset shape:', dataset_test.shape)
print('Dataset Mean:', np.mean(dataset_test))
print('Dataset Standard deviation:', np.std(dataset_test))
print('Dataset Max:', np.amax(dataset_test))
print('Dataset Min:', np.amin(dataset_test))

# Pickle this dataset for future use
pickle_file = 'kaggle_test_gray1.pickle'

try:
  f = open(pickle_file, 'wb')
  save = {
    'dataset_test': dataset_test,
    'img_nm': img_nm
    }
  pickle.dump(save, f, pickle.HIGHEST_PROTOCOL)
  f.close()
except Exception as e:
  print('Unable to save data to', pickle_file, ':', e)
  raise

print(len(img_nm))
print(img_nm[:50])
