import io
import random
import numpy as np
from PIL import Image

import keras
from keras import backend as K
from utils import DepthNorm, predict, evaluate

import tensorflow as tf

def make_image(tensor):
    height, width, channel = tensor.shape
    image = Image.fromarray(tensor.astype('uint8'))
    output = io.BytesIO()
    image.save(output, format='JPEG', quality=90)
    image_string = output.getvalue()
    output.close()
    return tf.Summary.Image(height=height, width=width, colorspace=channel, encoded_image_string=image_string)

def get_nyu_callbacks(model, basemodel, train_generator, test_generator, test_set, runPath):
    callbacks = []

    # Callback: Tensorboard
    class LRTensorBoard(keras.callbacks.TensorBoard):
        def __init__(self, log_dir):
            super().__init__(log_dir=log_dir)

            self.num_samples = 6
            self.train_idx = np.random.randint(low=0, high=len(train_generator), size=10)
            self.test_idx = np.random.randint(low=0, high=len(test_generator), size=10)

        def on_epoch_end(self, epoch, logs=None):            
            if not test_set == None:
                # Samples using current model
                import matplotlib.pyplot as plt
                from skimage.transform import resize
                plasma = plt.get_cmap('plasma')

                minDepth, maxDepth = 10, 1000

                train_samples = []
                test_samples = []

                for i in range(self.num_samples):
                    x_train, y_train = train_generator.__getitem__(self.train_idx[i], False)
                    x_test, y_test = test_generator[self.test_idx[i]]

                    x_train, y_train = x_train[0], np.clip(DepthNorm(y_train[0], maxDepth=1000), minDepth, maxDepth) / maxDepth 
                    x_test, y_test = x_test[0], np.clip(DepthNorm(y_test[0], maxDepth=1000), minDepth, maxDepth) / maxDepth

                    h, w = y_train.shape[0], y_train.shape[1]

                    rgb_train = resize(x_train, (h,w), preserve_range=True, mode='reflect', anti_aliasing=True)
                    rgb_test = resize(x_test, (h,w), preserve_range=True, mode='reflect', anti_aliasing=True)

                    gt_train = plasma(y_train[:,:,0])[:,:,:3]
                    gt_test = plasma(y_test[:,:,0])[:,:,:3]

                    predict_train = plasma(predict(model, x_train, minDepth=minDepth, maxDepth=maxDepth)[0,:,:,0])[:,:,:3]
                    predict_test = plasma(predict(model, x_test, minDepth=minDepth, maxDepth=maxDepth)[0,:,:,0])[:,:,:3]

                    train_samples.append(np.vstack([rgb_train, gt_train, predict_train]))
                    test_samples.append(np.vstack([rgb_test, gt_test, predict_test]))

                self.writer.add_summary(tf.Summary(value=[tf.Summary.Value(tag='Train', image=make_image(255 * np.hstack(train_samples)))]), epoch)
                self.writer.add_summary(tf.Summary(value=[tf.Summary.Value(tag='Test', image=make_image(255 * np.hstack(test_samples)))]), epoch)
                
                # Metrics
                e = evaluate(model, test_set['rgb'], test_set['depth'], test_set['crop'], batch_size=6, verbose=True)
                logs.update({'rel': e[3]})
                logs.update({'rms': e[4]})
                logs.update({'log10': e[5]})

            super().on_epoch_end(epoch, logs)
    callbacks.append( LRTensorBoard(log_dir=runPath) )

    # Callback: Learning Rate Scheduler
    lr_schedule = keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.7, patience=5, min_lr=0.00009, min_delta=1e-2)
    callbacks.append( lr_schedule ) # reduce learning rate when stuck

    # Callback: save checkpoints
    callbacks.append(keras.callbacks.ModelCheckpoint(runPath + '/weights.{epoch:02d}-{val_loss:.2f}.h5', monitor='val_loss',
        verbose=1, save_best_only=False, save_weights_only=False, mode='min', period=1))

    return callbacks