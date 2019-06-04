import sys

from keras import applications
from keras.models import Model, load_model
from keras.layers import Input, InputLayer, Conv2D, Activation, LeakyReLU, Concatenate
from layers import BilinearUpSampling2D
from loss import depth_loss_function


def create_model_resnet(existing='', is_halffeatures=True):
    if len(existing) == 0:
        print('Loading base model (ResNet50)..')

        # Encoder Layers
        base_model = applications.resnet50.ResNet50(weights='imagenet', include_top=False,
                                                    input_shape=(None, None, 3))

        print('Base model loaded.')

        # Starting point for decoder
        base_model_output_shape = base_model.layers[-1].output.shape #15, 20, 2048

        # Layer freezing?
        # mid_start = base_model.get_layer('activation_22')
        # for i in range(base_model.layers.index(mid_start)):
        #     base_model.layers[i].trainable = False
        for layer in base_model.layers: layer.trainable = True

        # Starting number of decoder filters
        if is_halffeatures:
            decode_filters = int(int(base_model_output_shape[-1]) / 2)
        else:
            decode_filters = int(base_model_output_shape[-1])

        # Define upsampling layer
        def upproject(tensor, filters, name, concat_with):
            up_i = BilinearUpSampling2D((2, 2), name=name + '_upsampling2d')(tensor)
            up_i = Concatenate(name=name + '_concat')(
                [up_i, base_model.get_layer(concat_with).output])  # Skip connection
            up_i = Conv2D(filters=filters, kernel_size=3, strides=1, padding='same', name=name + '_convA')(up_i)
            up_i = LeakyReLU(alpha=0.2)(up_i)
            up_i = Conv2D(filters=filters, kernel_size=3, strides=1, padding='same', name=name + '_convB')(up_i)
            up_i = LeakyReLU(alpha=0.2)(up_i)
            return up_i

        # Decoder Layers
        decoder = Conv2D(filters=decode_filters, kernel_size=1, padding='same', input_shape=base_model_output_shape,
                         name='conv2')(base_model.output)

        decoder = upproject(decoder, int(decode_filters / 2), 'up1', concat_with='res4a_branch2a') #30, 40, 256
        decoder = upproject(decoder, int(decode_filters / 4), 'up2', concat_with='res3a_branch2a') # 60, 80, 128
        decoder = upproject(decoder, int(decode_filters / 8), 'up3', concat_with='max_pooling2d_1') #120,  160, 64
        decoder = upproject(decoder, int(decode_filters / 16), 'up4', concat_with='activation_1')  #240, 320, 64
        if False: decoder = upproject(decoder, int(decode_filters / 32), 'up5', concat_with='input_1')

        # Extract depths (final layer)
        conv3 = Conv2D(filters=1, kernel_size=3, strides=1, padding='same', name='conv3')(decoder)

        # Create the model
        model = Model(inputs=base_model.input, outputs=conv3)
    else:
        # Load model from file
        if not existing.endswith('.h5'):
            sys.exit('Please provide a correct model file when using [existing] argument.')
        custom_objects = {'BilinearUpSampling2D': BilinearUpSampling2D, 'depth_loss_function': depth_loss_function}
        model = load_model(existing, custom_objects=custom_objects)
        print('\nExisting model loaded.\n')

    print('Model created.')

    return model