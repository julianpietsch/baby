import tensorflow as tf
from tensorflow.python.keras.losses import binary_crossentropy
from tensorflow.python.keras.layers import (
    Conv2D, BatchNormalization, Activation, MaxPooling2D,
    Conv2DTranspose, concatenate
)


### LOSS FUNCTIONS ###


def dice_coeff(y_true, y_pred):
    smooth = 1.
    # Flatten
    y_true_f = tf.reshape(y_true, [-1])
    y_pred_f = tf.reshape(y_pred, [-1])
    intersection = tf.reduce_sum(y_true_f * y_pred_f)
    score = ((2. * intersection + smooth) /
             (tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) + smooth))
    return score


def dice_loss(y_true, y_pred):
    loss = 1 - dice_coeff(y_true, y_pred)
    return loss


def bce_dice_loss(y_true, y_pred):
    loss = binary_crossentropy(y_true, y_pred) + dice_loss(y_true, y_pred)
    return loss


### U-NET FUNCTIONS ###


def conv_block(input_tensor, num_filters, stage, batchnorm=True):
    postfix = '_{}a'.format(stage + 1)
    encoder = Conv2D(num_filters, (3, 3), padding='same',
                     name='enc_conv' + postfix)(input_tensor)
    if batchnorm:
        encoder = BatchNormalization(name='enc_bn' + postfix)(encoder)
    encoder = Activation('relu', name='enc_act' + postfix)(encoder)

    postfix = '_{}b'.format(stage + 1)
    encoder = Conv2D(num_filters, (3, 3), padding='same',
                     name='enc_conv' + postfix)(encoder)
    if batchnorm:
        encoder = BatchNormalization(name='enc_bn' + postfix)(encoder)
    encoder = Activation('relu', name='enc_act' + postfix)(encoder)

    return encoder


def encoder_block(input_tensor, num_filters, stage, batchnorm=True):
    encoder = conv_block(input_tensor, num_filters, stage, batchnorm=batchnorm)
    encoder_pool = MaxPooling2D(
        (2, 2), strides=(2, 2), name='down_{}'.format(stage + 1))(encoder)
    return encoder_pool, encoder


def decoder_block(input_tensor, concat_tensor, num_filters, stage, batchnorm=True):
    postfix = '_{}'.format(stage + 1)
    decoder = Conv2DTranspose(num_filters, (2, 2), strides=(2, 2), padding='same',
                              name='up' + postfix)(input_tensor)
    decoder = concatenate([concat_tensor, decoder], axis=-1, name='skip' + postfix)
    if batchnorm:
        decoder = BatchNormalization(name='up_bn' + postfix)(decoder)
    decoder = Activation('relu', name='up_act' + postfix)(decoder)

    postfix = '_{}a'.format(stage + 1)
    decoder = Conv2D(num_filters, (3, 3), padding='same',
                     name='dec_conv' + postfix)(decoder)
    if batchnorm:
        decoder = BatchNormalization(name='dec_bn' + postfix)(decoder)
    decoder = Activation('relu', name='dec_act' + postfix)(decoder)

    postfix = '_{}b'.format(stage + 1)
    decoder = Conv2D(num_filters, (3, 3), padding='same',
                     name='dec_conv' + postfix)(decoder)
    if batchnorm:
        decoder = BatchNormalization(name='dec_bn' + postfix)(decoder)
    decoder = Activation('relu', name='dec_act' + postfix)(decoder)
    return decoder


def unet_block(input_tensor, layer_sizes, batchnorm=True):
    # Encoding
    upper_layer = input_tensor
    encoding_layers = []
    for i, num_filters in enumerate(layer_sizes[:-1]):
        upper_layer, encoder = encoder_block(upper_layer, num_filters, i,
                                             batchnorm=batchnorm)
        encoding_layers.append(encoder)

    # Centre
    lower_layer = conv_block(upper_layer, layer_sizes[-1], len(layer_sizes) - 1,
                             batchnorm=batchnorm)

    # Decoding
    for i, num_filters in reversed(list(enumerate(layer_sizes[:-1]))):
        lower_layer = decoder_block(lower_layer, encoding_layers[i],
                                    num_filters, i, batchnorm=batchnorm)

    return lower_layer


### MSD functions ###


def msd_block(input_tensor, depth, width, dilations, rep=1, batchnorm=True, prename=''):
    denselayer = input_tensor
    ndil = len(dilations)
    for d in range(depth):
        for w in range(width):
            idx = '_d{}_w{}'.format(d + 1, w + 1)
            dilation = dilations[((d * width + w) % ndil)]
            layer = Conv2D(
                rep, (3, 3), padding='same', dilation_rate=dilation,
                name='{}conv{}_dil{}'.format(prename, idx, dilation)
            )(denselayer)
            if batchnorm:
                layer = BatchNormalization(name=prename + 'bn' + idx)(layer)
            layer = Activation('relu', name=prename + 'act' + idx)(layer)
            denselayer = concatenate([denselayer, layer], axis=-1,
                                     name=prename + 'stack' + idx)
    return denselayer
