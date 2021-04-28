# If you publish results that make use of this software or the Birth Annotator
# for Budding Yeast algorithm, please cite:
# Julian M J Pietsch, Alán Muñoz, Diane Adjavon, Ivan B N Clark, Peter S
# Swain, 2021, Birth Annotator for Budding Yeast (in preparation).
# 
# 
# The MIT License (MIT)
# 
# Copyright (c) Julian Pietsch, Alán Muñoz and Diane Adjavon 2021
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to
# deal in the Software without restriction, including without limitation the
# rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
# sell copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# 
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
# IN THE SOFTWARE.
from tensorflow.python.keras.layers import (
    Conv2D, BatchNormalization, Activation, MaxPooling2D,
    Conv2DTranspose, concatenate, Dropout
)


### U-NET layers ###


def conv_block(input_tensor, num_filters, stage, batchnorm=True, dropout=0.):
    postfix = '_{}a'.format(stage + 1)
    encoder = Conv2D(num_filters, (3, 3), padding='same',
                     name='enc_conv' + postfix)(input_tensor)
    if batchnorm:
        encoder = BatchNormalization(name='enc_bn' + postfix)(encoder)
    encoder = Activation('relu', name='enc_act' + postfix)(encoder)
    if dropout > 0:
        encoder = Dropout(dropout, name='enc_dropout' + postfix)(encoder)

    postfix = '_{}b'.format(stage + 1)
    encoder = Conv2D(num_filters, (3, 3), padding='same',
                     name='enc_conv' + postfix)(encoder)
    if batchnorm:
        encoder = BatchNormalization(name='enc_bn' + postfix)(encoder)
    encoder = Activation('relu', name='enc_act' + postfix)(encoder)

    return encoder


def encoder_block(input_tensor, num_filters, stage, batchnorm=True,
                  dropout=0.):
    encoder = conv_block(input_tensor, num_filters, stage,
                         batchnorm=batchnorm,
                         dropout=dropout)
    encoder_pool = MaxPooling2D(
        (2, 2), strides=(2, 2), name='down_{}'.format(stage + 1))(encoder)
    return encoder_pool, encoder


def decoder_block(input_tensor, concat_tensor, num_filters, stage,
                  batchnorm=True, prename='', dropout=0.):
    postfix = '_{}'.format(stage + 1)
    decoder = Conv2DTranspose(num_filters, (2, 2), strides=(2, 2), padding='same',
                              name=prename + 'up' + postfix)(input_tensor)
    decoder = concatenate([concat_tensor, decoder], axis=-1,
                          name=prename + 'skip' + postfix)
    if batchnorm:
        decoder = BatchNormalization(name=prename + 'up_bn' + postfix)(decoder)
    decoder = Activation('relu', name=prename + 'up_act' + postfix)(decoder)
    if dropout > 0:
        decoder = Dropout(dropout, name='dec_dropout' + postfix)(decoder)

    postfix = '_{}a'.format(stage + 1)
    decoder = Conv2D(num_filters, (3, 3), padding='same',
                     name=prename + 'dec_conv' + postfix)(decoder)
    if batchnorm:
        decoder = BatchNormalization(name=prename + 'dec_bn' + postfix)(decoder)
    decoder = Activation('relu', name=prename + 'dec_act' + postfix)(decoder)
    if dropout > 0:
        decoder = Dropout(dropout, name='dec_dropout' + postfix)(decoder)

    postfix = '_{}b'.format(stage + 1)
    decoder = Conv2D(num_filters, (3, 3), padding='same',
                     name=prename + 'dec_conv' + postfix)(decoder)
    if batchnorm:
        decoder = BatchNormalization(name=prename + 'dec_bn' + postfix)(decoder)
    decoder = Activation('relu', name=prename + 'dec_act' + postfix)(decoder)
    if dropout > 0:
        decoder = Dropout(dropout, name='dec_dropout' + postfix)(decoder)
    return decoder


def unet_block(input_tensor, layer_sizes, batchnorm=True, dropout=0.):
    # Encoding
    upper_layer = input_tensor
    encoding_layers = []
    for i, num_filters in enumerate(layer_sizes[:-1]):
        upper_layer, encoder = encoder_block(upper_layer, num_filters, i,
                                             batchnorm=batchnorm,
                                             dropout=dropout)
        encoding_layers.append(encoder)

    # Centre
    lower_layer = conv_block(upper_layer, layer_sizes[-1], len(layer_sizes) - 1,
                             batchnorm=batchnorm, dropout=dropout)

    # Decoding
    for i, num_filters in reversed(list(enumerate(layer_sizes[:-1]))):
        lower_layer = decoder_block(lower_layer, encoding_layers[i],
                                    num_filters, i, batchnorm=batchnorm,
                                    dropout=dropout)

    return lower_layer


### MSD layers ###


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


### General layers ###


def make_outputs(nn_core, target_names):
    if type(target_names) == str:
        return Conv2D(1, (1, 1), activation='sigmoid', name=target_names)(nn_core)
    else:
        return [Conv2D(1, (1, 1), activation='sigmoid', name=name)(nn_core)
                for name in target_names]


def bottleneck(nn, nlayers=8, prename='btl', batchnorm=True):
    nn = Conv2D(nlayers, (1, 1), padding='same', name=prename + '_conv')(nn)
    if batchnorm:
        nn = BatchNormalization(name=prename + '_bn')(nn)
    return Activation('relu', name=prename + '_act')(nn)
