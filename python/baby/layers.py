# If you publish results that make use of this software or the Birth Annotator
# for Budding Yeast algorithm, please cite:
# Pietsch, J.M.J., Muñoz, A.F., Adjavon, D.-Y.A., Farquhar, I., Clark, I.B.N.,
# and Swain, P.S. (2023). Determining growth rates from bright-field images of
# budding cells through identifying overlaps. eLife. 12:e79812.
# https://doi.org/10.7554/eLife.79812
# 
# 
# The MIT License (MIT)
# 
# Copyright (c) Julian Pietsch, Alán Muñoz and Diane Adjavon 2023
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
import numpy as np
from itertools import repeat
from string import ascii_lowercase as alphabet
from tensorflow.keras.layers import (
    Conv2D, BatchNormalization, Activation, MaxPooling2D,
    Conv2DTranspose, concatenate, Dropout, LayerNormalization,
    add, DepthwiseConv2D
)


### U-NET layers ###


def res_block(input_tensor, nfilters, kernel=3, expand_ratio=4, drop=0.,
        activation='swish', prefix='', suffix='', use_bias=False,
        init='glorot_uniform', pre_activate=False, post_activate=False,
        convnext=False, **kwargs):
    """A flexible residual block

    Primarily inspired from EfficientNet block, but without the squeeze and
    excite block since that is unlikely to be important for the highly
    application-specific models we typically train.

    If expand_ratio < 1 and post_activate = True, it is a special case of the
    ResNeXt block where the number of groups equals the number of features.

    If convnext=True, then expansion occurs after depth-wise convolution,
    batch normalisation is omitted in favour of a single layer normalisation
    after depth-wise convolution and there is only a single activation in the
    expanded space.
    """

    tmplt = '_'.join([e for e in (prefix, '{}', suffix) if len(e) > 0])
    lbl = lambda l: tmplt.format(l)

    x = input_tensor

    if not convnext:
        if pre_activate:
            x = BatchNormalization(name=lbl('pre_bn'))(x)
            x = Activation(activation, name=lbl('pre_act'))(x)

        # Expand from bottleneck by a factor of expand_ratio
        x = Conv2D(x.shape[-1] * expand_ratio, 1, padding='same',
                use_bias=use_bias,
                kernel_initializer=init, name=lbl('expand'))(x)
        x = BatchNormalization(name=lbl('expand_bn'))(x)
        x = Activation(activation, name=lbl('expand_act'))(x)

    # Mix in spatial dimension
    x = DepthwiseConv2D(kernel, padding='same', use_bias=use_bias,
            depthwise_initializer=init, name=lbl('mixXY'))(x)
    if convnext:
        x = LayerNormalization(epsilon=1e-6, name=lbl('mixXY_ln'))(x)
        # Expand from bottleneck
        x = Conv2D(x.shape[-1] * expand_ratio, 1, padding='same',
                use_bias=use_bias, kernel_initializer=init,
                name=lbl('expand'))(x)
        x = Activation(activation, name=lbl('expand_act'))(x)
    else:
        # Do standard norm/activation
        x = BatchNormalization(name=lbl('mixXY_bn'))(x)
        x = Activation(activation, name=lbl('mixXY_act'))(x)

    # Mix across features
    x = Conv2D(nfilters, 1, padding='same', use_bias=use_bias,
            kernel_initializer=init, name=lbl('proj'))(x)
    if not pre_activate and not convnext:
        x = BatchNormalization(name=lbl('proj_bn'))(x)

    # Stochastically drop this entire block
    if drop > 0:
        x = Dropout(drop, noise_shape=(None, 1, 1, 1), name=lbl('drop'))(x)

    # Sum at bottleneck (though, will not be bottleneck if expand_ratio > 1)
    if input_tensor.shape[-1] != nfilters:
        input_tensor = Conv2D(nfilters, 1, padding='same', use_bias=use_bias,
                kernel_initializer=init, name=lbl('input_proj'))(input_tensor)
    x = add([input_tensor, x], name=lbl('add'))

    # Post activation to match ResNeXt models
    if post_activate and not pre_activate and not convnext:
        x = Activation(activation, name=lbl('add_act'))(x)

    return x


def conv_block(input_tensor, nfilters, kernel=3, prefix='', suffix='',
        batchnorm=True, dropout=0., activation='relu', init='glorot_uniform',
        **kwargs):
    """Standard convolution with batch norm and activation
    """

    tmplt = '_'.join([e for e in (prefix, '{}', suffix) if len(e) > 0])
    lbl = lambda l: tmplt.format(l)

    x = input_tensor
    x = Conv2D(nfilters, kernel, padding='same', kernel_initializer=init,
            name=lbl('conv'))(x)
    if batchnorm:
        x = BatchNormalization(name=lbl('conv_bn'))(x)
    x = Activation(activation, name=lbl('conv_act'))(x)
    if dropout > 0:
        x = Dropout(dropout, name=lbl('conv_dropout'))(x)

    return x


def encoder_block(input_tensor, nfilters, stage, repeats=2, block=conv_block,
        dropout=0., drop=repeat(0.), batchnorm=True, conv_pool=False,
        init='glorot_uniform', **kwargs):

    x = input_tensor
    for i in range(repeats - 1):
        x = block(x, nfilters, dropout=dropout, batchnorm=batchnorm,
                prefix='enc', init=init, suffix=f'{stage + 1}{alphabet[i]}',
                **kwargs)
    encoder = block(x, nfilters, batchnorm=batchnorm, init=init, prefix='enc',
            suffix=f'{stage + 1}{alphabet[repeats - 1]}', **kwargs)
    if conv_pool:
        encoder_pool = DepthwiseConv2D(2, strides=2, padding='same', 
                kernel_initializer=init, name=f'down_{stage + 1}')(encoder)
        if batchnorm:
            encoder_pool = BatchNormalization(
                    name=f'down_bn_{stage + 1}')(encoder_pool)
    else:
        encoder_pool = MaxPooling2D(
            (2, 2), strides=(2, 2), name=f'down_{stage + 1}')(encoder)

    return encoder_pool, encoder


def decoder_block(input_tensor, skip_tensor, nfilters, stage, repeats=2,
        block=conv_block, dropout=0., drop=repeat(0.), batchnorm=True,
        prename='', init='glorot_uniform', residual_skip=False,
        up_activate=True, activation='relu', **kwargs):

    x = input_tensor
    x = Conv2DTranspose(nfilters, 2, strides=2, padding='same',
            kernel_initializer=init, name=prename + f'up_{stage + 1}')(x)
    if batchnorm:
        x = BatchNormalization(name=prename + f'up_bn_{stage + 1}')(x)
    if up_activate:
        x = Activation(activation, name=prename + f'up_act_{stage + 1}')(x)
    if residual_skip:
        x = add([skip_tensor, x], name=prename + f'up_skip_{stage + 1}')
    else:
        x = concatenate([skip_tensor, x], axis=-1,
                name=prename + f'up_skip_{stage + 1}')

    drop = iter(drop)
    for i in range(repeats - 1):
        x = block(x, nfilters, dropout=dropout, drop=next(drop),
                batchnorm=batchnorm, init=init, prefix='dec',
                suffix=f'{stage + 1}{alphabet[i]}', **kwargs)
    decoder = block(x, nfilters, drop=next(drop), batchnorm=batchnorm,
            init=init, prefix='dec',
            suffix=f'{stage + 1}{alphabet[repeats - 1]}', **kwargs)

    return decoder


def unet_block(input_tensor, layer_sizes, enc_repeats=2, dec_repeats=2,
        drop=0., dropout=0., block=conv_block, stem=False, **kwargs): 

    nlayers = len(layer_sizes)

    # Rate of stochastic block dropping increases linearly with depth
    drop = iter(np.linspace(0, drop, nlayers * enc_repeats +  (nlayers - 1) *
        dec_repeats))

    if stem:
        x = input_tensor
        x = Conv2D(layer_sizes[0], 3, padding='same', name='stem')(x)
        if not kwargs.get('pre_activate', False):
            x = BatchNormalization(name='stem_bn')(x)
            x = Activation(kwargs.get('activation', 'relu'), name='stem_act')(x)
        input_tensor = x

    # Encoding
    upper_layer = input_tensor
    encoding_layers = []
    for i, nfilters in enumerate(layer_sizes[:-1]):
        upper_layer, encoder = encoder_block(upper_layer, nfilters, i,
                repeats=enc_repeats, dropout=dropout, drop=drop, block=block,
                **kwargs)
        encoding_layers.append(encoder)

    # Centre
    x = upper_layer
    for i in range(enc_repeats - 1):
        x = block(x, layer_sizes[-1], dropout=dropout, drop=next(drop),
                prefix='enc', suffix=f'{nlayers}{alphabet[i]}', **kwargs)
    lower_layer = block(x, layer_sizes[-1], drop=next(drop), prefix='enc',
            suffix=f'{nlayers}{alphabet[enc_repeats - 1]}', **kwargs)

    # Decoding
    for i, nfilters in reversed(list(enumerate(layer_sizes[:-1]))):
        lower_layer = decoder_block(lower_layer, encoding_layers[i], nfilters,
                i, repeats=dec_repeats, dropout=dropout, drop=drop,
                block=block, **kwargs)

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
