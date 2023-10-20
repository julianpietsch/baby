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
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Input
from tensorflow.keras.initializers import VarianceScaling

from .utils import named_obj
from .layers import msd_block, unet_block, conv_block, res_block, make_outputs
from .losses import bce_dice_loss, dice_coeff


def named_model_fn(name):

    def wrap(f):

        @named_obj(name)
        def model_fn(generator, flattener, weights={}, use_adamw=False, **kwargs):
            weights = {n: weights.get(n, 1) for n in flattener.names()}
            inputs = Input(shape=generator.shapes.input[1:])
            model = Model(inputs=[inputs],
                          outputs=make_outputs(f(inputs, **kwargs),
                                               flattener.names()))
            if use_adamw:
                from tensorflow_addons.optimizers import AdamW
                optimizer = AdamW(weight_decay=0.00025)
            else:
                optimizer = Adam(amsgrad=False)
            model.compile(optimizer=optimizer,
                          metrics=[dice_coeff],
                          loss=bce_dice_loss,
                          loss_weights=weights)
            return model

        return model_fn

    return wrap


@named_model_fn('MSD D80')
def msd_d80(inputs):
    # Todo Remove or add deprecation warning
    return msd_block(inputs, 80, 1, [1, 2, 4, 8])


@named_model_fn('Unet 4s')
def unet_4s(inputs):
    # Todo Remove or add deprecation warning
    return unet_block(inputs, [8, 16, 32, 64], batchnorm=True)


@named_model_fn('unet')
def unet(inputs, depth=4, layer_size=8, batchnorm=True, dropout=0.):
    layer_sizes = [layer_size*(2**i) for i in range(depth)]
    return unet_block(inputs, layer_sizes,
                      dropout=dropout, batchnorm=batchnorm)


@named_model_fn('unet3')
def unet3(inputs, depth=3, layer_size=8, batchnorm=True, dropout=0.):
    layer_sizes = [layer_size*(2**i) for i in range(depth)]
    return unet_block(inputs, layer_sizes,
                      dropout=dropout, batchnorm=batchnorm)


@named_model_fn('unet_hyper')
def unet_hyper(inputs, width=64, depth=4, initializer='glorot_uniform',
        block_type='conv', **kwargs):
    layer_sizes = [width*(2**i) for i in range(depth)]
    if initializer == 'variance_scaling':
        initializer = VarianceScaling(2., mode='fan_out')
    block_args = {
            'conv': dict(block=conv_block, stem=False),
            'effnet': dict(block=res_block, stem=True),
            'effnet-preact': dict(block=res_block, pre_activate=True, stem=True),
            'convnext': dict(block=res_block, stem=True, convnext=True)
            }[block_type]
    block_args.update(kwargs)
    return unet_block(inputs, layer_sizes, init=initializer, **block_args)


@named_obj('unet_convnext')
def unet_convnext(generator, flattener, weights={}, width=8, depth=4,
        kernel=7, enc_repeats=3, dec_repeats=2, expand_ratio=4,
        activation='swish', initializer='variance_scaling', **kwargs):
    """U-net model with ConvNeXt blocks and defaults

    With default parameters, produces a model of similar size to the default
    unet model here (depth 4, width 8).
    """
    return unet_hyper(generator, flattener, weights=weights, width=width,
            depth=depth, kernel=kernel, enc_repeats=enc_repeats,
            dec_repeats=dec_repeats, block_type='convnext',
            expand_ratio=expand_ratio, activation=activation,
            initializer=initializer, use_adamw=True, residual_skip=True,
            conv_pool=True, up_activate=False, **kwargs)


@named_model_fn('msd')
def msd(inputs, depth=80, width=1, n_dilations=4, dilation=1, batchnorm=True):
    dilations = [dilation * (2 ** i) for i in range(n_dilations)]
    return msd_block(inputs, depth, width, dilations, batchnorm=batchnorm)
