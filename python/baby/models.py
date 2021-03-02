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
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.optimizers import Adam
from tensorflow.python.keras.layers import Input

from .utils import named_obj
from .layers import msd_block, unet_block, make_outputs
from .losses import bce_dice_loss, dice_coeff


def named_model_fn(name):

    def wrap(f):

        @named_obj(name)
        def model_fn(generator, flattener, weights={}, **kwargs):
            weights = {n: weights.get(n, 1) for n in flattener.names()}
            inputs = Input(shape=generator.shapes.input[1:])
            model = Model(inputs=[inputs],
                          outputs=make_outputs(f(inputs, **kwargs),
                                               flattener.names()))
            model.compile(optimizer=Adam(amsgrad=False),
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


@named_model_fn('msd')
def msd(inputs, depth=80, width=1, n_dilations=4, dilation=1, batchnorm=True):
    dilations = [dilation * (2 ** i) for i in range(n_dilations)]
    return msd_block(inputs, depth, width, dilations, batchnorm=batchnorm)
