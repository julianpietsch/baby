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
from baby.layers import msd_block, make_outputs, unet_block
from baby.losses import dice_coeff, bce_dice_loss
from kerastuner import HyperModel
from tensorflow.python.keras import Input, Model
from tensorflow.python.keras.optimizers import Adam



class UNet(HyperModel):
    def __init__(self, input_shape, outputs, weights=None):
        if weights is None:
            weights = dict()
        self.weights = {n: weights.get(n, 1) for n in outputs}
        self.input_shape = input_shape
        self.outputs = outputs
        self.name = 'Unet'
        # TODO output defaults

    @property
    def defaults(self):
        return dict(depth=4, layer_size=8, batchnorm=True, dropout=0.0)

    def build(self, hp):
        inputs = Input(shape=self.input_shape)
        depth = hp.Int('depth', min_value=2, max_value=4, step=1)
        layer_size = hp.Choice('layer_size', values=[8, 16, 32])
        layer_sizes = [layer_size*(2**i) for i in range(depth)]
        batchnorm = hp.Boolean('batchnorm', default=True)
        dropout = hp.Float('dropout', min_value=0., max_value=0.7, step=0.1)
        unet = unet_block(inputs, layer_sizes, batchnorm=batchnorm,
                          dropout=dropout)
        model = Model(inputs=[inputs],
                      outputs=make_outputs(unet, self.outputs))
        # Todo: tuning optimizer
        model.compile(optimizer=Adam(amsgrad=False),
                      metrics=[dice_coeff],
                      loss=bce_dice_loss,
                      loss_weights=self.weights)
        return model

class MSD(HyperModel):
    def __init__(self, input_shape, outputs, weights=None):
        if weights is None:
            weights = dict()
        self.input_shape = input_shape
        self.outputs = outputs
        self.weights = {n: weights.get(n, 1) for n in self.outputs}
        self.name = 'MSD'
        self.defaults = dict(depth=80, width=1, n_dilations=4, dilation=1,
                             batnorm=True)

    def build(self, hp):
        inputs = Input(shape=self.input_shape)
        depth = hp.Int('depth', min_value=20, max_value=120, step=20)
        width = hp.Int('width', min_value=1, max_value=1, step=1)
        n_dilations = hp.Choice('n_dilations', values=[2, 4, 8])
        dilation = hp.Choice('dilation', values=[1, 2, 4])
        dilations = [dilation*(2**i) for i in range(n_dilations)]
        batchnorm = hp.Boolean('batchnorm', default=True)
        msd = msd_block(inputs, depth, width, dilations, batchnorm=batchnorm)
        model = Model(inputs=[inputs],
                      outputs=make_outputs(msd, self.outputs))
        # Todo: tuning optimizer
        model.compile(optimizer=Adam(amsgrad=False),
                      metrics=[dice_coeff],
                      loss=bce_dice_loss,
                      loss_weights=self.weights)
        return model


available_hypermodels=dict(
    unet=UNet,
    msd=MSD
)

def get_hypermodel(name, input_shape, outputs, weights=None):
    mdl_fn = available_hypermodels.get(name, None)
    if mdl_fn is None:
        raise("Model {} not available.".format(name))
    return mdl_fn(input_shape, outputs, weights)
