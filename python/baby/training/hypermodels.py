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
from baby.layers import msd_block, make_outputs, unet_block, res_block, conv_block
from baby.losses import dice_coeff, bce_dice_loss
from keras_tuner import HyperModel
from tensorflow.keras import Input, Model
from tensorflow.keras.initializers import VarianceScaling
from tensorflow.keras.optimizers import Adam
from tensorflow_addons.optimizers import AdamW



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
        # Universal variants
        width = hp.Choice('width', [8, 16, 32, 64])
        depth = hp.Int('depth', min_value=2, max_value=5)
        kernel = hp.Choice('kernel', [3, 5, 7])
        init = hp.Choice('initializer', ['glorot_uniform', 'variance_scaling'])
        if init == 'variance_scaling':
            init = VarianceScaling(2., mode='fan_out')
        activation = hp.Choice('activation', ['relu', 'gelu', 'swish'])
        conv_pool = hp.Boolean('conv_pool', default=True)
        up_activate = hp.Boolean('up_activate', default=True)
        residual_skip = hp.Boolean('residual_skip')
        enc_repeats = hp.Int('enc_repeats', min_value=2, max_value=4)
        dec_repeats = hp.Int('dec_repeats', min_value=2, max_value=4)
        block_type = hp.Choice('block_type',
                ['effnet', 'effnet-preact', 'convnext', 'conv'])
        block_args = {
                'conv': dict(block=conv_block, stem=False),
                'effnet': dict(block=res_block, stem=True),
                'effnet-preact': dict(block=res_block, pre_activate=True, stem=True),
                'convnext': dict(block=res_block, stem=True, convnext=True)
                }[block_type]
        expand_ratio, drop = 1, 0.
        if block_type != 'conv':
            expand_ratio = hp.Choice('expand_ratio', [0.5, 1., 2., 4.])
            width = width / expand_ratio
            drop = hp.Choice('block_drop', [0., 0.2])
        
        inputs = Input(shape=self.input_shape)
        layer_sizes = [width*(2**i) for i in range(depth)]
        unet = unet_block(inputs, layer_sizes, kernel=kernel, init=init,
                activation=activation, conv_pool=conv_pool,
                up_activate=up_activate, residual_skip=residual_skip,
                enc_repeats=enc_repeats, dec_repeats=dec_repeats,
                expand_ratio=expand_ratio, drop=drop, **block_args)
        model = Model(inputs=[inputs],
                      outputs=make_outputs(unet, self.outputs))
        # Todo: tuning optimizer
        model.compile(optimizer=AdamW(weight_decay=0.00025),
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
