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


@named_model_fn('Unet')
def unet(inputs, depth=4, layer_size=8, batchnorm=True, dropout=0.):
    # Todo: make all the input arguments necessary
    layer_sizes = [layer_size*(2**i) for i in range(depth)]
    return unet_block(inputs, layer_sizes,
                      dropout=dropout, batchnorm=batchnorm)


@named_model_fn('MSD')
def msd(inputs, depth=80, width=80, n_dilations=4, dilation=1, batchnorm=True):
    dilations = [dilation * (2 ** i) for i in range(n_dilations)]
    return msd_block(inputs, depth, width, dilations, batchnorm=batchnorm)
