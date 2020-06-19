from tensorflow.python.keras.models import Model
from tensorflow.python.keras.optimizers import Adam
from tensorflow.python.keras.layers import Input

from .utils import named_obj
from .layers import msd_block, make_outputs
from .losses import bce_dice_loss, dice_coeff


def named_model_fn(f, name):

    @named_obj(name)
    def model_fn(generator, flattener, weights={}):
        weights = {n: weights.get(n, 1) for n in flattener.names()}
        inputs = Input(shape=generator.shape.input)
        model = Model(inputs=[inputs],
                      outputs=make_outputs(f(inputs), flattener.names()))
        model.compile(optimizer=Adam(amsgrad=False),
                      metrics=[dice_coeff],
                      loss=bce_dice_loss,
                      loss_weights=weights)

    return model_fn


@named_model_fn('MSD D80')
def msd_d80(inputs):
    return msd_block(inputs, 80, 1, [1, 2, 4, 8])
