from tensorflow.python.keras.models import Model
from tensorflow.python.keras.optimizers import Adam
from tensorflow.python.keras import layers as KL

from .layers import msd_block, make_outputs
from .losses import bce_dice_loss, dice_coeff


def msd_d80(generator, flattener, weights={}):
    weights = {n: weights.get(n, 1) for n in flattener.names()}
    inputs = KL.Input(shape=generator.shape.input)
    nn = msd_block(inputs, 80, 1, [1, 2, 4, 8])
    model = Model(inputs=[inputs],
                  outputs=make_outputs(nn, flattener.names()))
    model.compile(optimizer=Adam(amsgrad=False),
                  metrics=[dice_coeff],
                  loss=bce_dice_loss,
                  loss_weights=weights)
    return model
