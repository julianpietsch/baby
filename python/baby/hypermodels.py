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

    def build(self, hp):
        inputs = Input(shape=self.input_shape)
        depth = hp.Int('depth', min_value=2, max_value=4, step=1)
        layer_size = hp.Choice('dilation', values=[8, 16, 32])
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
    def __init__(self, generator, flattener, weights=None):
        if weights is None:
            weights = dict()
        self.input_shape = generator.shapes.input[1:]
        self.flattener = flattener
        self.weights = {n : weights.get(n, 1) for n in flattener.names()}

    def build(self, hp):
        inputs = Input(shape=self.input_shape)
        depth = hp.Int('depth', min_value=20, max_value=120, step=20)
        width = hp.Int('width', min_value=1, max_value=5, step=1)
        dilation = hp.Choice('dilation', values=[1, 2, 4])
        dilations = [dilation*(2**i) for i in range(depth)]
        batchnorm = hp.Boolean('batchnorm', default=True)
        msd = msd_block(inputs, depth, width, dilations, batchnorm=batchnorm)
        model = Model(inputs=[inputs],
                      outputs=make_outputs(msd, self.flattener.names()))
        # Todo: tuning optimizer
        model.compile(optimizer=Adam(amsgrad=False),
                      metrics=[dice_coeff],
                      loss=bce_dice_loss,
                      loss_weights=self.weights)
        return model