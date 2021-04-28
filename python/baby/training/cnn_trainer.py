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
import json
import pathlib
from typing import List, Tuple

import matplotlib.pyplot as plt
import pickle
import tensorflow as tf
from baby.augmentation import Augmenter
from baby.generator import ImageLabel
from baby.preprocessing import SegmentationFlattening
from matplotlib.colors import to_rgba
from scipy.signal import savgol_filter
from tensorflow.python.keras.callbacks import ModelCheckpoint, TensorBoard, \
    LearningRateScheduler
from tensorflow.python.keras.models import load_model

from baby import models
from baby.errors import BadType, BadProcess
from baby.losses import bce_dice_loss, dice_coeff
from baby.utils import get_name, schedule_steps

custom_objects = {'bce_dice_loss': bce_dice_loss, 'dice_coeff': dice_coeff}

OPT_WEIGHTS_FILE = 'weights.h5'
INIT_WEIGHTS_FILE = 'init_weights.h5'
FINAL_WEIGHTS_FILE = 'final_weights.h5'
HISTORY_FILE = 'history.pkl'
LOG_DIR = 'logs'


class CNNTrainer:
    """
    Methods for optimising the weights of the CNNs using gradient descent.
    """
    def __init__(self, save_dir: pathlib.Path,
                 cnn_set: tuple, gen: ImageLabel, aug: Augmenter,
                 flattener: SegmentationFlattening,
                 max_cnns: int = 3,
                 cnn_fn: str = None):
        """
        Methods for optimising the weights of the CNNs using gradient descent.

        :param save_dir: base directory in which to save weights and outputs
        :param cnn_set: the names of CNN architectures to be trained
        :param gen: the data generator
        :param aug: the data augmentor
        :param flattener: the data flattener
        :param max_cnns: the maximum number of CNNs to train/keep, default is 3
        :param cnn_fn: the CNN architecture to start with, defaults to None
        """
        self.flattener = flattener
        self.aug = aug
        self.gen = gen
        # TODO no longer needed with hyperparmeter optim
        self._max_cnns = max_cnns
        self.save_dir = save_dir
        self.cnn_set = cnn_set
        self._cnn_fn = cnn_fn
        self._cnns = dict()
        self._opt_cnn = None

    @property
    def cnn_fn(self):
        """The current CNN architecture function."""
        if self._cnn_fn is None:
            self.cnn_fn = self.cnn_set[0]
        return getattr(models, self._cnn_fn)

    @cnn_fn.setter
    def cnn_fn(self, fn):
        if fn not in self.cnn_set:
            raise BadType('That model is not in "parameters.cnn_set"')
        if not hasattr(models, fn):
            raise BadType('That is not a recognised model')
        self._cnn_fn = fn

    @property
    def cnn_dir(self):
        """The subdirectory into which to save the current CNN output."""
        d = self.save_dir / self.cnn_name
        if not d.is_dir():
            d.mkdir()
        return d

    @property
    def cnn_name(self):
        """The name of the currently active CNN architecture."""
        return get_name(self.cnn_fn)

    @property
    def cnn(self):
        """The keras Model for the active CNN."""
        if self.cnn_name not in self._cnns:
            if len(self._cnns) > self._max_cnns:
                # To avoid over-consuming memory reset graph
                # TODO: ensure TF1/TF2 compat and check RTX bug
                tf.keras.backend.clear_session()
                # Reset any potentially loaded models
                self._cnns = dict()
                self._opt_cnn = None
            # Todo: separate generator from trainer
            #   Make model accept an input shape and a set of outputs
            self.gen.train.aug = self.aug.train
            print('Loading "{}" CNN...'.format(self.cnn_name))
            # Load hyperparameters
            hyper_param_file = self.cnn_dir / "hyperparameters.json"
            if not hyper_param_file.exists():
                # Todo: just use defaults
                raise FileNotFoundError("Hyperparameter file {} for {} not "
                                        "found.".format(hyper_param_file,
                                                        self.cnn_name))
            with open(hyper_param_file, 'r') as fd:
                hyperparameters = json.load(fd)
            model = self.cnn_fn(self.gen.train, self.flattener,
                                **hyperparameters)
            self._cnns[self.cnn_name] = model

            # Save initial weights if they haven't already been saved
            filename = self.cnn_dir / INIT_WEIGHTS_FILE
            if not filename.exists():
                print('Saving initial weights...')
                model.save_weights(str(filename))
        return self._cnns[self.cnn_name]

    @property
    def histories(self):
        """A CNN name to training history dictionary."""
        # Always get the most up-to-date version from disk
        hdict = {}
        active_cnn_fn = self.cnn_name
        try:
            for cnn_id in self.cnn_set:
                self.cnn_fn = cnn_id
                history_file = self.cnn_dir / HISTORY_FILE
                if not history_file.exists():
                    continue
                with open(history_file, 'rb') as f:
                    history = pickle.load(f)
                history['name'] = self.cnn_name
                history['file'] = history_file
                hdict[cnn_id] = history
        finally:
            self.cnn_fn = active_cnn_fn
        return hdict

    @property
    def opt_dir(self):
        """The directory in which the weights of the best CNN are saved."""
        history = min(self.histories.values(),
                      default=None,
                      key=lambda x: min(x['history']['val_loss']))
        if not history:
            raise BadProcess('No trained CNN models found')
        return history['file'].parent

    @property
    def opt_cnn(self):
        """The keras model for the CNN with the lowest loss."""
        if self._opt_cnn is None:
            opt_dir = self.opt_dir
            opt_file = opt_dir / OPT_WEIGHTS_FILE
            if not opt_file.exists():
                raise BadProcess(
                    'Optimised model for {} model is missing'.format(
                        opt_dir.name))
            self._opt_cnn = load_model(str(opt_file),
                                       custom_objects=custom_objects)
        return self._opt_cnn

    def fit(self, epochs: int = 400,
            schedule: List[Tuple[int, float]] = None,
            replace: bool = False,
            extend: bool = False):
        """Fit the active CNN to minimise loss on the (augmented) generator.

        :param epochs: number of epochs to train, defaults to 400
        :param schedule: learning rate schedule, defaults to fixed rate 0.001
        :param replace: force training if CNN already has a weights file
        :param extend: train using existing CNN weights as initial weights
        """
        # First check output names match current flattener names
        assert (all([
            m == f
            for m, f in zip(self.cnn.output_names, self.flattener.names())
        ]))

        if schedule is None:
            schedule = [(1e-3, epochs)]

        finalfile = self.cnn_dir / FINAL_WEIGHTS_FILE
        if extend:
            self.cnn.load_weights(str(finalfile))
        else:
            initfile = self.cnn_dir / INIT_WEIGHTS_FILE
            self.cnn.load_weights(str(initfile))

        optfile = self.cnn_dir / OPT_WEIGHTS_FILE
        if not replace and optfile.is_file():
            raise BadProcess('Optimised weights already exist')

        logdir = self.cnn_dir / LOG_DIR
        callbacks = [
            ModelCheckpoint(filepath=str(optfile),
                            monitor='val_loss',
                            save_best_only=True,
                            verbose=1),
            TensorBoard(log_dir=str(logdir)),
            LearningRateScheduler(
                lambda epoch: schedule_steps(epoch, schedule))
        ]
        self.gen.train.aug = self.aug.train
        self.gen.val.aug = self.aug.val
        history = self.cnn.fit_generator(generator=self.gen.train,
                                         validation_data=self.gen.val,
                                         epochs=epochs,
                                         callbacks=callbacks)

        # Save history
        with open(self.cnn_dir / HISTORY_FILE, 'wb') as f:
            pickle.dump({
                'history': history.history,
                'epoch': history.epoch
            }, f)

        print('Saving final weights...')
        self.cnn.save_weights(str(finalfile))

    def plot_histories(self, key: str = 'loss',
                       log: bool = True, window: int = 21,
                       ax: plt.axis = None,
                       save: bool = True,
                       legend: bool = True):
        """
        Plot the loss/metric histories of all of the trained CNNs.

        # TODO add an image as an example

        :param key: which metric to plot, defaults to loss
        :param log: set y-axis to log scale, defaults to True
        :param window: filter window for Savitsky-Golay filter applied to
        metric before plotting, defaults to 21 which assumes at least 22
        epochs of training, 50 suggested.
        :param ax: axis on which to plot the losses, defaults to None
        :param save: save the plot to `save_dir / f"histories_{key}.png"`,
        defaults to True
        :param legend: adds a legend to the plot, defaults to True
        :return:
        """
        if save:
            fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(4, 4))
        if ax is None:
            ax = plt.gca()

        max_epoch = 1
        hdict = self.histories
        for history in hdict.values():
            epoch = history['epoch']
            max_epoch = max([max_epoch, max(epoch)])
            val = history['history']['val_' + key]
            hndl = ax.plot(epoch,
                           savgol_filter(val, window, 3),
                           label=history['name'] + ' Val')
            val = history['history'][key]
            colour = to_rgba(hndl[0].get_color(), 0.7)
            ax.plot(epoch,
                    savgol_filter(val, window, 3),
                    ':',
                    color=colour,
                    label=history['name'] + ' Train')

        ax.set(xlabel='Epochs',
               ylabel=key.replace('_', ' ').title(),
               xlim=[0, max_epoch])
        if log:
            ax.set_yscale('log')
        if legend:
            ax.legend()

        if save:
            fig.savefig(self.save_dir / 'histories_{}.png'.format(key))
            plt.close(fig)
