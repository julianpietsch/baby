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
from typing import List, Tuple, Union
from types import MappingProxyType
import json
import pathlib
import pickle
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgba
from scipy.signal import savgol_filter
import tensorflow as tf
from tensorflow.keras.callbacks import (ModelCheckpoint, TensorBoard,
                                        LearningRateScheduler, CSVLogger)
from tensorflow.keras.models import load_model

from baby.augmentation import Augmenter
from baby.generator import ImageLabel
from baby.preprocessing import SegmentationFlattening
from baby import models
from baby.errors import BadType, BadProcess
from baby.losses import bce_dice_loss, dice_coeff
from baby.utils import get_name, schedule_steps
from .utils import SharedParameterContainer
from .flattener_trainer import FlattenerTrainer

custom_objects = {'bce_dice_loss': bce_dice_loss, 'dice_coeff': dice_coeff}

OPT_WEIGHTS_FILE = 'weights.h5'
INIT_WEIGHTS_FILE = 'init_weights.h5'
FINAL_WEIGHTS_FILE = 'final_weights.h5'
HISTORY_FILE = 'history.pkl'
CSV_HISTORY_FILE = 'history.csv'
LOG_DIR = 'logs'
HYPER_PARAMS_FILE = 'hyperparameters.json'


class CNNTrainer:
    """Methods for optimising CNN weights using gradient descent.

    After training, the optimised weights are be stored in a file named
    :py:data:`OPT_WEIGHTS_FILE` within the :py:attr:`cnn_dir` directory. The
    directory is specific to each of the architectures listed in ``cnn_set``.

    Args:
        shared_params: Training and segmentation parameters as provided by
            :py:class:`utils.SharedParameterContainer`.
        flattener_trainer: Trainer that defines optimised targets for the CNN.
        max_cnns: The maximum number of CNNs to keep in memory, default is 3.
        cnn_fn: The CNN architecture to start with. Defaults to the
            :py:attr:`utils.BabyTrainingParameters.cnn_fn` architecture
            as found in ``shared_params.parameters``.
    """
    def __init__(self,
                 shared_params: SharedParameterContainer,
                 flattener_trainer: FlattenerTrainer,
                 max_cnns: int = 3,
                 cnn_fn: str = None):
        self._shared_params = shared_params
        self._flattener_trainer = flattener_trainer

        # TODO no longer needed with hyperparmeter optim
        self._max_cnns = max_cnns
        self._cnn_fn = cnn_fn
        self._cnns = dict()
        self._opt_cnn = None

    @property
    def save_dir(self):
        """Base directory in which to save trained models"""
        return self._shared_params.save_dir

    @property
    def cnn_set(self):
        return self._shared_params.parameters.cnn_set

    @property
    def gen(self):
        """Data generators used for training models.

        A :py:class:`utils.TrainValTestProperty` of training, validation and
        test data generators with augmentations obtained from
        :py:attr:`FlattenerTrainer.default_gen`.
        """
        return self._flattener_trainer.default_gen

    @property
    def flattener(self):
        """Target definitions for models trained by this class."""
        return self._flattener_trainer.flattener

    @property
    def cnn_fn(self):
        """The current CNN architecture function."""
        if self._cnn_fn is None:
            if self._shared_params.parameters.cnn_fn is None:
                self.cnn_fn = self.cnn_set[0]
            else:
                self.cnn_fn = self._shared_params.parameters.cnn_fn
        return getattr(models, self._cnn_fn)

    @cnn_fn.setter
    def cnn_fn(self, fn):
        if fn not in self.cnn_set:
            raise BadType('That model is not in "parameters.cnn_set"')
        if not hasattr(models, fn):
            raise BadType('That is not a recognised model')
        self._cnn_fn = fn
        self._hyperparameters = None

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
    def hyperparameters(self):
        """Custom hyperparameters defined for the active CNN architecture.

        A ``dict`` specifying keyword arguments to be passed when building the
        active CNN architecture. If left unset, any defaults as given in
        :py:mod:`baby.models` will be used.

        NB: The property is returned as a :py:class:`types.MappingProxyType`,
        so ``dict`` items cannot be modified. To change hyperparameters, the
        whole ``dict`` needs to be replaced.
        """
        if not getattr(self, '_hyperparameters', None):
            # Load hyperparameters
            hyper_param_file = self.cnn_dir / "hyperparameters.json"
            if hyper_param_file.exists():
                with open(hyper_param_file, 'r') as fd:
                    self._hyperparameters = json.load(fd)
            else:
                # Use defaults specified in `models`
                self._hyperparameters = {}
        return MappingProxyType(self._hyperparameters)

    @hyperparameters.setter
    def hyperparameters(self, params):
        if not type(params) == dict or type(params) == MappingProxyType:
            raise BadType('Hyperparameters must be specified as a `dict`')
        hyper_param_file = self.cnn_dir / "hyperparameters.json"
        with open(hyper_param_file, 'w') as f:
            json.dump(dict(params), f)
        self._hyperparameters = dict(params)
        # If the hyperparameters have changed, we need to regenerate the model
        # and initial weights
        if self.cnn_name in self._cnns:
            del self._cnns[self.cnn_name]
        # Delete initial weights if they have already been saved
        init_weights_file = self.cnn_dir / INIT_WEIGHTS_FILE
        init_weights_file.unlink(missing_ok=True)

    @property
    def cnn(self):
        """The keras Model for the active CNN."""
        if self.cnn_name not in self._cnns:
            n_loaded = getattr(self, '_n_cnns_loaded', 0)
            if n_loaded > self._max_cnns:
                # To avoid over-consuming memory reset graph
                # TODO: ensure TF1/TF2 compat and check RTX bug
                tf.keras.backend.clear_session()
                # Reset any potentially loaded models
                self._cnns = dict()
                self._opt_cnn = None
                n_loaded = 0

            print('Loading "{}" CNN...'.format(self.cnn_name))
            model = self.cnn_fn(self.gen.train, self.flattener,
                                **self.hyperparameters)
            self._cnns[self.cnn_name] = model
            self._n_cnns_loaded = n_loaded + 1

            # Save initial weights if they haven't already been saved
            init_weights_file = self.cnn_dir / INIT_WEIGHTS_FILE
            if not init_weights_file.exists():
                print('Saving initial weights...')
                model.save_weights(str(init_weights_file))
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
            schedule: Union[str, List[Tuple[int, float]]] = None,
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

        if callable(schedule):
            schedulefn = schedule
        else:
            schedulefn = lambda epoch: schedule_steps(epoch, schedule)

        finalfile = self.cnn_dir / FINAL_WEIGHTS_FILE
        if extend:
            self.cnn.load_weights(str(finalfile))
        else:
            initfile = self.cnn_dir / INIT_WEIGHTS_FILE
            self.cnn.load_weights(str(initfile))

        optfile = self.cnn_dir / OPT_WEIGHTS_FILE
        if not replace and optfile.is_file():
            raise BadProcess('Optimised weights already exist')

        csv_history_file = self.cnn_dir / CSV_HISTORY_FILE
        logdir = self.cnn_dir / LOG_DIR
        callbacks = [
            ModelCheckpoint(filepath=str(optfile),
                            monitor='val_loss',
                            save_best_only=True,
                            verbose=1),
            CSVLogger(filename=str(csv_history_file), append=extend),
            TensorBoard(log_dir=str(logdir)),
            LearningRateScheduler(schedulefn)
        ]

        if tf.version.VERSION.startswith('1'):
            history = self.cnn.fit_generator(generator=self.gen.train,
                                             validation_data=self.gen.val,
                                             epochs=epochs,
                                             callbacks=callbacks)
        else:
            history = self.cnn.fit(self.gen.train,
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
