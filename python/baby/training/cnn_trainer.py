import json
import matplotlib.pyplot as plt
import pickle
import tensorflow as tf
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
    def __init__(self, save_dir, cnn_set, gen, aug, flattener, max_cnns=3,
                 cnn_fn=None):
        self.flattener = flattener
        self.aug = aug
        self.gen = gen
        self._max_cnns = max_cnns
        self.save_dir = save_dir
        self.cnn_set = cnn_set
        self._cnn_fn = cnn_fn
        self._cnns = dict()

    @property
    def cnn_fn(self):
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
        d = self.save_dir / self.cnn_name
        if not d.is_dir():
            d.mkdir()
        return d

    @property
    def cnn_name(self):
        return get_name(self.cnn_fn)

    @property
    def cnn(self):
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
        history = min(self.histories.values(),
                      default=None,
                      key=lambda x: min(x['history']['val_loss']))
        if not history:
            raise BadProcess('No trained CNN models found')
        return history['file'].parent

    @property
    def opt_cnn(self):
        if self._opt_cnn is None:
            opt_dir = self.cnn_opt_dir
            opt_file = opt_dir / OPT_WEIGHTS_FILE
            if not opt_file.exists():
                raise BadProcess(
                    'Optimised model for {} model is missing'.format(
                        opt_dir.name))
            self._opt_cnn = load_model(str(opt_file),
                                       custom_objects=custom_objects)
        return self._opt_cnn

    def fit(self, epochs=400, schedule=None, replace=False, extend=False):
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

    def plot_histories(self, key='loss', log=True, window=21, ax=None,
                       save=True, legend=True):
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
