import json
import pickle
import shutil
import warnings
from contextlib import contextmanager
from pathlib import Path
from typing import NamedTuple, Union, Tuple, Any
import numpy as np
from baby.hypermodels import get_hypermodel
from kerastuner import RandomSearch, Tuner, Hyperband, BayesianOptimization, \
    HyperModel
from numpy.polynomial import Polynomial
from scipy.optimize import curve_fit
import pandas as pd
from tensorflow.python.keras.models import load_model
from tqdm import trange
from skimage import filters
from skimage import transform
from skimage.measure import regionprops
from scipy.signal import savgol_filter
from scipy.ndimage.morphology import binary_fill_holes, binary_erosion
from matplotlib.colors import to_rgba
from matplotlib import pyplot as plt
import tensorflow as tf

if tf.__version__.startswith('2'):
    tf.compat.v1.disable_eager_execution()

from tensorflow.python.keras.callbacks import (ModelCheckpoint, TensorBoard,
                                               LearningRateScheduler)

from .utils import (get_name, EncodableNamedTuple, find_file,
                    as_python_object, jsonify, schedule_steps)
from .errors import BadParam, BadFile, BadType, BadProcess
from .io import TrainValPairs
from .preprocessing import (robust_norm, seg_norm, SegmentationFlattening,
                            dwsquareconn)
from .augmentation import (Augmenter, SmoothingSigmaModel, ScalingAugmenter,
                           _filled_canny, _apply_crop)
from .generator import ImageLabel
from .visualise import colour_segstack
from .losses import bce_dice_loss, dice_coeff
from . import models
from .segmentation import (binary_edge, mask_iou, squareconn,
                           morph_radial_thresh_fit, draw_radial)
from .track_trainer import TrackTrainer, BudTrainer

# from bud_test import BudTrainer

custom_objects = {'bce_dice_loss': bce_dice_loss, 'dice_coeff': dice_coeff}

OPT_WEIGHTS_FILE = 'weights.h5'
INIT_WEIGHTS_FILE = 'init_weights.h5'
FINAL_WEIGHTS_FILE = 'final_weights.h5'
HISTORY_FILE = 'history.pkl'
LOG_DIR = 'logs'


def fix_tf_rtx_gpu_bug():
    """Run to set up TensorFlow session with RTX series NVidia GPUs
    
    Compensates for cuDNN bug in tensorflow + RTX series NVidia GPUs.
    """
    if tf.version.VERSION.startswith('1'):
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        tf_session = tf.Session(config=config)
        tf.keras.backend.set_session(tf_session)
    elif tf.version.VERSION.startswith('2'):
        gpus = tf.config.experimental.list_physical_devices('GPU')
        if gpus:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logical_gpus = tf.config.experimental.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,", len(logical_gpus),
                  "Logical GPUs")
        else:
            raise Exception(
                'Unsupported version of tensorflow encountered ({})'.format(
                    tf.version.VERSION))


class TrainValProperty(NamedTuple):
    train: Any
    val: Any


@EncodableNamedTuple
class BabyTrainerParameters(NamedTuple):
    """Global parameters for training BABY models
    """
    train_val_pairs_file: str = 'train_val_pairs.json'
    smoothing_sigma_stats_file: str = 'smoothing_sigma_stats.csv'
    smoothing_sigma_model_file: str = 'smoothing_sigma_model.json'
    flattener_stats_file: str = 'flattener_stats.json'
    flattener_file: str = 'flattener.json'
    cnn_set: Tuple[str, ...] = ('msd_d80', 'unet_4s')
    cnn_fn: Union[None, str] = None
    batch_size: int = 8
    in_memory: bool = True
    xy_out: int = 80
    target_pixel_size: float = 0.263
    substacks: Union[None, int] = None


@contextmanager
def augmented_generator(gen: ImageLabel, aug: Augmenter):
    # Save the previous augmenter if any
    saved_aug = gen.aug
    gen.aug = aug
    try:
        yield gen
    # Todo: add except otherwise there might be an issue of there is an error?
    finally:
        gen.aug = saved_aug


class SmoothingModelTrainer:
    def __init__(self, save_dir, stats_file, model_file):
        self.save_dir = save_dir
        self.stats_file = save_dir / stats_file
        self.model_file = save_dir / model_file
        self._model = None

    def generate_smoothing_sigma_stats(self, train_gen, val_gen):
        with augmented_generator(train_gen, lambda x, y: (x, y)) as gen:
            sss_train = _generate_smoothing_sigma_stats(gen)
        with augmented_generator(val_gen, lambda x, y: (x, y)) as gen:
            sss_val = _generate_smoothing_sigma_stats(gen)
        sss_train['validation'] = False
        sss_val['validation'] = True
        sss = pd.concat((sss_train, sss_val))
        sss.to_csv(self.stats_file)

    @property
    def stats(self):
        if self._model is None:
            if not self.stats_file.exists():
                raise BadProcess(
                    'smoothing sigma stats have not been generated')
            self._model = pd.read_csv(self.stats_file)
        return TrainValProperty(self._model[~self._model['validation']],
                                self._model[self._model['validation']])

    @property
    def model(self):
        if self._model is None:
            if self.model_file.is_file():
                smoothing_sigma_model = SmoothingSigmaModel()
                smoothing_sigma_model.load(self.model_file)
                self._model = smoothing_sigma_model
            else:
                raise BadProcess(
                    'The "smoothing_sigma_model" has not been assigned yet')
        return self._model

    @model.setter
    def model(self, ssm):
        if isinstance(ssm, str):
            ssm_file = find_file(ssm, self.save_dir, 'smoothing_sigma_model')
            ssm = SmoothingSigmaModel()
            ssm.load(ssm_file)
        if not isinstance(ssm, SmoothingSigmaModel):
            raise BadType(
                '"smoothing_sigma_model" must be of type "baby.augmentation.SmoothingSigmaModel"'
            )
        ssm.save(self.model_file)
        self._model = ssm

    def _get_grouped_sss(self):
        group_best_iou = lambda x: x.loc[x['iou'].idxmax(), :]
        idcols = ['ind', 'cell', 'scaling', 'rotation']
        stats = self.stats.train
        stats = stats.groupby(idcols).apply(group_best_iou)
        filts = {
            'identity': (stats.scaling == 1) & (stats.rotation == 0),
            'scaling': stats.scaling != 1,
            'rotation': stats.rotation != 0
        }
        return stats, filts

    def fit(self, filt='identity'):
        stats, filts = self._get_grouped_sss()

        if filt:
            stats = stats[filts[filt]]

        # Get initial parameters from linear fit of log transformed nedge
        b = 10  # initial guess for offset term in final model
        # Fit s = c + m * log(n - b); want n = b + exp((s - c)/m)
        pinv = Polynomial.fit(np.log(np.clip(stats.nedge - b, 1, None)),
                              stats.sigma,
                              deg=1)
        c = pinv(0)
        m = pinv(1) - c

        # Fit n = b + a * exp(p * s); inverse: s = log(n - c) / p - log(a) / p
        model = lambda s, a, p, b: b + a * np.exp(p * s)
        p0 = (np.exp(-c / m), 1 / m, b)
        params, _ = curve_fit(model, stats.sigma, stats.nedge, p0=p0)

        self.model = SmoothingSigmaModel(*params)

    def plot_fitted_model(self):
        stats, filts = self._get_grouped_sss()
        model = lambda s, a, p, b: b + a * np.exp(p * s)
        params = (self.model._a, self._model._b, self.model._c)

        fig, axs = plt.subplots(2,
                                len(filts),
                                figsize=(12, 12 * 2 / len(filts)))
        sigma_max = stats.sigma.max()
        nedge_max = stats.nedge.max()
        sigma = np.linspace(0, sigma_max, 100)
        for ax, (k, f) in zip(axs[0], filts.items()):
            ax.scatter(stats[f].sigma,
                       stats[f].nedge,
                       16,
                       alpha=0.05,
                       edgecolors='none')
            ax.plot(sigma, model(sigma, *params), 'r')
            ax.set(title=k.title(),
                   xlabel='sigma',
                   ylabel='nedge',
                   ylim=[0, nedge_max])

        nedge = np.linspace(1, nedge_max, 100)
        for ax, (k, f) in zip(axs[1], filts.items()):
            ax.scatter(stats[f].nedge,
                       stats[f].sigma,
                       16,
                       alpha=0.05,
                       edgecolors='none')
            ax.plot(nedge, [self.model(n) for n in nedge], 'r')
            ax.set(title=k.title(),
                   xlabel='nedge',
                   ylabel='sigma',
                   ylim=[0, sigma_max])

        fig.savefig(self.save_dir / 'fitted_smoothing_sigma_model.png')


class FlattenerTrainer:
    def __init__(self, save_dir, stats_file, flattener_file):
        self.save_dir = save_dir
        self.stats_file = self.save_dir / stats_file
        self.flattener_file = self.save_dir / flattener_file
        self._flattener = None
        self._stats = None

    def generate_flattener_stats(self, train_gen, val_gen,
                                 train_aug, val_aug, max_erode=5):
        # Set up temporary flattener
        old_flattener = getattr(self, '_flattener', None)
        self.flattener = lambda x, y: x
        try:
            with augmented_generator(train_gen, train_aug) as gen:
                fs_train = _generate_flattener_stats(gen, max_erode)
            with augmented_generator(val_gen, val_aug) as gen:
                fs_val = _generate_flattener_stats(gen, max_erode)
        finally:
            self.flattener = old_flattener

        with open(self.stats_file, 'wt') as f:
            json.dump({'train': fs_train, 'val': fs_val}, f)
        self._stats = None  # trigger reload of property

    @property
    def stats(self):
        if self._stats is None:
            if not self.stats_file.exists():
                raise BadProcess(
                    'flattener stats have not been generated')
            with open(self.stats_file, 'rt') as f:
                self._stats = json.load(f)
        # Fixme: this recreates an object at each call, can we just save the
        #  object?
        return TrainValProperty(self._stats.get('train', {}),
                                self._stats.get('val', {}))

    @property
    def flattener(self):
        if self._flattener is None:
            if self.flattener_file.is_file():
                f = SegmentationFlattening()
                f.load(self.flattener_file)
                self._flattener = f
            else:
                raise BadProcess('The "flattener" has not been assigned yet')
        return self._flattener

    @flattener.setter
    def flattener(self, f):
        if isinstance(f, str):
            flattener_file = find_file(f, self.save_dir, 'flattener')
            f = SegmentationFlattening()
            f.load(flattener_file)
        if not isinstance(f, SegmentationFlattening):
            raise BadType(
                '"flattener" must be of type "baby.preprocessing.SegmentationFlattening"'
            )
        f.save(self.flattener_file)
        self._flattener = f

    def fit(self, nbins=30, min_size=10, pad_frac=0.03, bud_max=200):
        if pad_frac > 0.25 or pad_frac < 0:
            raise BadParam('"pad_frac" must be between 0 and 0.2')

        # Find the best split
        overlapping = self.stats.train.get('overlap_sizes', [])
        erosion_sizes = self.stats.train.get('erosion_sizes', [])
        if len(overlapping) == 0 or len(erosion_sizes) == 0 or \
                len(list(zip(*erosion_sizes))) != len(overlapping):
            raise BadProcess(
                '"flattener_stats.json" file appears to be corrupted')

        o_noerode = overlapping[0]
        x, y, _ = zip(*o_noerode)
        max_size = max(x + y)
        pad = max([pad_frac * max_size, min_size])
        edges = np.linspace(pad, max_size - pad, nbins)[1:-1]

        o_maxerode = _best_overlapping(overlapping, erosion_sizes, min_size)
        split0, w0 = _find_best_fgroup_split(o_maxerode, edges, pad=pad)

        ogL, ogH = zip(*[_group_overlapping(o, split0, pad=pad) for o in
                         overlapping])
        szgL, szgH = _group_sizes(erosion_sizes, split0, pad=pad)

        ogL = _best_overlapping(ogL, szgL, min_size)
        ogH = _best_overlapping(ogH, szgH, min_size)

        w_ogL = sum([w for _, _, w in ogL])
        w_ogH = sum([w for _, _, w in ogH])
        if w_ogL == w_ogH:
            w_ogL, w_ogH = w0

        if w_ogL > w_ogH:
            edges = np.linspace(pad, split0 - pad, nbins)
            og = ogL
        else:
            edges = np.linspace(split0 + pad, max_size, nbins)
            og = ogH

        split1, _ = _find_best_fgroup_split(og, edges, pad=pad)

        splits = list(sorted([split0, split1]))

        szg0, szg12 = _group_sizes(erosion_sizes, splits[0], pad=pad)
        szg1, szg2 = _group_sizes(szg12, splits[1], pad=pad)

        ne0 = _best_nerode(szg0, min_size)
        ne1 = _best_nerode(szg1, min_size)
        ne2 = _best_nerode(szg2, min_size)

        flattener = SegmentationFlattening()

        flattener.addGroup('small', upper=int(np.round(splits[0] + pad)))
        flattener.addGroup('medium', lower=int(np.round(splits[0] - pad)),
                           upper=int(np.round(splits[1] + pad)))
        flattener.addGroup('large', lower=int(np.round(splits[1] - pad)))
        flattener.addGroup('buds', upper=bud_max, budonly=True)

        flattener.addTarget('lge_inte', 'large', 'interior', nerode=ne2)
        flattener.addTarget('lge_edge', 'large', 'edge')
        flattener.addTarget('mid_inte', 'medium', 'interior', nerode=ne1)
        flattener.addTarget('mid_edge', 'medium', 'edge')
        flattener.addTarget('sml_inte', 'small', 'filled', nerode=ne0)
        flattener.addTarget('sml_edge', 'small', 'edge')
        flattener.addTarget('bud_neck', 'buds', 'budneck')

        flattener.save(self.flattener_file)
        self._flattener = None

    def plot_stats(self, nbins=30):
        overlapping = self.stats.train.get('overlap_sizes', [])
        max_erode = len(overlapping)
        fig, axs = plt.subplots(1, max_erode, figsize=(16, 16 / max_erode))
        x, y, _ = zip(*overlapping[0])
        max_size = max(x + y)
        for ax, (e, os) in zip(axs, enumerate(overlapping)):
            if len(os) > 0:
                x, y, w = zip(*os)
            else:
                x, y, w = 3 * [[]]
            ax.hist2d(x, y, bins=nbins, weights=w,
                      range=[[0, max_size], [0, max_size]])
            ax.plot((0, max_size), (0, max_size), 'r')
            ax.set_title('nerosions = {:d}'.format(e))
        fig.savefig(self.save_dir / 'flattener_stats.png')


def instantiate_tuner(model, method='random', **kwargs):
    method = method.lower()
    if method == 'random':
        return RandomSearch(model, **kwargs, project_name=model.name)
    elif method == 'hyperband':
        return Hyperband(model, **kwargs, project_name=model.name)
    elif method == 'bayesian':
        return BayesianOptimization(model, **kwargs, project_name=model.name)
    else:
        raise (ValueError, 'Method {} is not supported.'.format(method))


class HyperParameterTrainer:
    """
    Class that chooses the best hyperparameters for a specific model-type.

    Note: uses Keras-tuner Hypermodels -- requires tensorflow 2

    Outputs: a set of parameters for that form of model, into a file.
    If using tensorflow 1: these parameters need to be set by the user by
    default.
    """

    # Todo: where do we stop tweaking?
    #  - Model parameters
    #  - Augmentation choices?
    #  - Optimizer and learning rate?
    def __init__(self, save_dir: Path, cnn_set, gen, aug, outputs,
                 tuner_params: Union[Tuner, None, dict, str] = None):
        self.save_dir = save_dir
        self.aug = aug
        self.gen = gen
        self.outputs = outputs
        self._init_hypermodels(cnn_set)
        self._tuners = dict()

        self._cnn = None
        self.cnn = self.cnn_set[0]

        if tuner_params is None:
            self._tuner_params = dict(method='random',
                                      objective='val_loss',
                                      max_trials=3,
                                      directory='./',
                                      overwrite=True)
        elif isinstance(tuner_params, str):
            with open(tuner_params, 'r') as fd:
                self._tuner_params = json.load(fd)
        elif isinstance(tuner_params, dict):
            # Todo: save tuner to file
            self._tuner_params = tuner_params
        self._best_parameters = None

    def _init_hypermodels(self, cnn_set):
        self.cnn_set = [None] * len(cnn_set)
        for i, cnn in enumerate(cnn_set):
            if isinstance(cnn, str):
                # Get from Hypermodel file by name
                shapes = self.gen.train.shapes
                self.cnn_set[i] = get_hypermodel(cnn, shapes.input[1:],
                                                 self.outputs)
            elif not isinstance(cnn, HyperModel):
                raise TypeError("Non-standard CNNs must be in the form of a "
                                "keras.Hypermodel, "
                                "received {}".format(type(cnn)))

    @property
    def cnn(self):
        return self._cnn

    @cnn.setter
    def cnn(self, cnn):
        if isinstance(cnn, HyperModel):
            self._cnn = cnn
        else:
            raise TypeError("CNN instance must be of type "
                            "kerastuner.Hypermodel.")

    @property
    def cnn_dir(self):
        cnn_dir = self.save_dir / self.cnn.name
        if not cnn_dir.exists():
            cnn_dir.mkdir()
        return cnn_dir

    @property
    def tuner(self):
        if self.cnn.name not in self._tuners:
            self._tuners[self.cnn.name] = instantiate_tuner(self.cnn,
                                                            **self._tuner_params)
        return self._tuners[self.cnn.name]

    # Todo: set tuner parameters, not tuner itself
    @tuner.setter
    def tuner(self, tuner):
        if isinstance(tuner, str):
            with open(tuner, 'r') as fd:
                params = json.load(fd)
            self._tuner_params = params
        elif isinstance(tuner, dict):
            self._tuner_params = tuner
        # Invalidate current tuners
        self._tuners = dict()

    @property
    def best_parameters(self):
        if self._best_parameters is None:
            self._best_parameters = self.tuner.get_best_hyperparameters(
            )[0].values
        return self._best_parameters

    def save_best_parameters(self, filename):
        with open(filename, 'w') as fd:
            json.dump(self.best_parameters, fd)

    def search(self, epochs=100, steps_per_epoch=10, validation_steps=10,
               **kwargs):
        """
        Runs search with the instance's generator and tuner.

        Keyword arguments are those you would normally use in a `model.fit` call.
        For instance:
        ```python
        tuner.search(generator,
                     steps_per_epoch=train_steps,
                     epochs=args.nb_epochs,
                     callbacks=[early_stopping, checkpointer, tensor_board],
                     validation_data=val_generator,
                     validation_steps=val_steps,
                     verbose=1,
                     workers=args.nb_workers,
                     class_weight=class_weight)
        ```
        :param kwargs:
        :return:
        """
        with augmented_generator(self.gen.train, self.aug.train) as train_gen:
            with augmented_generator(self.gen.val, self.aug.val) as val_gen:
                self.tuner.search(train_gen,
                                  steps_per_epoch=steps_per_epoch,
                                  epochs=epochs,
                                  validation_data=val_gen,
                                  validation_steps=validation_steps,
                                  **kwargs)
        # Get best parameters
        self.save_best_parameters(self.cnn_dir / 'hyperparameters.json')


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


class BabyTrainer(object):
    """Manager to set up and train BABY models
    
    :param save_dir: directory in which to save parameters and logs (and
        from which to auto-load parameters and logs)
    :param train_val_images: either a dict with keys 'training' and
        'validation' and values specifying lists of file name pairs, or the
        name of a json file containing such a dict. The file name pairs
        should correspond to image-label pairs suitable for input to
        `baby.generator.ImageLabel`.
    :param flattener: either a `baby.preprocessing.SegmentationFlattening`
        object, or the name of a json file that is a saved
        `SegmentationFlattening` object.
    """

    def __init__(self, save_dir, base_dir=None, params=None, max_cnns=3):

        # Register the save dir
        if base_dir is not None:
            base_dir = Path(base_dir)
            if not base_dir.is_dir():
                raise BadParam('"base_dir" must be a valid directory or None')
        else:
            base_dir = Path.cwd()
        self.base_dir = base_dir
        # All other directories are relative to the base dir
        save_dir = Path(save_dir)
        if not save_dir.is_absolute():
            save_dir = base_dir / save_dir
        if not save_dir.is_dir():
            raise BadParam('"save_dir" must be a valid directory')
        self.save_dir = save_dir
        self._parameters_file = 'parameters.json'

        # Register parameters if specified
        if isinstance(params, BabyTrainerParameters):
            self.parameters = params
        elif isinstance(params, str):
            filename = find_file(params, save_dir, 'params')
            savename = save_dir / self._parameters_file
            if not savename.is_file():
                shutil.copy(filename, savename)

        self._max_cnns = max_cnns
        # Trainers
        self._smoothing_sigma_trainer = None
        self._flattener_trainer = None
        self._hyperparameter_trainer = None
        self._cnn_trainer = None
        self._track_trainer = None
        self._bud_trainer = None

    @property
    def smoothing_sigma_trainer(self):
        if self._smoothing_sigma_trainer is None:
            self._smoothing_sigma_trainer = SmoothingModelTrainer(
                save_dir=self.save_dir,
                stats_file=self.parameters.smoothing_sigma_stats_file,
                model_file=self.parameters.smoothing_sigma_model_file
            )
        return self._smoothing_sigma_trainer

    @property
    def flattener_trainer(self):
        if self._flattener_trainer is None:
            self._flattener_trainer = FlattenerTrainer(
                save_dir=self.save_dir,
                stats_file=self.parameters.flattener_stats_file,
                flattener_file=self.parameters.flattener_file
            )
        return self._flattener_trainer

    @property
    def hyperparameter_trainer(self):
        if self._hyperparameter_trainer is None:
            self._hyperparameter_trainer = HyperParameterTrainer(
                save_dir=self.save_dir,
                cnn_set=self.parameters.cnn_set,
                gen=self.gen,
                aug=self.aug,
                outputs=self.flattener_trainer.flattener.names(),
                tuner_params=None
                # Todo: tuner file if it exists in parameters
            )
        return self._hyperparameter_trainer

    @property
    def cnn_trainer(self):
        if self._cnn_trainer is None:
            self._cnn_trainer = CNNTrainer(
                save_dir=self.save_dir,
                cnn_set=self.parameters.cnn_set,
                gen=self.gen,
                aug=self.aug,
                flattener=self.flattener_trainer.flattener,
                cnn_fn=self.parameters.cnn_fn,
                max_cnns=self._max_cnns  # Todo: private access OK?
            )
        return self._cnn_trainer

    @property
    def track_trainer(self):
        if self._track_trainer is None:
            self._track_trainer = TrackTrainer(self.data._metadata, self.data)
        return self._track_trainer

    @property
    def bud_trainer(self):
        if self._bud_trainer is None:
            self._bud_trainer = BudTrainer(self.data._metadata, self.data)
        return self._bud_trainer

    @property
    def parameters(self):
        if not hasattr(self, '_parameters') or not self._parameters:
            param_file = self.save_dir / self._parameters_file
            if param_file.is_file():
                with open(param_file, 'rt') as f:
                    params = json.load(f, object_hook=as_python_object)
                if not isinstance(params, BabyTrainerParameters):
                    raise BadFile('Parameters file has been corrupted')
                self._parameters = params
            else:
                self.parameters = BabyTrainerParameters()
        return self._parameters

    @parameters.setter
    def parameters(self, params):
        if isinstance(params, dict):
            p = {}
            if hasattr(self, '_parameters') and self._parameters:
                p = self._parameters._asdict()
            p.update(params)
            params = BabyTrainerParameters(**p)
        elif not isinstance(params, BabyTrainerParameters):
            params = BabyTrainerParameters(*params)
        self._parameters = params
        with open(self.save_dir / self._parameters_file, 'wt') as f:
            json.dump(jsonify(self._parameters), f)

    @property
    def in_memory(self):
        return self.parameters.in_memory

    @in_memory.setter
    def in_memory(self, x):
        self.parameters = dict(in_memory=x)

    def _check_for_data_update(self):
        if getattr(self, '_ncells', None) != self._impairs.ncells:
            # Reset generators
            self._gen_train = None
            self._gen_val = None
            # Trigger save of the data
            datafile = self.save_dir / self.parameters.train_val_pairs_file
            self._impairs.save(datafile, self.base_dir)
            self._ncells = self._impairs.ncells

    @property
    def data(self):
        if not hasattr(self, '_impairs') or not self._impairs:
            self._impairs = TrainValPairs()
            pairs_file = self.save_dir / self.parameters.train_val_pairs_file
            if pairs_file.is_file():
                self._impairs.load(pairs_file, self.base_dir)
        self._check_for_data_update()
        return self._impairs

    @data.setter
    def data(self, train_val_pairs):
        if isinstance(train_val_pairs, str):
            pairs_file = find_file(train_val_pairs, self.save_dir, 'data')
            train_val_pairs = TrainValPairs()
            train_val_pairs.load(pairs_file, self.base_dir)
        if not isinstance(train_val_pairs, TrainValPairs):
            raise BadType('"data" must be of type "baby.io.TrainValPairs"')
        self._impairs = train_val_pairs
        self._check_for_data_update()

    @property
    def gen(self):
        # NB: generator init ensures all specified images exist
        # NB: only dummy augmenters are assigned to begin with
        p = self.parameters
        if not getattr(self, '_gen_train', None):
            if len(self.data.training) == 0:
                raise BadProcess('No training images have been added')

            # Initialise generator for training images
            self._gen_train = ImageLabel(self.data.training,
                                         batch_size=p.batch_size,
                                         aug=Augmenter(),
                                         preprocess=(robust_norm, seg_norm),
                                         in_memory=p.in_memory)

        if not getattr(self, '_gen_val', None):
            if len(self.data.validation) == 0:
                raise BadProcess('No validation images have been added')
            # Initialise generator for validation images
            self._gen_val = ImageLabel(self.data.validation,
                                       batch_size=p.batch_size,
                                       aug=Augmenter(),
                                       preprocess=(robust_norm, seg_norm),
                                       in_memory=p.in_memory)

        return TrainValProperty(self._gen_train, self._gen_val)

    def plot_gen_sample(self, validation=False):
        g = self.gen.val if validation else self.gen.train
        g.aug = self.aug.val if validation else self.aug.train
        img_batch, lbl_batch = g[0]
        lbl_batch = np.concatenate(lbl_batch, axis=3)

        f = self.flattener
        target_names = f.names()
        edge_inds = np.flatnonzero([t.prop == 'edge' for t in f.targets])

        ncol = len(img_batch)
        nrow = len(target_names) + 1
        fig = plt.figure(figsize=(3 * ncol, 3 * nrow))
        for b, (bf, seg) in enumerate(zip(img_batch, lbl_batch)):
            plt.subplot(nrow, ncol, b + 0 * ncol + 1)
            plt.imshow(bf[:, :, 0], cmap='gray')
            plt.imshow(colour_segstack(seg[:, :, edge_inds]))

            for i, name in enumerate(target_names):
                plt.subplot(nrow, ncol, b + (i + 1) * ncol + 1)
                plt.imshow(seg[:, :, i], cmap='gray')
                plt.title(name)

        fig.savefig(self.save_dir / '{}_generator_sample.png'.format(
            'validation' if validation else 'training'))

    def generate_smoothing_sigma_stats(self):
        train_gen = augmented_generator(self.gen.train, lambda x, y: (x, y))
        val_gen = augmented_generator(self.gen.train, lambda x, y: (x, y))
        self.smoothing_sigma_trainer.generate_smoothing_sigma_stats(
            train_gen, val_gen)

    @property
    def smoothing_sigma_stats(self):
        warnings.warn("nursery.smoothing_sigma_stats will soon be "
                      "deprecated, use nursery.smoothing_sigma_trainer.stats "
                      "instead",
                      DeprecationWarning)
        return self.smoothing_sigma_trainer.stats

    @property
    def smoothing_sigma_model(self):
        return self.smoothing_sigma_trainer.model

    def generate_flattener_stats(self, max_erode=5):
        warnings.warn("nursery.generate_flattener_stats will soon be "
                      "deprecated, use "
                      "nursery.flattener_trainer.generate_stats(train_gen, "
                      "val_gen, train_aug, val_aug, max_erode=5) instead",
                      DeprecationWarning)
        self.flattener_trainer.generate_flattener_stats(*self.gen, *self.aug,
                                                        max_erode=max_erode)

    @property
    def flattener_stats(self):
        warnings.warn("nursery.flattener_stats will soon be "
                      "deprecated, use nursery.flattener_trainer.stats "
                      "instead",
                      DeprecationWarning)
        return self.flattener_trainer.stats

    @property
    def flattener(self):
        warnings.warn("nursery.flattener will soon be "
                      "deprecated, use nursery.flattener_trainer.flattener "
                      "instead",
                      DeprecationWarning)
        return self.flattener_trainer.flattener

    @flattener.setter
    def flattener(self, f):
        warnings.warn("nursery.flattener will soon be "
                      "deprecated, use nursery.flattener_trainer.flattener "
                      "instead",
                      DeprecationWarning)
        self.flattener_trainer.flattener = f

    @property
    def aug(self):
        p = self.parameters
        t = ScalingAugmenter(self.smoothing_sigma_model,
                             self.flattener,
                             xy_out=p.xy_out,
                             target_pixel_size=p.target_pixel_size,
                             substacks=p.substacks,
                             probs={
                                 'rotate': 0.2,
                                 'vshift': 0.25,
                                 'hshift': 0.25
                             })
        v = ScalingAugmenter(self.smoothing_sigma_model,
                             self.flattener,
                             xy_out=p.xy_out,
                             target_pixel_size=p.target_pixel_size,
                             substacks=p.substacks,
                             p_noop=1,
                             probs={
                                 'vshift': 0.25,
                                 'hshift': 0.25
                             })
        return TrainValProperty(t, v)

    @property
    def cnn_fn(self):
        warnings.warn("nursery.cnn_fn will soon be "
                      "deprecated, use nursery.cnn_trainer.cnn_fn "
                      "instead",
                      DeprecationWarning)
        return self.cnn_trainer.cnn_fn

    @cnn_fn.setter
    def cnn_fn(self, fn):
        warnings.warn("nursery.cnn_fn will soon be "
                      "deprecated, use nursery.cnn_trainer.cnn_fn "
                      "instead",
                      DeprecationWarning)
        self.cnn_trainer.cnn_fn = fn

    @property
    def cnn_dir(self):
        warnings.warn("nursery.cnn_dir will soon be "
                      "deprecated, use nursery.cnn_trainer.cnn_dir "
                      "instead",
                      DeprecationWarning)
        return self.cnn_trainer.cnn_dir

    @property
    def cnn_name(self):
        warnings.warn("nursery.cnn_bane will soon be "
                      "deprecated, use nursery.cnn_trainer.cnn_name "
                      "instead",
                      DeprecationWarning)
        return self.cnn_trainer.cnn_name

    @property
    def cnn(self):
        warnings.warn("nursery.cnn will soon be "
                      "deprecated, use nursery.cnn_trainer.cnn "
                      "instead",
                      DeprecationWarning)
        return self.cnn_trainer.cnn

    @property
    def histories(self):
        warnings.warn("nursery.histories will soon be "
                      "deprecated, use nursery.cnn_trainer.histories "
                      "instead",
                      DeprecationWarning)
        return self.cnn_trainer.histories

    @property
    def cnn_opt_dir(self):
        warnings.warn("nursery.opt_dir will soon be "
                      "deprecated, use nursery.cnn_trainer.opt_dir "
                      "instead",
                      DeprecationWarning)
        return self.cnn_trainer.opt_dir

    @property
    def cnn_opt(self):
        warnings.warn("nursery.cnn_opt will soon be "
                      "deprecated, use nursery.cnn_trainer.opt_cnn "
                      "instead",
                      DeprecationWarning)
        return self.cnn_trainer.opt_cnn

    def fit_smoothing_model(self, filt='identity'):
        warnings.warn("nursery.fit_smoothing_model will soon be "
                      "deprecated, use nursery.smoothing_signa_trainer.fit "
                      "instead",
                      DeprecationWarning)
        self.smoothing_sigma_trainer.fit(filt=filt)

    def plot_fitted_smoothing_sigma_model(self):
        warnings.warn("nursery.plot_fitted_smoothing_sigma_model will soon be "
                      "deprecated, use "
                      "nursery.smoothing_signa_trainer.plot_fitted_model "
                      "instead",
                      DeprecationWarning)
        self.smoothing_sigma_trainer.plot_fitted_model()

    def fit_flattener(self, **kwargs):
        warnings.warn("nursery.fit_flattener will soon be "
                      "deprecated, use nursery.flattener_trainer.fit "
                      "instead",
                      DeprecationWarning)
        self.flattener_trainer.fit(**kwargs)

    def plot_flattener_stats(self, **kwargs):
        warnings.warn("nursery.plot_flattener_stats will soon be "
                      "deprecated, use nursery.flattener_trainer.plot_stats "
                      "instead",
                      DeprecationWarning)
        self.flattener_trainer.plot_stats(**kwargs)

    def fit_cnn(self, **kwargs):
        warnings.warn("nursery.fit_cnn will soon be "
                      "deprecated, use nursery.cnn_trainer.fit "
                      "instead",
                      DeprecationWarning)
        self.cnn_trainer.fit(**kwargs)

    def plot_histories(self, **kwargs):
        warnings.warn("nursery.plot_histories will soon be "
                      "deprecated, use nursery.cnn_trainer.plot_histories "
                      "instead",
                      DeprecationWarning)
        self.cnn_trainer.plot_histories(**kwargs)

    # Todo: move to SegmentationParamTrainer
    # Todo fill
    def fit_seg_params(self):
        pass


class Nursery(BabyTrainer):
    pass


def load_history(subdir):
    with open(LOG_DIR / subdir / 'history.pkl', 'rb') as f:
        return pickle.load(f)


def get_best_and_worst(model, gen):
    best = {}
    worst = {}
    for i in range(len(gen)):
        inputim, target = gen[i]
        pred = model.predict(inputim)
        for l, output in enumerate(model.output_names):
            if output not in best:
                best[output] = []
            if output not in worst:
                worst[output] = []
            target_bin = target[l][..., 0] > 0
            pred_bin = np.squeeze(pred[l]) > 0.5
            for p, t, im in zip(pred_bin, target_bin, inputim):
                u = np.sum(p | t)
                if u == 0:
                    # Ignore any empty traps
                    continue
                iou = np.sum(p & t) / u
                out = (iou, i, p, t, im)
                if len(best[output]) < 8:
                    best[output].append(out)
                    worst[output].append(out)
                else:
                    best_IoUs = [b[0] for b in best[output]]
                    best_minind = np.argmin(best_IoUs)
                    if iou > best_IoUs[best_minind]:
                        best[output][best_minind] = out
                    worst_IoUs = [w[0] for w in worst[output]]
                    worst_maxind = np.argmax(worst_IoUs)
                    if iou < worst_IoUs[worst_maxind]:
                        worst[output][worst_maxind] = out

    return best, worst


def _generate_smoothing_sigma_stats(gen):
    sigmas = np.arange(0.4, 5.0, 0.20)
    rotations = np.arange(7, 45, 7)
    scaling = np.linspace(0.5, 1.5, 6)

    square_edge = lambda m: binary_edge(m, squareconn)
    smoothing_stats = []
    for t in trange(len(gen.paths)):
        _, (segs, _) = gen.get_by_index(t)

        if segs.shape[2] == 1 and segs.sum() == 0:
            continue

        ncell = segs.shape[2]
        segs_fill = binary_fill_holes(segs, dwsquareconn)
        segs_edge = binary_edge(segs, dwsquareconn)
        for c in range(ncell):
            sfill = segs_fill[..., c]
            sedge = segs_edge[..., c]
            nedge = segs[..., c].sum()

            # fit radial spline to generate accurate reference edges for
            # resize transformation
            rprops = regionprops(sfill.astype('int'))[0]
            centre = np.array(rprops.centroid)
            radii, angles = morph_radial_thresh_fit(sedge, sfill, rprops)
            genedge = draw_radial(radii, angles, centre, sedge.shape)
            genfill = binary_fill_holes(genedge, squareconn)

            # Limit the number of rotation and scaling operations by
            # randomly choosing one per cell:
            r = np.random.choice(rotations)
            z = np.random.choice(scaling)
            for s in sigmas:
                # Use gaussian blurred filled image for augmentations
                sblur = filters.gaussian(sfill, s)
                genblur = filters.gaussian(genfill, s)

                spf = _filled_canny(sblur)
                smoothing_stats += [{
                    'ind': t,
                    'cell': c,
                    'sigma': s,
                    'rotation': 0,
                    'scaling': 1,
                    'nedge': nedge,
                    'iou': mask_iou(spf, sfill),
                    'edge_iou': mask_iou(square_edge(spf), sedge)
                }]

                sr = transform.rotate(sblur,
                                      angle=r,
                                      mode='reflect',
                                      resize=True)
                sr = transform.rotate(sr,
                                      angle=-r,
                                      mode='reflect',
                                      resize=False)
                srf = _filled_canny(_apply_crop(sr, spf.shape))
                smoothing_stats += [{
                    'ind': t,
                    'cell': c,
                    'sigma': s,
                    'rotation': r,
                    'scaling': 1,
                    'nedge': nedge,
                    'iou': mask_iou(srf, sfill),
                    'edge_iou': mask_iou(square_edge(srf), sedge)
                }]

                insize = np.array(spf.shape)
                outsize = np.round(insize * z).astype('int')
                centre_sc = outsize / 2 + z * (centre - insize / 2)
                genedge_sc = draw_radial(z * radii, angles, centre_sc,
                                         outsize)
                genfill_sc = binary_fill_holes(genedge_sc, squareconn)
                sd = transform.resize(genblur, outsize, anti_aliasing=False)
                sdf = _filled_canny(sd)
                smoothing_stats += [{
                    'ind': t,
                    'cell': c,
                    'sigma': s,
                    'rotation': 0,
                    'scaling': z,
                    'nedge': nedge,
                    'iou': mask_iou(sdf, genfill_sc),
                    'edge_iou': mask_iou(square_edge(sdf), genedge_sc)
                }]

    return pd.DataFrame(smoothing_stats)


def _generate_flattener_stats(gen, max_erode):
    nerode = list(range(max_erode + 1))
    overlap_sizes = [[] for e in nerode]
    erosion_sizes = []

    for t in range(len(gen.paths)):
        _, segs = gen.get_by_index(t)
        nsegs = segs.shape[2]
        segs = segs > 0
        s_sizes = [int(segs[..., s].sum()) for s in range(nsegs)]
        esizes = [[] for s in range(nsegs)]
        for e in nerode:
            for s0 in range(nsegs):
                seg0 = segs[..., s0]
                n0 = int(seg0.sum())
                esizes[s0].append(n0)
                if n0 == 0:
                    continue
                for s1 in range(s0 + 1, nsegs):
                    seg1 = segs[..., s1]
                    o = float(np.sum(seg0 & seg1) / np.sum(seg0 | seg1))
                    if o > 0:
                        sizes = tuple(sorted([s_sizes[s0], s_sizes[s1]]))
                        overlap_sizes[e].append(sizes + (o,))
            segs = binary_erosion(segs, dwsquareconn)
        erosion_sizes.extend(esizes)

    return {'overlap_sizes': overlap_sizes, 'erosion_sizes': erosion_sizes}


def _group_sizes(es, thresh, pad=0):
    return ([s for s in es if s[0] < thresh + pad],
            [s for s in es if s[0] >= thresh - pad])


def _group_overlapping(os, thresh, pad=0):
    return ([
                (x, y, w) for x, y, w in os if
                x < thresh + pad and y < thresh + pad
            ], [(x, y, w) for x, y, w in os if
                x >= thresh - pad and y >= thresh - pad
                ])


def _best_overlapping(overlapping, erosion_sizes, min_size):
    sz_erosions = list(zip(*erosion_sizes))
    e_invalid = [any([c < min_size for c in e]) for e in
                 sz_erosions[:0:-1]]
    o_valid = [o for o, e in zip(overlapping[:0:-1], e_invalid) if not e]
    o_valid += [overlapping[0]]
    return o_valid[0]


def _sum_group_overlapping(os, thresh, pad=0):
    return tuple(
        sum([w
             for _, _, w in og])
        for og in _group_overlapping(os, thresh, pad=pad))


def _find_best_fgroup_split(os, edges, pad=0):
    overlaps = [
        _sum_group_overlapping(os, thresh, pad=pad) for thresh in edges
    ]
    return min(zip(edges, overlaps), key=lambda x: sum(x[1]))


def _best_nerode(szg, min_size):
    ne = [
        n for n, e in list(enumerate(zip(*szg)))[:0:-1]
        if not any([c < min_size for c in e])
    ]
    return ne[0] if len(ne) > 0 else 0
