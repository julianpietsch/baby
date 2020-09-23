import json
import pickle
import shutil
import warnings
from pathlib import Path

import numpy as np
import tensorflow as tf

from matplotlib import pyplot as plt

if tf.__version__.startswith('2'):
    tf.compat.v1.disable_eager_execution()
    from .hyper_parameter_trainer import HyperParameterTrainer
else:
    from .v1_hyper_parameter_trainer import HyperParamV1 as \
        HyperParameterTrainer

from .utils import BabyTrainerParameters, TrainValProperty, \
    augmented_generator
from .smoothing_model_trainer import SmoothingModelTrainer
from .flattener_trainer import FlattenerTrainer
from .cnn_trainer import CNNTrainer
from .track_trainer import TrackTrainer
from .bud_trainer import BudTrainer

from baby.utils import find_file, as_python_object, jsonify
from baby.errors import BadParam, BadFile, BadType, BadProcess
from baby.io import TrainValPairs
from baby.preprocessing import (robust_norm, seg_norm)
from baby.augmentation import (Augmenter, ScalingAugmenter)
from baby.generator import ImageLabel
from baby.visualise import colour_segstack




LOG_DIR = 'logs'


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







