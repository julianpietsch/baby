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
import inspect
import json
import pickle
import warnings
from itertools import product, chain
from pathlib import Path
from typing import NamedTuple

import numpy as np
import pandas as pd
import tensorflow as tf
from matplotlib import pyplot as plt

from baby.tracker.training import CellTrainer
from baby.training import CNNTrainer
from baby.errors import BadParam, BadFile, BadType, BadProcess

if tf.__version__.startswith('2'):
    # tf.compat.v1.disable_eager_execution()
    from .hyper_parameter_trainer import HyperParameterTrainer
else:
    from .v1_hyper_parameter_trainer import HyperParamV1 as \
            HyperParameterTrainer

from .utils import (BabyTrainerParameters, TrainValTestProperty,
                    standard_augmenter, SharedParameterContainer,
                    SharedDataContainer)
from .smoothing_model_trainer import SmoothingModelTrainer
from .flattener_trainer import FlattenerTrainer
from .segmentation_trainer import SegmentationTrainer, SegExample
from .bud_trainer import BudTrainer


class BabyTrainer(object):
    """Manager to set up and train BABY models.

    Args:
        save_dir (str or Path): Directory in which to save parameters and logs
            (and from which to auto-load parameters and logs). Must be an
            absolute path, or specified relative to ``base_dir``.
        base_dir (str or Path or None): Base directory within which all
            relevant image files can be found. By default (i.e., specifying
            ``None``), uses the current working directory. References to the
            image files will be saved relative to this directory. If the base
            directory is copied/moved but the structure within that directory
            is left intact, then references to the image files will be
            correctly maintained.
        params (None or BabyTrainerParameters or str or Path): Optionally
            specify initial training parameters as a
            :py:class:`BabyTrainerParameters` instance or the path to a saved
            :py:class:`BabyTrainerParameters` instance.
    """

    def __init__(self, save_dir, base_dir=None, params=None, max_cnns=3):

        if base_dir is not None:
            base_dir = Path(base_dir)
            if not base_dir.is_dir():
                raise BadParam('"base_dir" must be a valid directory or None')
        else:
            base_dir = Path.cwd()
        self.base_dir = base_dir

        # The save directory may be specified relative to the base directory
        save_dir = Path(save_dir)
        if not save_dir.is_absolute():
            save_dir = base_dir / save_dir
        if not save_dir.is_dir():
            raise BadParam('"save_dir" must be a valid directory')

        # Initialise shared parameters and data
        self._shared_params = SharedParameterContainer(
            save_dir, params=params)
        self._shared_data = SharedDataContainer(
            self._shared_params, base_dir=base_dir)

        self._max_cnns = max_cnns

        # Trainers
        self._smoothing_sigma_trainer = None
        self._flattener_trainer = None
        self._hyperparameter_trainer = None
        self._cnn_trainer = None
        self._track_trainer = None
        self._bud_trainer = None
        self._segmentation_trainer = None

    @property
    def parameters(self):
        return self._shared_params.parameters

    @parameters.setter
    def parameters(self, params):
        self._shared_params.parameters = params

    @property
    def segmentation_parameters(self):
        return self._shared_params.segmentation_parameters

    @segmentation_parameters.setter
    def segmentation_parameters(self, params):
        self._shared_params.segmentation_parameters = params

    @property
    def save_dir(self):
        return self._shared_params.save_dir

    @property
    def in_memory(self):
        return self.parameters.in_memory

    @in_memory.setter
    def in_memory(self, x):
        self.parameters = dict(in_memory=x)

    @property
    def data(self):
        return self._shared_data.data

    @data.setter
    def data(self, x):
        self._shared_data.data = x

    @property
    def tracker_data(self):
        return self._shared_data.tracker_data

    @tracker_data.setter
    def tracker_data(self, x):
        self._shared_data.tracker_data = x

    @property
    def gen(self):
        return self.flattener_trainer.default_gen

    def gen_with_aug(self, aug):
        return self._shared_data.gen_with_aug(aug)

    @property
    def smoothing_sigma_trainer(self):
        if self._smoothing_sigma_trainer is None:
            self._smoothing_sigma_trainer = SmoothingModelTrainer(
                self._shared_params, self._shared_data)
        return self._smoothing_sigma_trainer

    @property
    def flattener_trainer(self):
        if self._flattener_trainer is None:
            self._flattener_trainer = FlattenerTrainer(
                self._shared_params, self._shared_data,
                self.smoothing_sigma_trainer)
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
                    self._shared_params,
                    self.flattener_trainer,
                    max_cnns=self._max_cnns  # Todo: private access OK?
                    )
        return self._cnn_trainer

    @property
    def segmentation_trainer(self):
        if self._segmentation_trainer is None:
            self._segmentation_trainer = SegmentationTrainer(
                self._shared_params, self._shared_data, 
                self.smoothing_sigma_trainer, self.cnn_trainer)
        return self._segmentation_trainer

    @property
    def track_trainer(self):
        if self._track_trainer is None:
            self._track_trainer = CellTrainer(self.tracker_data._metadata,
                    self.tracker_data)
        return self._track_trainer

    @track_trainer.setter
    def track_trainer(self, all_feats2use):
        self._track_trainer = CellTrainer(self.tracker_data._metadata,
                data=self.tracker_data,
                all_feats2use=all_feats2use)

    @property
    def bud_trainer(self):
        if self._bud_trainer is None:
            self._bud_trainer = BudTrainer(
                self._shared_params, self._shared_data,
                self.segmentation_trainer)
        return self._bud_trainer

    def fit_smoothing_model(self, filt='identity'):
        try:
            self.smoothing_sigma_trainer.stats
        except BadProcess:
            self.smoothing_sigma_trainer.generate_smoothing_sigma_stats()
        self.smoothing_sigma_trainer.fit(filt=filt)

    def plot_fitted_smoothing_model(self):
        warnings.warn(
                "nursery.plot_fitted_smoothing_sigma_model will soon be "
                "deprecated, use "
                "nursery.smoothing_signa_trainer.plot_fitted_model "
                "instead", DeprecationWarning)
        self.smoothing_sigma_trainer.plot_fitted_model()

    @property
    def smoothing_sigma_model(self):
        return self.smoothing_sigma_trainer.model

    @smoothing_sigma_model.setter
    def smoothing_sigma_model(self, m):
        self.smoothing_sigma_trainer.model = m

    def fit_flattener(self, max_erode=5, **kwargs):
        try:
            self.flattener_trainer.stats
        except BadProcess:
            self.flattener_trainer.generate_flattener_stats(
                max_erode=max_erode)
        self.flattener_trainer.fit(**kwargs)

    def plot_flattener_stats(self, **kwargs):
        self.flattener_trainer.plot_stats(**kwargs)

    @property
    def flattener(self):
        return self.flattener_trainer.flattener

    @flattener.setter
    def flattener(self, f):
        self.flattener_trainer.flattener = f

    def plot_generator_sample(self, **kwargs):
        self.flattener_trainer.plot_default_gen_sample(**kwargs)

    def fit_cnn(self, **kwargs):
        self.cnn_trainer.fit(**kwargs)

    def plot_cnn_histories(self, **kwargs):
        self.cnn_trainer.plot_histories(**kwargs)

    @property
    def cnn_dir(self):
        """Directory containing saved weights for the optimised CNN."""
        return self.cnn_trainer.opt_dir

    @property
    def cnn(self):
        """Optimised CNN from the ``cnn_trainer``."""
        return self.cnn_trainer.opt_cnn

class Nursery(BabyTrainer):
    pass


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
