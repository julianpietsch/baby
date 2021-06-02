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
import inspect
import json
import pickle
import shutil
import warnings
from itertools import product, chain
from pathlib import Path
from typing import NamedTuple

import numpy as np
import pandas as pd
import tensorflow as tf
from baby.brain import default_params
from baby.morph_thresh_seg import MorphSegGrouped
from baby.performance import calc_IoUs, best_IoU, calc_AP
from baby.seg_trainer import SegFilterParamOptim, _sub_params
from baby.tracker.training import CellTrainer, BudTrainer
from baby.training import CNNTrainer

from matplotlib import pyplot as plt
from tqdm import tqdm

if tf.__version__.startswith('2'):
    tf.compat.v1.disable_eager_execution()
    from .hyper_parameter_trainer import HyperParameterTrainer
else:
    from .v1_hyper_parameter_trainer import HyperParamV1 as \
        HyperParameterTrainer

from .utils import BabyTrainerParameters, TrainValTestProperty, \
    augmented_generator
from .smoothing_model_trainer import SmoothingModelTrainer
from .flattener_trainer import FlattenerTrainer

from baby.utils import find_file, as_python_object, jsonify, batch_iterator, \
    split_batch_pred
from baby.errors import BadParam, BadFile, BadType, BadProcess
from baby.io import TrainValTestPairs
from baby.preprocessing import (robust_norm, seg_norm)
from baby.augmentation import (Augmenter, ScalingAugmenter)
from baby.generator import ImageLabel
from baby.visualise import colour_segstack

LOG_DIR = 'logs'


# TODO: ADD TO UTILS in training
class SegExample(NamedTuple):
    """CNN output paired with target segmented outlines and info

    Used for optimising segmentation hyperparameters and for training bud
    assigment models
    """
    pred: np.ndarray
    target: np.ndarray
    info: dict
    img: np.ndarray


# interior_threshold: threshold on predicted interior
# nclosing: number of closing operations on threshold mask
# nopening: number of opening operations on threshold mask
#
# TODO Create SegTrainer
# TODO Add default parameters to SegTrainer
#   Structure taht the parameters should actually be (shape/size)
#   Thresholds modified to the correct parameters
base_seg_params = {
    'interior_threshold': [0.5, 0.5, 0.5],
    'nclosing': [0, 0, 0],
    'nopening': [0, 0, 0],
    'connectivity': [2, 2, 2],
    'edge_sub_dilations': [0, 0, 0]
}

# Todo add default parameters to SegTrainer
#       Search space !
seg_param_coords = {
    'nclosing': [0, 1, 2],
    'nopening': [0, 1, 2],
    'interior_threshold': np.arange(0.3, 1.0, 0.05).tolist(),
    'connectivity': [1, 2],
    'edge_sub_dilations': [0, 1, 2]
}


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
                model_file=self.parameters.smoothing_sigma_model_file)
        return self._smoothing_sigma_trainer

    @property
    def flattener_trainer(self):
        if self._flattener_trainer is None:
            self._flattener_trainer = FlattenerTrainer(
                save_dir=self.save_dir,
                stats_file=self.parameters.flattener_stats_file,
                flattener_file=self.parameters.flattener_file)
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
            self._track_trainer = CellTrainer(self.tracker_data._metadata,
                                              self.tracker_data)
        return self._track_trainer

    @track_trainer.setter
    def track_trainer(self, all_feats2use=None):
        self._track_trainer = CellTrainer(self.tracker_data._metadata,
                                          data=self.tracker_data,
                                          all_feats2use=all_feats2use)

    @property
    def bud_trainer(self):
        props_file = self.save_dir / self.parameters.mother_bud_props_file
        if not hasattr(self, '_bud_trainer') or not self._bud_trainer:
            self._bud_trainer = BudTrainer(
                props_file=props_file,
                px_size=self.parameters.target_pixel_size)
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
            p = self.parameters._asdict()
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
            self._gen_test = None
            # Trigger save of the data
            datafile = self.save_dir / self.parameters.train_val_test_pairs_file
            self._impairs.save(datafile, self.base_dir)
            self._ncells = self._impairs.ncells

            # And for thet tracker datasets too
    def _tracker_check_for_data_update(self):
        if getattr(self, '_tracker_ncells',
                   None) != self._tracker_impairs.ncells:
            datafile = self.save_dir / self.parameters.tracker_tvt_pairs_file
            self._tracker_impairs.save(datafile, self.base_dir)
            self._tracker_ncells = self._tracker_impairs.ncells

    @property
    def data(self):
        if not hasattr(self, '_impairs') or not self._impairs:
            self._impairs = TrainValTestPairs()
            pairs_file = self.save_dir / self.parameters.train_val_test_pairs_file
            if pairs_file.is_file():
                self._impairs.load(pairs_file, self.base_dir)
        self._check_for_data_update()
        return self._impairs

    @data.setter
    def data(self, train_val_test_pairs):
        if isinstance(train_val_test_pairs, str):
            pairs_file = find_file(train_val_test_pairs, self.save_dir,
                                   'data')
            train_val_test_pairs = TrainValTestPairs()
            train_val_test_pairs.load(pairs_file, self.base_dir)
        if not isinstance(train_val_test_pairs, TrainValTestPairs):
            raise BadType(
                '"data" must be of type "baby.io.TrainValTestPairs"')
        self._impairs = train_val_test_pairs
        self._check_for_data_update()

    @property
    def tracker_data(self):
        if not hasattr(self, '_impairs') or not self._impairs:
            self._tracker_impairs = TrainValTestPairs()
            pairs_file = self.save_dir / self.parameters.tracker_tvt_pairs_file
            if pairs_file.is_file():
                self._tracker_impairs.load(pairs_file, self.base_dir)
        self._check_for_data_update()
        return self._tracker_impairs

    @data.setter
    def tracker_data(self, train_val_test_pairs):
        if isinstance(train_val_test_pairs, str):
            pairs_file = find_file(train_val_test_pairs, self.save_dir,
                                   'data')
            train_val_test_pairs = TrainValTestPairs()
            train_val_test_pairs.load(pairs_file, self.base_dir)
        if not isinstance(train_tvt_pairs, TrainValTestPairs):
            raise BadType(
                '"data" must be of type "baby.io.TrainValTestPairs"')
        self._tracker_impairs = train_val_test_pairs
        self._tracker_check_for_data_update()

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

        if not getattr(self, '_gen_test', None):
            if len(self.data.testing) == 0:
                raise BadProcess('No testing images have been added')
            # Initialise generator for testing images
            self._gen_test = ImageLabel(self.data.testing,
                                       batch_size=p.batch_size,
                                       aug=Augmenter(),
                                       preprocess=(robust_norm, seg_norm),
                                       in_memory=p.in_memory)

        return TrainValTestProperty(self._gen_train, self._gen_val,
                                    self._gen_test)

    def plot_gen_sample(self, validation=False):
        # TODO: Move to flattener?
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
        # train_gen = augmented_generator(self.gen.train, lambda x, y: (x, y))
        # val_gen = augmented_generator(self.gen.train, lambda x, y: (x, y))
        self.smoothing_sigma_trainer.generate_smoothing_sigma_stats(
            self.gen.train, self.gen.val)

    @property
    def smoothing_sigma_stats(self):
        warnings.warn(
            "nursery.smoothing_sigma_stats will soon be "
            "deprecated, use nursery.smoothing_sigma_trainer.stats "
            "instead", DeprecationWarning)
        return self.smoothing_sigma_trainer.stats

    @property
    def smoothing_sigma_model(self):
        return self.smoothing_sigma_trainer.model

    def generate_flattener_stats(self, max_erode=5):
        warnings.warn(
            "nursery.generate_flattener_stats will soon be "
            "deprecated, use "
            "nursery.flattener_trainer.generate_stats(train_gen, "
            "val_gen, train_aug, val_aug, max_erode=5) instead",
            DeprecationWarning)

        flattener = lambda x, y: x
        # NB: use isval=True for training aug since we do not need extra
        # augmentations for calibrating the flattener
        tAug = _std_aug(self.smoothing_sigma_model, flattener,
                        self.parameters, isval=True)
        vAug = _std_aug(self.smoothing_sigma_model,
                        flattener,
                        self.parameters,
                        isval=True)

        self.flattener_trainer.generate_flattener_stats(*self.gen[:2],
                                                        tAug,
                                                        vAug,
                                                        max_erode=max_erode)

    @property
    def flattener_stats(self):
        warnings.warn(
            "nursery.flattener_stats will soon be "
            "deprecated, use nursery.flattener_trainer.stats "
            "instead", DeprecationWarning)
        return self.flattener_trainer.stats

    @property
    def flattener(self):
        warnings.warn(
            "nursery.flattener will soon be "
            "deprecated, use nursery.flattener_trainer.flattener "
            "instead", DeprecationWarning)
        return self.flattener_trainer.flattener

    @flattener.setter
    def flattener(self, f):
        warnings.warn(
            "nursery.flattener will soon be "
            "deprecated, use nursery.flattener_trainer.flattener "
            "instead", DeprecationWarning)
        self.flattener_trainer.flattener = f

    @property
    def aug(self):
        p = self.parameters
        t = _std_aug(self.smoothing_sigma_model, self.flattener,
                     self.parameters)
        v = _std_aug(self.smoothing_sigma_model,
                     self.flattener,
                     self.parameters,
                     isval=True)
        w = _std_aug(self.smoothing_sigma_model,
                     self.flattener,
                     self.parameters,
                     isval=True)
        return TrainValTestProperty(t, v, w)

    @property
    def cnn_fn(self):
        warnings.warn(
            "nursery.cnn_fn will soon be "
            "deprecated, use nursery.cnn_trainer.cnn_fn "
            "instead", DeprecationWarning)
        return self.cnn_trainer.cnn_fn

    @cnn_fn.setter
    def cnn_fn(self, fn):
        warnings.warn(
            "nursery.cnn_fn will soon be "
            "deprecated, use nursery.cnn_trainer.cnn_fn "
            "instead", DeprecationWarning)
        self.cnn_trainer.cnn_fn = fn

    @property
    def cnn_dir(self):
        warnings.warn(
            "nursery.cnn_dir will soon be "
            "deprecated, use nursery.cnn_trainer.cnn_dir "
            "instead", DeprecationWarning)
        return self.cnn_trainer.cnn_dir

    @property
    def cnn_name(self):
        warnings.warn(
            "nursery.cnn_bane will soon be "
            "deprecated, use nursery.cnn_trainer.cnn_name "
            "instead", DeprecationWarning)
        return self.cnn_trainer.cnn_name

    @property
    def cnn(self):
        warnings.warn(
            "nursery.cnn will soon be "
            "deprecated, use nursery.cnn_trainer.cnn "
            "instead", DeprecationWarning)
        return self.cnn_trainer.cnn

    @property
    def histories(self):
        warnings.warn(
            "nursery.histories will soon be "
            "deprecated, use nursery.cnn_trainer.histories "
            "instead", DeprecationWarning)
        return self.cnn_trainer.histories

    @property
    def cnn_opt_dir(self):
        warnings.warn(
            "nursery.opt_dir will soon be "
            "deprecated, use nursery.cnn_trainer.opt_dir "
            "instead", DeprecationWarning)
        return self.cnn_trainer.opt_dir

    @property
    def cnn_opt(self):
        warnings.warn(
            "nursery.cnn_opt will soon be "
            "deprecated, use nursery.cnn_trainer.opt_cnn "
            "instead", DeprecationWarning)
        return self.cnn_trainer.opt_cnn

    def fit_smoothing_model(self, filt='identity'):
        warnings.warn(
            "nursery.fit_smoothing_model will soon be "
            "deprecated, use nursery.smoothing_signa_trainer.fit "
            "instead", DeprecationWarning)
        self.smoothing_sigma_trainer.fit(filt=filt)

    def plot_fitted_smoothing_sigma_model(self):
        warnings.warn(
            "nursery.plot_fitted_smoothing_sigma_model will soon be "
            "deprecated, use "
            "nursery.smoothing_signa_trainer.plot_fitted_model "
            "instead", DeprecationWarning)
        self.smoothing_sigma_trainer.plot_fitted_model()

    def fit_flattener(self, **kwargs):
        warnings.warn(
            "nursery.fit_flattener will soon be "
            "deprecated, use nursery.flattener_trainer.fit "
            "instead", DeprecationWarning)
        self.flattener_trainer.fit(**kwargs)

    def plot_flattener_stats(self, **kwargs):
        warnings.warn(
            "nursery.plot_flattener_stats will soon be "
            "deprecated, use nursery.flattener_trainer.plot_stats "
            "instead", DeprecationWarning)
        self.flattener_trainer.plot_stats(**kwargs)

    def fit_cnn(self, **kwargs):
        warnings.warn(
            "nursery.fit_cnn will soon be "
            "deprecated, use nursery.cnn_trainer.fit "
            "instead", DeprecationWarning)
        self.cnn_trainer.fit(**kwargs)

    def plot_histories(self, **kwargs):
        warnings.warn(
            "nursery.plot_histories will soon be "
            "deprecated, use nursery.cnn_trainer.plot_histories "
            "instead", DeprecationWarning)
        self.cnn_trainer.plot_histories(**kwargs)

    # TODO: move to Segmentation Param Trainer
    @property
    def seg_examples(self):
        p = self.parameters
        a = ScalingAugmenter(self.smoothing_sigma_model,
                             lambda lbl, _: lbl,
                             xy_out=p.xy_out,
                             target_pixel_size=p.target_pixel_size,
                             substacks=p.substacks,
                             p_noop=1,
                             probs={
                                 'vshift': 0.25,
                                 'hshift': 0.25
                             })

        def seg_example_aug(img, lbl):
            # Assume that the label preprocessing function also returns info
            _, info = lbl
            img, lbl = a(img, lbl)
            # In this case, always prefer the validation augmenter
            return img, lbl > 0, info

        def example_generator(dgen):
            opt_cnn = self.cnn_opt
            b_iter = batch_iterator(list(range(dgen.n_pairs)),
                                    batch_size=dgen.batch_size)
            with tqdm(total=dgen.n_pairs) as pbar:
                for b_inds in b_iter:
                    batch = [
                        dgen.get_by_index(b, aug=seg_example_aug)
                        for b in b_inds
                    ]
                    preds = split_batch_pred(
                        opt_cnn.predict(np.stack([img for img, _, _ in batch
                                                 ])))
                    for pred, (img, lbl, info) in zip(preds, batch):
                        pbar.update()
                        lbl = lbl.transpose(2, 0, 1)
                        # Filter out examples that have been augmented away
                        valid = lbl.sum(axis=(1, 2)) > 0
                        lbl = lbl[valid]
                        clab = info.get('cellLabels', []) or []
                        if type(clab) is int:
                            clab = [clab]
                        clab = [l for l, v in zip(clab, valid) if v]
                        info['cellLabels'] = clab
                        buds = info.get('buds', []) or []
                        if type(buds) is int:
                            buds = [buds]
                        buds = [b for b, v in zip(buds, valid) if v]
                        info['buds'] = buds
                        yield SegExample(pred, lbl, info, img)

        if self.in_memory:
            if getattr(self, '_seg_examples', None) is None:
                self._seg_examples = TrainValTestProperty(
                    list(example_generator(self.gen.train)),
                    list(example_generator(self.gen.val)),
                    list(example_generator(self.gen.test)))
            return TrainValTestProperty((e for e in self._seg_examples.train),
                                        (e for e in self._seg_examples.val),
                                        (e for e in self._seg_examples.test))
        else:
            self._seg_examples = None
            return TrainValTestProperty(example_generator(self.gen.train),
                                        example_generator(self.gen.val),
                                        example_generator(self.gen.test))

    # TODO Move to Segmentation Parameter TRainer
    @property
    def seg_param_stats(self):
        if getattr(self, '_seg_param_stats', None) is None:
            p = self.parameters
            stats_file = self.save_dir / p.segmentation_stats_file
            if not stats_file.is_file():
                raise BadProcess('"fit_seg_params" has not been run yet')
            self._seg_param_stats = pd.read_csv(stats_file, index_col=0)
        return self._seg_param_stats

    # TODO Move to Segmentation Parameter TRainer
    @property
    def seg_params(self):
        params_file = self.save_dir / self.parameters.segmentation_param_file
        with open(params_file, 'rt') as f:
            params = json.load(f)
        return params

    # TODO Move to Segmentation Parameter TRainer
    @seg_params.setter
    def seg_params(self, val):
        if not type(val) == dict:
            raise BadParam('"seg_params" should be a "dict"')
        msg_args = inspect.getfullargspec(MorphSegGrouped.__init__).args
        if not set(val.keys()).issubset(msg_args):
            raise BadParam(
                '"seg_params" must specify arguments to "MorphSegGrouped"')
        params_file = self.save_dir / self.parameters.segmentation_param_file
        with open(params_file, 'wt') as f:
            json.dump(jsonify(val), f)

    # TODO Deprecate and keep in bud_trainer
    def generate_bud_stats(self):
        self.bud_trainer.generate_property_table(self.seg_examples,
                                                 self.flattener)

    # TODO Depcrecate and keep in bud_trainer
    def fit_bud_model(self, **kwargs):
        self.bud_trainer.explore_hyperparams(**kwargs)
        model_file = self.save_dir / self.parameters.mother_bud_model_file
        self.bud_trainer.save_model(model_file)

    #Todo Move to Segmentation Parameter Trainer
    def fit_seg_params(self, njobs=5, scoring='F0_5'):
        param_grid = list(product(*seg_param_coords.values()))
        basic_pars = list(seg_param_coords.keys())

        # TODO switch back to validation examples
        val_examples = list(self.seg_examples.val)
        from joblib import Parallel, delayed
        rows = []
        for gind in range(3)[::-1]:
            rows.extend(
                Parallel(n_jobs=njobs)(
                    delayed(_seg_filter_optim)(gind,
                                               pars,
                                               basic_pars,
                                               self.flattener,
                                               val_examples,
                                               base_params=base_seg_params,
                                               scoring=scoring)
                    for pars in tqdm(param_grid)))

        rows_expanded = [
            dict(
                chain(*[[('_'.join((k, str(g))), gv)
                         for g, gv in enumerate(v)] if type(v) == list else [(
                             k, v)]
                        for k, v in chain([
                            ('group', row['group']), ('score', row['score'])
                        ], row['basic'].items(), row['filter'].items())]))
            for row in rows
        ]

        self._seg_param_stats = pd.DataFrame(rows_expanded)
        stats_file = self.save_dir / self.parameters.segmentation_stats_file
        self._seg_param_stats.to_csv(stats_file)

        self.refit_filter_seg_params(scoring=scoring)

    # TODO move to Segmentation Parameter Trainer
    def refit_filter_seg_params(self,
                                lazy=False,
                                bootstrap=False,
                                scoring='F0_5'):

        # Merge the best parameters from each group into a single parameter set
        merged_params = {k: v.copy() for k, v in base_seg_params.items()}
        stats = self.seg_param_stats
        for g, r in enumerate(stats.groupby('group').score.idxmax()):
            for k in merged_params:
                merged_params[k][g] = stats.loc[r, k + '_' + str(g)]

        sfpo = SegFilterParamOptim(self.flattener,
                                   basic_params=merged_params,
                                   scoring=scoring)
        sfpo.generate_stat_table(self.seg_examples.val)

        sfpo.fit_filter_params(lazy=lazy, bootstrap=bootstrap)
        merged_params.update(sfpo.opt_params)
        self.seg_params = merged_params

    # TODO Move to Segmentation Parameter Trainer
    def validate_seg_params(self, iou_thresh=0.7, save=True):
        segmenter = MorphSegGrouped(self.flattener,
                                    return_masks=True,
                                    fit_radial=True,
                                    use_group_thresh=True,
                                    **self.seg_params)
        edge_inds = [
            i for i, t in enumerate(self.flattener.targets)
            if t.prop == 'edge'
        ]
        stats = {}
        dfs = {}
        for k, seg_exs in zip(self.seg_examples._fields, self.seg_examples):
            stats[k] = []
            for seg_ex in seg_exs:
                seg = segmenter.segment(seg_ex.pred, refine_outlines=True)
                edge_scores = np.array([
                    seg_ex.pred[edge_inds, ...].max(axis=0)[s].mean()
                    for s in seg.edges
                ])
                IoUs = calc_IoUs(seg_ex.target, seg.masks)
                bIoU, _ = best_IoU(IoUs)
                stats[k].append((edge_scores, IoUs, np.mean(bIoU),
                                 np.min(bIoU, initial=1),
                                 calc_AP(IoUs,
                                         probs=edge_scores,
                                         iou_thresh=iou_thresh)[0]))
            dfs[k] = pd.DataFrame([s[2:] for s in stats[k]],
                                  columns=['IoU_mean', 'IoU_min', 'AP'])

        print({k: df.mean() for k, df in dfs.items()})

        nrows = len(dfs)
        ncols = dfs['val'].shape[1]
        fig, axs = plt.subplots(nrows=nrows,
                                ncols=ncols,
                                figsize=(ncols * 4, nrows * 4))
        for axrow, (k, df) in zip(axs, dfs.items()):
            for ax, col in zip(axrow, df.columns):
                ax.hist(df.loc[:, col], bins=26, range=(0, 1))
                ax.set(xlabel=col, title=k)
        if save:
            fig.savefig(self.save_dir / 'seg_validation_plot.png')
            plt.close(fig)


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


def _seg_filter_optim(g,
                      p,
                      pk,
                      flattener,
                      seg_gen,
                      base_params=default_params,
                      scoring='F0_5'):
    p = _sub_params({(k, g): v for k, v in zip(pk, p)}, base_params)
    sfpo = SegFilterParamOptim(flattener, basic_params=p, scoring=scoring)
    sfpo.generate_stat_table(seg_gen)
    sfpo.fit_filter_params(lazy=True, bootstrap=False)
    return {
        'group': g,
        'basic': p,
        'filter': sfpo.opt_params,
        'score': sfpo.opt_score
    }


def _std_aug(ssm, flattener, p, isval=False):
    probs = {'vshift': 0.25, 'hshift': 0.25}
    extra_args = {}
    if isval:
        extra_args['p_noop'] = 1
    else:
        probs['rotate'] = 0.2

    return ScalingAugmenter(ssm,
                            flattener,
                            xy_out=p.xy_out,
                            target_pixel_size=p.target_pixel_size,
                            substacks=p.substacks,
                            probs=probs,
                            **extra_args)
