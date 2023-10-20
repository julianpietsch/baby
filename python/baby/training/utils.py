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
from typing import NamedTuple, Any, Tuple, Union
from itertools import repeat
import shutil
import json
from pathlib import Path

import tensorflow as tf
from tensorflow.keras.optimizers.schedules import CosineDecay

from baby.errors import BadFile, BadParam, BadProcess, BadType
from baby.io import TrainValTestPairs
from baby.preprocessing import (robust_norm, robust_norm_dw, seg_norm,
                                SegmentationFlattening)
from baby.augmentation import Augmenter, ScalingAugmenter, SmoothingSigmaModel
from baby.generator import ImageLabel, AugmentedGenerator
from baby.utils import (find_file, EncodableNamedTuple,
                        jsonify, as_python_object)
from baby.morph_thresh_seg import SegmentationParameters


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


class TrainValTestProperty(NamedTuple):
    train: Any
    val: Any
    test: Any


@EncodableNamedTuple
class BabyTrainerParameters(NamedTuple):
    """Global parameters for training BABY models
    """
    train_val_test_pairs_file: str = 'train_val_pairs.json'
    tracker_tvt_pairs_file: str = 'tracker_tvt_pairs.json'
    smoothing_sigma_stats_file: str = 'smoothing_sigma_stats.csv'
    smoothing_sigma_model_file: str = 'smoothing_sigma_model.json'
    flattener_stats_file: str = 'flattener_stats.json'
    flattener_file: str = 'flattener.json'
    segmentation_stats_file: str = 'segmentation_stats.csv'
    segmentation_param_file: str = 'segmentation_params.json'
    mother_bud_props_file: str = 'mother_bud_props.csv'
    mother_bud_model_file: str = 'mother_bud_model.pkl'
    cnn_set: Tuple[str, ...] = ('unet_4s',)
    cnn_fn: Union[None, str] = None
    batch_size: int = 8
    in_memory: bool = True
    xy_out: int = 80
    target_pixel_size: float = 0.263
    substacks: Union[None, int] = None
    aug_probs: dict = {}
    aug_p_noop: float = 0.05
    base_seg_params: dict = {}
    seg_param_coords: dict = {}
    input_norm_dw: bool = False
    only_basic_augs: bool = True
    balanced_sampling: bool = False
    use_sample_weights: bool = False
    canny_padding: int = 2
    n_jobs: int = 4


TRAINING_PARAMETERS_FILENAME =  'training_parameters.json'
"""File name to which training parameters are saved"""


class SharedParameterContainer(object):
    """A container of current training-related parameters.

    Designed to be used by :py:class:`training.BabyTrainer` and passed to the
    children trainers. Updates on this object by the
    :py:class:`training.BabyTrainer` are then automatially propagated to the
    classes that use the parameters.

    Parameters are auto-saved to a file
    :py:const:`TRAINING_PARAMETERS_FILENAME` within the ``save_dir``.

    Args:
        save_dir (str or Path): directory in which to save parameters and/or
            from which to auto-load parameters.
        params (None or BabyTrainerParameters or str or Path): Optionally
            specify initial training parameters as a
            :py:class:`BabyTrainerParameters` instance or the path to a saved
            :py:class:`BabyTrainerParameters` instance.
    """
    def __init__(self, save_dir, params=None):
        self.save_dir = Path(save_dir)
        self._parameters_file = TRAINING_PARAMETERS_FILENAME

        # Trigger parameter loading/initialisation via property
        self.parameters
        self.segmentation_parameters

        # Register parameters if specified
        if isinstance(params, BabyTrainerParameters):
            self.parameters = params
        elif params is not None:
            filename = find_file(params, save_dir, 'params')
            savename = save_dir / self._parameters_file
            if filename != savename:
                shutil.copy(filename, savename)

    @property
    def parameters(self):
        """A :py:class:`BabyTrainerParameters` instance.

        If not already initialised, parameters are loaded from the save file
        if found, otherwise they are initialised to defaults as per
        :py:class:`BabyTrainerParameters`.

        This can be set to either a new ``BabyTrainerParameters`` instance, or
        to a ``dict`` that specifies existing parameter values to replace.
        """
        if not getattr(self, '_parameters', None):
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
            if not getattr(self, '_parameters', None):
                self._parameters = BabyTrainerParameters()
            params = self._parameters._replace(**params)
        elif not isinstance(params, BabyTrainerParameters):
            params = BabyTrainerParameters(*params)
        self._parameters = params
        with open(self.save_dir / self._parameters_file, 'wt') as f:
            json.dump(jsonify(self._parameters), f)

    @property
    def segmentation_parameters(self):
        """A :py:class:`baby.segmentation.SegmentationParameters` instance.

        If not already initialised, parameters are loaded from the save file
        if found, otherwise they are initialised to defaults as per
        :py:class:`baby.segmentation.SegmentationParameters`.

        This can be set to either a new ``SegmentationParameters`` instance,
        or to a ``dict`` that specifies existing parameter values to replace.
        """
        if not getattr(self, '_segmentation_parameters', None):
            seg_param_file = (self.save_dir /
                              self.parameters.segmentation_param_file)
            if seg_param_file.is_file():
                with open(seg_param_file, 'rt') as f:
                    params = json.load(f, object_hook=as_python_object)
                if not isinstance(params, SegmentationParameters):
                    raise BadFile(
                        'Segmentation parameters file has been corrupted.')
                self._segmentation_parameters = params
            else:
                self.segmentation_parameters = SegmentationParameters()
        return self._segmentation_parameters

    @segmentation_parameters.setter
    def segmentation_parameters(self, params):
        if isinstance(params, dict):
            if not getattr(self, '_segmentation_parameters', None):
                self._segmentation_parameters = SegmentationParameters()
            params = self._segmentation_parameters._replace(**params)
        elif not isinstance(params, SegmentationParameters):
            params = SegmentationParameters(*params)
        self._segmentation_parameters = params
        seg_param_file = (self.save_dir /
                          self.parameters.segmentation_param_file)
        with open(seg_param_file, 'wt') as f:
            json.dump(jsonify(self._segmentation_parameters), f, indent=2)


class SharedDataContainer(object):
    """A container of current data and generators.

    Designed to be used by :py:class:`training.BabyTrainer` and passed to the
    children trainers. Updates on this object by the
    :py:class:`training.BabyTrainer` are then automatially propagated to the
    classes that use the parameters.

    Parameters are auto-saved to a file
    :py:const:`TRAINING_PARAMETERS_FILENAME` within the ``save_dir``.

    Args:
        shared_params (SharedParameterContainer): Shared parameters.
        base_dir (None or str or Path): Base directory within which all
            relevant image files can be found. References to the image files
            will be saved relative to this directory.
    """
    def __init__(self, shared_params, base_dir=None):
        self._shared_params = shared_params

        if base_dir is not None:
            base_dir = Path(base_dir)
            if not base_dir.is_dir():
                raise BadParam('"base_dir" must be a valid directory or None')
        else:
            base_dir = Path.cwd()
        self.base_dir = base_dir

    @property
    def save_dir(self):
        return self._shared_params.save_dir

    @property
    def parameters(self):
        return self._shared_params.parameters

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

    def _tracker_check_for_data_update(self):
        # And for thet tracker datasets too
        if (getattr(self, '_tracker_ncells', None)
            != self._tracker_impairs.ncells):
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
        """Training, validation and test data generators.

        This attribute provides three :py:class:`ImageLabel` generators as a
        :py:class:`TrainValTestProperty`, with each generator assigned just a
        dummy augmenter to begin with.

        Note:
            Generator initialisation requires that all specified images exist.
        """

        old_gen_params = getattr(self, '_current_gen_params', None)
        new_gen_params = tuple(getattr(self.parameters, p) for p in
                               ('in_memory', 'input_norm_dw', 'batch_size',
                                'balanced_sampling', 'use_sample_weights'))
        if old_gen_params != new_gen_params:
            self._gen_train = None
            self._gen_val = None
            self._gen_test = None
            self._current_gen_params = new_gen_params

        (in_memory, input_norm_dw, batch_size,
         balanced_sampling, use_sample_weights) = new_gen_params
        input_norm = robust_norm_dw if input_norm_dw else robust_norm

        if not getattr(self, '_gen_train', None):
            if len(self.data.training) == 0:
                raise BadProcess('No training images have been added')
            # Initialise generator for training images
            self._gen_train = ImageLabel(self.data.training,
                                         batch_size=batch_size,
                                         aug=Augmenter(),
                                         preprocess=(input_norm, seg_norm),
                                         in_memory=in_memory,
                                         balanced_sampling=balanced_sampling,
                                         use_sample_weights=use_sample_weights)

        if not getattr(self, '_gen_val', None):
            if len(self.data.validation) == 0:
                raise BadProcess('No validation images have been added')
            # Initialise generator for validation images
            self._gen_val = ImageLabel(self.data.validation,
                                       batch_size=batch_size,
                                       aug=Augmenter(),
                                       preprocess=(input_norm, seg_norm),
                                       in_memory=in_memory,
                                       balanced_sampling=balanced_sampling,
                                       use_sample_weights=use_sample_weights)

        if not getattr(self, '_gen_test', None):
            if len(self.data.testing) == 0:
                raise BadProcess('No testing images have been added')
            # Initialise generator for testing images
            self._gen_test = ImageLabel(self.data.testing,
                                        batch_size=batch_size,
                                        aug=Augmenter(),
                                        preprocess=(input_norm, seg_norm),
                                        in_memory=in_memory,
                                        balanced_sampling=balanced_sampling,
                                        use_sample_weights=use_sample_weights)

        self._gen_train.n_jobs = self.parameters.n_jobs
        self._gen_val.n_jobs = self.parameters.n_jobs
        self._gen_test.n_jobs = self.parameters.n_jobs
        return TrainValTestProperty(self._gen_train, self._gen_val,
                self._gen_test)

    def gen_with_aug(self, aug):
        """Returns generators wrapped with alternative augmenters.

        Args:
            aug (Augmenter or Tuple[Augmenter, Augmenter, Augmenter]):
                Augmenter to use, or tuple of different augmenters for
                training, validation and testing generators.

        Returns:
            :py:class:`TrainValTestProperty` of :py:class:`AugmentedGenerator`
            objects for training, validation and testing generators.
        """
        atrain, aval, atest = aug if type(aug) == tuple else repeat(aug, 3)
        gtrain, gval, gtest = self.gen
        return TrainValTestProperty(
            AugmentedGenerator(gtrain, atrain),
            AugmentedGenerator(gval, aval),
            AugmentedGenerator(gtest, atest))


VALIDATION_AUGMENTATIONS = {'vshift', 'hshift'}


def standard_augmenter(ssm, flattener, params, isval=False):
    """Returns an augmenter for training on flattenable inputs.

    Args:
        ssm (SmoothingSigmaModel): Smoothing model to use.
        flattener (SegmentationFlattening): Flattener to apply after
            augmentation.
        params (BabyTrainerParameters): Parameters to use when constructing
            augmenter.
        isval (bool): If ``True``, set unspecified augmentation probabilities
            to zero for generating validation data. If ``False``, augmentation
            probabilities are left at defaults (see also
            :py:class:`ScalingAugmenter` and
            :py:class:`BabyTrainerParameters`).

    Returns:
        A :py:class:`ScalingAugmenter` with specified ``ssm`` and
        ``flattener`` and parameterisation according to ``params``.
    """
    probs = {'vshift': 0.25, 'hshift': 0.25}
    extra_args = dict(canny_padding=params.canny_padding,
                      p_noop=params.aug_p_noop)
    if isval:
        extra_args['p_noop'] = 1
    else:
        probs['rotate'] = 0.25

    probs.update(params.aug_probs)
    if isval:
        probs = {k: v for k, v in probs.items()
                 if k in VALIDATION_AUGMENTATIONS}

    return ScalingAugmenter(ssm,
            flattener,
            xy_out=params.xy_out,
            target_pixel_size=params.target_pixel_size,
            substacks=params.substacks,
            probs=probs,
            only_basic_augs=params.only_basic_augs,
            **extra_args)


def warmup_and_cosine_decay(learning_rate=0.001, warmup_steps=30, decay_steps=370):
    decay = CosineDecay(learning_rate, decay_steps)
    def warmup_schedule(step):
        if step < warmup_steps:
            return step / warmup_steps * learning_rate
        else:
            return decay(step - warmup_steps)
    return warmup_schedule
