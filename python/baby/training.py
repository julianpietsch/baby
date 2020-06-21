from pathlib import Path
import shutil
import json
from typing import NamedTuple, Union
import numpy as np
import tensorflow as tf
from tensorflow.python.keras import backend as K
from tensorflow.python.keras.callbacks import (ModelCheckpoint, TensorBoard,
                                               LearningRateScheduler)
from tensorflow.python.keras.models import load_model

from .utils import (get_name, EncodableNamedTuple, find_file,
                    as_python_object, jsonify)
from .errors import BadParam, BadFile, BadType, BadProcess
from .io import TrainValPairs
from .preprocessing import robust_norm, seg_norm, SegmentationFlattening
from .augmentation import Augmenter, SmoothingSigmaModel, DownscalingAugmenter
from .generator import ImageLabel
from .losses import bce_dice_loss, dice_coeff
from . import models

custom_objects = {'bce_dice_loss': bce_dice_loss, 'dice_coeff': dice_coeff}


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


@EncodableNamedTuple
class BabyTrainerParameters(NamedTuple):
    """Global parameters for training BABY models
    """
    train_val_pairs_file: str = 'train_val_pairs.json'
    smoothing_sigma_model_file: str = 'smoothing_sigma_model.json'
    flattener_file: str = 'flattener.json'
    model_fn: str = 'msd_d80'
    batch_size: int = 8
    in_memory: bool = True
    xy_out: int = 80


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

    def __init__(self, save_dir, params=None):

        # Register the save dir
        save_dir = Path(save_dir)
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
    def data(self):
        if not hasattr(self, '_impairs') or not self._impairs:
            self._impairs = TrainValPairs()
            pairs_file = self.save_dir / self.parameters.train_val_pairs_file
            if pairs_file.is_file():
                self._impairs.load(pairs_file)
        return self._impairs

    @data.setter
    def data(self, train_val_pairs):
        if isinstance(train_val_pairs, str):
            pairs_file = find_file(train_val_pairs, self.save_dir, 'data')
            train_val_pairs = TrainValPairs()
            train_val_pairs.load(pairs_file)
        if not isinstance(train_val_pairs, TrainValPairs):
            raise BadType('"data" must be of type "baby.io.TrainValPairs"')
        self._impairs = train_val_pairs
        self._train_gen = None
        self._val_gen = None

    def _ensure_generators(self):
        has_train = hasattr(self, '_train_gen') and self._train_gen
        has_val = hasattr(self, '_val_gen') and self._val_gen
        if not has_train or not has_val:
            p = self.parameters

            # Before updating the generators, ensure that the current set of
            # training/validation data has been saved:
            if len(self.data.training) == 0:
                raise BadProcess('No training images have been added')
            if len(self.data.validation) == 0:
                raise BadProcess('No validation images have been added')
            self.data.save(self.save_dir / p.train_val_pairs_file)

            # Initialise generators for the training and validation images
            # NB: these also check to make sure all the specified images exist
            # NB: only dummy augmenters are assigned
            self._train_gen = ImageLabel(self.data.training,
                                         batch_size=p.batch_size,
                                         aug=Augmenter(),
                                         preprocess=(robust_norm, seg_norm),
                                         in_memory=p.in_memory)
            self._val_gen = ImageLabel(self.data.validation,
                                       batch_size=p.batch_size,
                                       aug=Augmenter(),
                                       preprocess=(robust_norm, seg_norm),
                                       in_memory=p.in_memory)

    @property
    def train_gen(self):
        self._ensure_generators()
        return self._train_gen

    @property
    def val_gen(self):
        self._ensure_generators()
        return self._val_gen

    @property
    def smoothing_sigma_model(self):
        if not hasattr(self, '_ssm') or not self._ssm:
            p = self.parameters
            ssm_file = self.save_dir / p.smoothing_sigma_model_file
            if ssm_file.is_file():
                ssm = SmoothingSigmaModel()
                ssm.load(ssm_file)
                self._ssm = ssm
            else:
                raise BadProcess(
                    'The "smoothing_sigma_model" has not been assigned yet')
        return self._ssm

    @smoothing_sigma_model.setter
    def smoothing_sigma_model(self, ssm):
        if isinstance(ssm, str):
            ssm_file = find_file(ssm, self.save_dir, 'smoothing_sigma_model')
            ssm = SmoothingSigmaModel()
            ssm.load(ssm_file)
        if not isinstance(ssm, SmoothingSigmaModel):
            raise BadType(
                '"smoothing_sigma_model" must be of type "baby.augmentation.SmoothingSigmaModel"'
            )
        p = self.parameters
        ssm.save(self.save_dir / p.smoothing_sigma_model_file)
        self._ssm = ssm

    @property
    def flattener(self):
        if not hasattr(self, '_flattener') or not self._flattener:
            p = self.parameters
            flattener_file = self.save_dir / p.flattener_file
            if flattener_file.is_file():
                f = SegmentationFlattening()
                f.load(flattener_file)
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
        f.save(self.save_dir / self.parameters.flattener_file)
        self._flattener = f

    @property
    def train_aug(self):
        # Both training and validation make use of the DownscalingAugmenter
        p = self.parameters
        return DownscalingAugmenter(self.smoothing_sigma_model,
                                    self.flattener,
                                    xy_out=p.xy_out,
                                    xy_scaled=81,
                                    pixdev=4,
                                    probs={
                                        'rotate': 0.2,
                                        'vshift': 0.25,
                                        'hshift': 0.25
                                    })

    @property
    def val_aug(self):
        # Validation data now must make use of DownscalingAugmenter, so need to update
        # the generator as well:
        p = self.parameters
        return DownscalingAugmenter(self.smoothing_sigma_model,
                                    self.flattener,
                                    xy_out=p.xy_out,
                                    xy_scaled=81,
                                    pixdev=0,
                                    p_noop=1,
                                    probs={
                                        'vshift': 0.25,
                                        'hshift': 0.25
                                    })

    @property
    def cnn_fn(self):
        return getattr(models, self.parameters.model_fn)

    @property
    def cnn_name(self):
        return get_name(self.cnn_fn)

    @property
    def cnn(self):
        if not hasattr(self, '_cnn') or not self._cnn:
            self.train_gen.aug = self.train_aug
            self._cnn = self.cnn_fn(self.train_gen, self.flattener)
        return self._cnn

    def fit_smoothing_model(self):
        pass

    def fit_flattener(self):
        pass

    def fit_cnn(self):
        pass

    def fit_seg_params(self):
        pass


class Nursery(BabyTrainer):
    pass


def save_init(model, model_name):
    filename = log_dir / '{}_init_weights.h5'.format(model_name)
    if filename.exists():
        print('Initial weights already saved.')
    else:
        print('Saving initial weights...')
        model.save_weights(str(filename))


def make_callbacks(subdir, schedule=[(1e-3, 400)]):
    outdir = log_dir / subdir
    weights_file = outdir / 'weights.hdf5'
    assert not weights_file.exists(
    ), 'Trained weights already exist for this subdir'

    return [
        ModelCheckpoint(filepath=str(weights_file),
                        monitor='val_loss',
                        save_best_only=True,
                        verbose=1),
        TensorBoard(log_dir=str(outdir / 'logs')),
        LearningRateScheduler(lambda epoch: schedule_steps(epoch, schedule))
    ]


def save_history(subdir, history):
    with open(log_dir / subdir / 'history.pkl', 'wb') as f:
        pickle.dump({'history': history.history, 'epoch': history.epoch}, f)


def train_model(model,
                subdir,
                model_name=None,
                epochs=400,
                schedule=None,
                check=False):
    # First check output names match current flattener names
    assert (all(
        [m == f for m, f in zip(model.output_names, flattener.names())]))

    if schedule is None:
        schedule = [(1e-3, epochs)]
    if model_name is None:
        model_name = subdir
    init_weights_file = log_dir / '{}_init_weights.h5'.format(model_name)
    assert init_weights_file.exists(
    ), 'Initial weights have not been saved for this model'

    model.load_weights(str(init_weights_file))

    if check:
        return

    history = model.fit_generator(generator=train_gen,
                                  validation_data=val_gen,
                                  epochs=400,
                                  callbacks=make_callbacks(subdir))
    save_history(subdir, history)


def load_history(subdir):
    with open(log_dir / subdir / 'history.pkl', 'rb') as f:
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
