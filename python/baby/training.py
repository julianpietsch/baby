from pathlib import Path
import shutil
import json
import pickle
from typing import NamedTuple, Union, Tuple
import numpy as np
from scipy.signal import savgol_filter
from matplotlib.colors import to_rgba
import tensorflow as tf
from tensorflow.python.keras import backend as K
from tensorflow.python.keras.callbacks import (ModelCheckpoint, TensorBoard,
                                               LearningRateScheduler)
from tensorflow.python.keras.models import load_model

from .utils import (get_name, EncodableNamedTuple, find_file,
                    as_python_object, jsonify, schedule_steps)
from .errors import BadParam, BadFile, BadType, BadProcess
from .io import TrainValPairs, load_tiled_image
from .preprocessing import robust_norm, seg_norm, SegmentationFlattening
from .augmentation import Augmenter, SmoothingSigmaModel, DownscalingAugmenter
from .generator import ImageLabel
from .losses import bce_dice_loss, dice_coeff
from . import models

#Alan imports
from .tracker import Tracker
from scipy.ndimage import binary_fill_holes
from skimage.measure import regionprops_table

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


@EncodableNamedTuple
class BabyTrainerParameters(NamedTuple):
    """Global parameters for training BABY models
    """
    train_val_pairs_file: str = 'train_val_pairs.json'
    smoothing_sigma_model_file: str = 'smoothing_sigma_model.json'
    flattener_file: str = 'flattener.json'
    cnn_set: Tuple[str, ...] = ('msd_d80',)
    cnn_fn: Union[None, str] = None
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

    def __init__(self, save_dir, params=None, max_cnns=3):

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

        self._max_cnns = max_cnns

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
        if not hasattr(self, '_active_cnn_fn') or not self._active_cnn_fn:
            self.cnn_fn = self.parameters.cnn_set[0]
        return getattr(models, self._active_cnn_fn)

    @cnn_fn.setter
    def cnn_fn(self, fn):
        if fn not in self.parameters.cnn_set:
            raise BadType('That model is not in "parameters.cnn_set"')
        if not hasattr(models, fn):
            raise BadType('That is not a recognised model')
        self._active_cnn_fn = fn

    @property
    def cnn_dir(self):
        self.cnn_fn  # ensure that _active_cnn_fn is initialised
        d = self.save_dir / self._active_cnn_fn
        if not d.is_dir():
            d.mkdir()
        return d

    @property
    def cnn_name(self):
        return get_name(self.cnn_fn)

    @property
    def cnn(self):
        if not hasattr(self, '_cnns') or not self._cnns:
            self._cnns = {}
        self.cnn_fn  # ensure that _active_cnn_fn is initialised
        cnn_id = self._active_cnn_fn
        if cnn_id not in self._cnns:
            if len(self._cnns) > self._max_cnns:
                # To avoid over-consuming memory reset graph
                # TODO: ensure TF1/TF2 compat and check RTX bug
                tf.keras.backend.clear_session()
                self._cnns = {}
            self.train_gen.aug = self.train_aug
            print('Loading "{}" CNN...'.format(self.cnn_name))
            model = self.cnn_fn(self.train_gen, self.flattener)
            self._cnns[cnn_id] = model

            # Save initial weights if they haven't already been saved
            filename = self.cnn_dir / INIT_WEIGHTS_FILE
            if not filename.exists():
                print('Saving initial weights...')
                model.save_weights(str(filename))
        return self._cnns[cnn_id]

    @property
    def ctrack_trainer(self):
        self._trackTrainer = TrackingTrainer(self.data._metadata, self.data)

    def fit_smoothing_model(self):
        pass

    def fit_flattener(self):
        pass

    def fit_cnn(self, epochs=400, schedule=None, replace=False, extend=False):
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
        self.train_gen.aug = self.train_aug
        self.val_gen.aug = self.val_aug
        history = self.cnn.fit_generator(generator=self.train_gen,
                                         validation_data=self.val_gen,
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

    def plot_histories(self,
                       key='loss',
                       log=False,
                       window=21,
                       ax=None,
                       legend=False):
        if ax is None:
            from matplotlib import pyplot as plt
            ax = plt.gca()

        max_epoch = 1
        active_cnn_fn = None
        if hasattr(self, '_active_cnn_fn'):
            active_cnn_fn = self._active_cnn_fn
        try:
            for cnn_name in self.parameters.cnn_set:
                self.cnn_fn = cnn_name
                history_file = self.cnn_dir / HISTORY_FILE
                if not history_file.exists():
                    continue
                with open(history_file, 'rb') as f:
                    history = pickle.load(f)
                epoch = history['epoch']
                max_epoch = max([max_epoch, max(epoch)])
                val = history['history']['val_' + key]
                hndl = ax.plot(epoch,
                               savgol_filter(val, window, 3),
                               label=self.cnn_name + ' Val')
                val = history['history'][key]
                colour = to_rgba(hndl[0].get_color(), 0.7)
                ax.plot(epoch,
                        savgol_filter(val, window, 3),
                        ':',
                        color=colour,
                        label=self.cnn_name + ' Train')
        finally:
            self._active_cnn_fn = active_cnn_fn

        ax.set(xlabel='Epochs',
               ylabel=key.replace('_', ' ').title(),
               xlim=[0, max_epoch])
        if log:
            ax.set_yscale('log')
        if legend:
            ax.legend()

    def fit_seg_params(self):
        pass


class Nursery(BabyTrainer):
    pass


class TrackingTrainer(Tracker):

    def __init__(self, meta, data=None, masks=None):
        super().__init__()
        self.meta = meta
        self.meta.set_index(['position', 'trap', 'tp'], inplace=True)
        self.data = data
        if masks is None:
            self.masks = [
                load_tiled_image(mask)[0] for bf, mask in self.data.training
            ]
        self.process_traps()
        self.gen_train_data()

    def get_img_feats(self, img_array):
        props_df = pd.DataFrame([
            regionprops_table(img, properties=self.feats2use, cache=True)
            for img in img_array
        ]).applymap(lambda x: x[0])

        return props_df

    def gen_train(self):
        '''
        Generates the data for training using all the loaded images.
        '''

        traps = *map(tuple,
                     np.unique([ind[:2] for ind in self.meta.index], axis=0)),

        train = *map(self.gen_train_from_trap, traps),
        self.train = np.concatenate(train)

    def gen_train_from_pair(self, pair_loc):
        subdf = self.meta[['list_index', 'cellLabels']].loc(axis=0)[pair_loc]

        truemat = np.equal.outer(*subdf['cellLabels'].values).reshape(-1)
        propsmat = self.df_calc_feat_matrix(pair_loc).reshape(-1, self.nfeats)

        return [x for x in zip(truemat, propsmat)]

    def process_traps(self):
        '''
        Process all traps (run for finished experiments), combine results with location df and drop
        unused columns.

        Generates a region_proprieties DataFrame
        '''

        print('entra')
        nindex = []
        props_list = []
        for ind, (index,
                  lbl) in zip(self.meta.index,
                              self.meta[['list_index', 'cellLabels']].values):
            trapfeats = [
                regionprops_table(self.masks[index][..., i].astype('int'),
                                  properties=self.feats2use)  #
                for i in range(
                    len(lbl)
                )  #TODO Continue here once metadata issue is fixed <2020-06-22 Mon>
            ]
            for cell, feats in zip(lbl, trapfeats):
                nindex.append(ind + (cell,))
                props_list.append(feats)

        out_dict = {key: [] for key in props_list[0].keys()}
        nindex = pd.MultiIndex.from_tuples(nindex, names=self.cindices)

        for cells_props in props_list:
            for key, val in cells_props.items():
                out_dict[key].append(val[0])

        self.rprops = pd.DataFrame(out_dict, index=nindex)
        self.rprop_keys = self.rprops.columns

    def gen_train_from_trap(self, trap_loc):
        subdf = self.meta[['list_index', 'cellLabels'
                          ]].loc(axis=0)[trap_loc].sort_values('tp')
        pairs = [
            trap_loc + tuple((pair,))
            for pair in zip(subdf.index[:-1], subdf.index[1:])
        ]

        res_tuples = [
            tup for pair in pairs for tup in self.gen_train_from_pair(pair)
        ]

        return res_tuples


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
