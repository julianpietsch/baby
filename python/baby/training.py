from pathlib import Path
import shutil
import json
import pickle
from typing import NamedTuple, Union, Tuple, Any
import numpy as np
from numpy.polynomial import Polynomial
from scipy.optimize import curve_fit
import pandas as pd
from tqdm import tqdm, trange
from skimage import filters
from skimage import transform
from skimage.measure import regionprops
from scipy.signal import savgol_filter
from scipy.ndimage.morphology import binary_fill_holes, binary_erosion
from matplotlib.colors import to_rgba
from matplotlib import pyplot as plt
import tensorflow as tf
from tensorflow.python.keras import backend as K
from tensorflow.python.keras.callbacks import (ModelCheckpoint, TensorBoard,
                                               LearningRateScheduler)
from tensorflow.python.keras.models import load_model

from .utils import (get_name, EncodableNamedTuple, find_file,
                    as_python_object, jsonify, schedule_steps, batch_iterator,
                    split_batch_pred, TrainValProperty)
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
from .performance import calc_IoUs, best_IoU, calc_AP
from .tracker.training import CellTrainer, BudTrainer
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


@EncodableNamedTuple
class BabyTrainerParameters(NamedTuple):
    """Global parameters for training BABY models
    """
    train_val_pairs_file: str = 'train_val_pairs.json'
    smoothing_sigma_stats_file: str = 'smoothing_sigma_stats.csv'
    smoothing_sigma_model_file: str = 'smoothing_sigma_model.json'
    flattener_stats_file: str = 'flattener_stats.json'
    flattener_file: str = 'flattener.json'
    mother_bud_props_file: str = 'mother_bud_props.csv'
    mother_bud_model_file: str = 'mother_bud_model.pkl'
    cnn_set: Tuple[str, ...] = ('msd_d80', 'unet_4s')
    cnn_fn: Union[None, str] = None
    batch_size: int = 8
    in_memory: bool = True
    xy_out: int = 80
    target_pixel_size: float = 0.263
    substacks: Union[None, int] = None


class SegExample(NamedTuple):
    """CNN output paired with target segmented outlines and info

    Used for optimising segmentation hyperparameters and for training bud
    assigment models
    """
    pred: np.ndarray
    target: np.ndarray
    info: dict


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
            self._impairs.save(datafile)
            self._ncells = self._impairs.ncells

    @property
    def data(self):
        if not hasattr(self, '_impairs') or not self._impairs:
            self._impairs = TrainValPairs()
            pairs_file = self.save_dir / self.parameters.train_val_pairs_file
            if pairs_file.is_file():
                self._impairs.load(pairs_file)
        self._check_for_data_update()
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
        gt = self.gen.train
        a = gt.aug
        gt.aug = lambda x, y: (x, y)
        sss_train = _generate_smoothing_sigma_stats(gt)
        gt.aug = a

        gv = self.gen.val
        a = gv.aug
        gv.aug = lambda x, y: (x, y)
        sss_val = _generate_smoothing_sigma_stats(gv)
        gv.aug = a

        sss_train['validation'] = False
        sss_val['validation'] = True
        sss = pd.concat((sss_train, sss_val))
        sss_file = self.parameters.smoothing_sigma_stats_file
        sss_file = self.save_dir / sss_file
        sss.to_csv(sss_file)

    @property
    def smoothing_sigma_stats(self):
        if getattr(self, '_sss', None) is None:
            sss_file = self.parameters.smoothing_sigma_stats_file
            sss_file = self.save_dir / sss_file
            if not sss_file.exists():
                raise BadProcess(
                    'smoothing sigma stats have not been generated')
            self._sss = pd.read_csv(sss_file)
        return TrainValProperty(self._sss[~self._sss['validation']],
                                self._sss[self._sss['validation']])

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

    def generate_flattener_stats(self, max_erode=5):
        # Set up temporary flattener
        old_flattener = getattr(self, '_flattener', None)
        self._flattener = lambda x, y: x
        try:
            gt = self.gen.train
            gt.aug = self.aug.train
            fs_train = _generate_flattener_stats(gt, max_erode)
            gv = self.gen.val
            gv.aug = self.aug.val
            fs_val = _generate_flattener_stats(gv, max_erode)
        finally:
            self._flattener = old_flattener

        fs_file = self.parameters.flattener_stats_file
        fs_file = self.save_dir / fs_file
        with open(fs_file, 'wt') as f:
            json.dump({'train': fs_train, 'val': fs_val}, f)
        self._fs = None  # trigger reload of property

    @property
    def flattener_stats(self):
        if getattr(self, '_fs', None) is None:
            fs_file = self.parameters.flattener_stats_file
            fs_file = self.save_dir / fs_file
            if not fs_file.exists():
                raise BadProcess('flattener stats have not been generated')
            with open(fs_file, 'rt') as f:
                self._fs = json.load(f)
        return TrainValProperty(self._fs.get('train', {}),
                                self._fs.get('val', {}))

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
                # Reset any potentially loaded models
                self._cnns = {}
                self._opt_cnn = None
            self.gen.train.aug = self.aug.train
            print('Loading "{}" CNN...'.format(self.cnn_name))
            model = self.cnn_fn(self.gen.train, self.flattener)
            self._cnns[cnn_id] = model

            # Save initial weights if they haven't already been saved
            filename = self.cnn_dir / INIT_WEIGHTS_FILE
            if not filename.exists():
                print('Saving initial weights...')
                model.save_weights(str(filename))
        return self._cnns[cnn_id]

    @property
    def histories(self):
        # Always get the most up-to-date version from disk
        hdict = {}
        active_cnn_fn = getattr(self, '_active_cnn_fn', None)
        try:
            for cnn_id in self.parameters.cnn_set:
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
            self._active_cnn_fn = active_cnn_fn
        return hdict

    @property
    def cnn_opt_dir(self):
        history = min(self.histories.values(),
                      default=None,
                      key=lambda x: min(x['history']['val_loss']))
        if not history:
            raise BadProcess('No trained CNN models found')
        return history['file'].parent

    @property
    def cnn_opt(self):
        if not getattr(self, '_opt_cnn', None):
            opt_dir = self.cnn_opt_dir
            opt_file = opt_dir / OPT_WEIGHTS_FILE
            if not opt_file.exists():
                raise BadProcess(
                    'Optimised CNN for {} model is missing'.format(
                        opt_dir.name))
            self._opt_cnn = load_model(str(opt_file),
                                       custom_objects=custom_objects)
        return self._opt_cnn

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
                    for pred, (_, lbl, info) in zip(preds, batch):
                        pbar.update()
                        yield SegExample(pred, lbl.transpose(2, 0, 1), info)

        if self.in_memory:
            if getattr(self, '_seg_examples', None) is None:
                self._seg_examples = TrainValProperty(
                    list(example_generator(self.gen.train)),
                    list(example_generator(self.gen.val)))
            return TrainValProperty((e for e in self._seg_examples.train),
                                    (e for e in self._seg_examples.val))
        else:
            self._seg_examples = None
            return TrainValProperty(example_generator(self.gen.train),
                                    example_generator(self.gen.val))

    @property
    def track_trainer(self):
        if not hasattr(self, '_track_trainer'):
            self._track_trainer = CellTrainer(self.data._metadata,
                                               data=self.data)
        return self._track_trainer

    @track_trainer.setter
    def track_trainer(self, all_feats2use=None):
        self._track_trainer = CellTrainer(self.data._metadata,
                                           data=self.data,
                                           all_feats2use=all_feats2use)

    @property
    def bud_trainer(self):
        props_file = self.save_dir / self.parameters.mother_bud_props_file
        if not hasattr(self, '_bud_trainer'):
            self._bud_trainer = BudTrainer(
                props_file=props_file,
                px_size=self.parameters.target_pixel_size)
        return self._bud_trainer

    def generate_bud_stats(self):
        self.bud_trainer.generate_property_table(self.seg_examples,
                                                 self.flattener)

    def _get_grouped_sss(self):
        group_best_iou = lambda x: x.loc[x['iou'].idxmax(), :]
        idcols = ['ind', 'cell', 'scaling', 'rotation']
        stats = self.smoothing_sigma_stats.train
        stats = stats.groupby(idcols).apply(group_best_iou)
        filts = {
            'identity': (stats.scaling == 1) & (stats.rotation == 0),
            'scaling': stats.scaling != 1,
            'rotation': stats.rotation != 0
        }
        return stats, filts

    def fit_smoothing_model(self, filt='identity'):
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

        self.smoothing_sigma_model = SmoothingSigmaModel(*params)

    def plot_fitted_smoothing_sigma_model(self):
        stats, filts = self._get_grouped_sss()
        ssm = self.smoothing_sigma_model
        model = lambda s, a, p, b: b + a * np.exp(p * s)
        params = (ssm._a, ssm._b, ssm._c)

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
            ax.plot(nedge, [ssm(n) for n in nedge], 'r')
            ax.set(title=k.title(),
                   xlabel='nedge',
                   ylabel='sigma',
                   ylim=[0, sigma_max])

        fig.savefig(self.save_dir / 'fitted_smoothing_sigma_model.png')

    def fit_flattener(self,
                      nbins=30,
                      min_size=10,
                      pad_frac=0.03,
                      bud_max=200):
        if pad_frac > 0.25 or pad_frac < 0:
            raise BadParam('"pad_frac" must be between 0 and 0.2')

        # Find the best split
        overlapping = self.flattener_stats.train.get('overlap_sizes', [])
        erosion_sizes = self.flattener_stats.train.get('erosion_sizes', [])
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

        ogL, ogH = zip(
            *[_group_overlapping(o, split0, pad=pad) for o in overlapping])
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
        flattener.addGroup('medium',
                           lower=int(np.round(splits[0] - pad)),
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

        p = self.parameters
        flattener_file = self.save_dir / p.flattener_file
        flattener.save(flattener_file)
        self._flattener = None

    def plot_flattener_stats(self, nbins=30):
        overlapping = self.flattener_stats.train.get('overlap_sizes', [])
        max_erode = len(overlapping)
        fig, axs = plt.subplots(1, max_erode, figsize=(16, 16 / max_erode))
        x, y, _ = zip(*overlapping[0])
        max_size = max(x + y)
        for ax, (e, os) in zip(axs, enumerate(overlapping)):
            if len(os) > 0:
                x, y, w = zip(*os)
            else:
                x, y, w = 3 * [[]]
            ax.hist2d(x,
                      y,
                      bins=30,
                      weights=w,
                      range=[[0, max_size], [0, max_size]])
            ax.plot((0, max_size), (0, max_size), 'r')
            ax.set_title('nerosions = {:d}'.format(e))
        fig.savefig(self.save_dir / 'flattener_stats.png')

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

    def plot_histories(self,
                       key='loss',
                       log=True,
                       window=21,
                       ax=None,
                       save=True,
                       legend=True):
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

    def fit_bud_model(self):
        self.bud_trainer.explore_hyperparams()
        model_file = self.save_dir / self.parameters.mother_bud_model_file
        self.bud_trainer.save_model(model_file)

    def fit_seg_params(self):
        # interior_threshold: threshold on predicted interior
        # nclosing: number of closing operations on threshold mask
        # nopening: number of opening operations on threshold mask
        #
        default_params = {
                'interior_threshold': [0.8, 0.8, 0.8],
                'nclosing': [2, 2, 2],
                'nopening': [0, 0, 0],
                'connectivity': [2,2,2],
                'edge_sub_dilations': [1, 1, 1],
                'pedge_thresh': None, 'fit_radial': True,
                'use_group_thresh': True, 'group_thresh_expansion': 0.1,
                }
        params = {
                'group_thresh_expansion': [-1,0,0.1,0.2], # -1 -> no thresh
                'nclosing': [0,1,2], 'nopening': [0,1,2],
                'interior_threshold': np.arange(0.5, 1.0, 0.1).tolist()
                }

        from joblib import Parallel, delayed
        import itertools
        param_grid = list(itertools.product(*params.values()))

        param_pred_segs = {}
        for gind in range(3)[::-1]:
            for pars in param_grid:
                key = 'G({}), NC({:d}), NO({:d}), IT({:.2f})'.format(gind, *pars)
                print(key)
                new_params = {k: v.copy() if type(v) == list else v for k, v in default_params.items()}
                for k, v in zip(params.keys(), pars):
                    new_params[k][gind] = v
                param_pred_segs[key] =  Parallel(n_jobs=4, prefer='threads')(
                        delayed(morph_seg_grouped)(o, flattener, **new_params)
                        for o in val_pred
                        )

        param_IoU_mats = {k: [calc_IoUs(t, p) for t, p in zip(val_truth, psegs)]
                          for k, psegs in param_pred_segs.items()}
        param_edge_probs = {
            k: [np.array([np.dstack(pred)[..., [1, 3, 6]].max(axis=2)[s].mean() for s in segs])
                for pred, segs in zip(val_pred, psegs)]
            for k, psegs in param_pred_segs.items()
        }

        param_mIoUs = pd.DataFrame({k: [np.mean(best_IoU(i)[0]) for i in IoU_mats]
                                    for k, IoU_mats in param_IoU_mats.items()})
        param_minIoUs = pd.DataFrame({k: [np.min(best_IoU(i)[0], initial=1) for i in IoU_mats]
                                      for k, IoU_mats in param_IoU_mats.items()})
        param_mAPs = pd.DataFrame({k: [calc_AP(i, probs=p, iou_thresh=0.7)[0]
                                       for i, p in zip(IoU_mats, param_edge_probs[k])]
                                   for k, IoU_mats in param_IoU_mats.items()})


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
        (x, y, w) for x, y, w in os if x < thresh + pad and y < thresh + pad
    ], [(x, y, w) for x, y, w in os if x >= thresh - pad and y >= thresh - pad
       ])


def _best_overlapping(overlapping, erosion_sizes, min_size):
    sz_erosions = list(zip(*erosion_sizes))
    e_invalid = [any([c < min_size for c in e]) for e in sz_erosions[:0:-1]]
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
