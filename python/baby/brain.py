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
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
from pathlib import Path
import json
from itertools import repeat, chain

import numpy as np
import pickle

import tensorflow as tf
from tensorflow.keras import models, layers
from tensorflow.keras import backend as K

try:
    from tensorflow.keras.optimizers import AdamW
except ImportError:
    try:
        from tensorflow.keras.optimizers.experimental import AdamW
    except ImportError:
        try:
            from tensorflow_addons.optimizers import AdamW
        except ImportError:
            raise ImportError('You need to pip install tensorflow-addons with this version of tensorflow')

from skimage import transform

from . import modelsets
from .losses import bce_dice_loss, dice_loss, dice_coeff
from .preprocessing import SegmentationFlattening
from .morph_thresh_seg import MorphSegGrouped, SegmentationParameters
from .tracker.core import MasterTracker, MMTracker
from .utils import batch_iterator, split_batch_pred, as_python_object
from .brain_util import (_segment, _track, _segment_and_track_parallel,
                         _apply_preprocessing, _rescale_output,
                         _tile_generator, _stitch_tiles, _batch_generator)
from .errors import BadFile, BadParam

tf_version = [int(v) for v in tf.__version__.split('.')]


DEFAULT_MODELSET = 'yeast-alcatras-brightfield-EMCCD-60x-5z'
DEFAULT_MODELSET_PARAMS = None


def _default_params():
    global DEFAULT_MODELSET_PARAMS
    if DEFAULT_MODELSET_PARAMS is None:
        DEFAULT_MODELSET_PARAMS = modelsets.get_params(DEFAULT_MODELSET)
    return DEFAULT_MODELSET_PARAMS


class BabyBrain(object):
    '''Coordinator for the segmentation and tracking models

    Models are specified by file name, which can either be the path to a model
    file or the name of the file in the default "models" dir shipped with this
    package. If any are left unspecified, then default models will be loaded.

    Args:
        morph_model_file: neural network model taking a stack of images and
            outputting predictions according to the paired flattener model.
        flattener_file: a saved `SegmentationFlattening` model specifying
            the trained target types for each output layer of the neural network.
        celltrack_model_file: file name of the saved machine learning model
            for default tracking predictions.
        celltrack_backup_model_file: file name of the saved machine learning
            model for backup tracking predictions.
        budassign_model_file: file name of the saved machine learning model
            for predicting bud assignments
        pixel_size (float): Target pixel size for inputs to the trained CNN.
        default_image_size (None or Tuple[int] or int): Optionally specify an
            alternative to the input size of the trained CNN as a ``(W, H)``
            tuple giving the ``W`` and ``H`` of the image. If just an ``int``
            is specified, then ``W = H`` is assumed.
        params (str or Path or SegmentationParameters): Segmentation
            parameters to use with :py:class:`MorphSegGrouped`. May be
            specified as :py:class:`SegmentationParameters`, or the path to a
            saved :py:class:`SegmentationParameters`.
        modelset_path (str or Path): path to a folder containing the files
            specified by the other arguments. The shared folder of the local
            model set cache is always checked if file is not found at
            `modelset_path`. See :py:func:`modelsets.resolve` for details on
            how paths to model files are resolved.
        session: optionally specify the Tensorflow session to load the
            neural network model into (useful only for Tensorflow versions <2)
        graph: optionally specify the Tensorflow graph to load the neural
            network model into (useful only for Tensorflow versions <2)
        suppress_errors: whether or not to catch Exceptions raised during
            segmentation or tracking. If True, then any Exceptions will be logged
            using standard Python logging.
        error_dump_dir: optionally specify a directory in which to dump
            input parameters when an error is caught.
    '''

    def __init__(self,
                 morph_model_file=None,
                 flattener_file=None,
                 celltrack_model_file=None,
                 celltrack_backup_model_file=None,
                 budassign_model_file=None,
                 mmtracking=False,
                 pixel_size=0.263,
                 default_image_size=None,
                 params=SegmentationParameters(),
                 nstepsback=None,
                 clogging_thresh=0.75,
                 min_bud_tps=3,
                 isbud_thresh=0.5,
                 input_norm_dw=False,
                 modelset_path=DEFAULT_MODELSET,
                 session=None,
                 graph=None,
                 print_info=False,
                 suppress_errors=False,
                 error_dump_dir=None):

        self.reshaped_models = {}

        if morph_model_file is None:
            morph_model_file = _default_params()['morph_model_file']
        morph_model_file = modelsets.resolve(morph_model_file, modelset_path)

        if flattener_file is None:
            flattener_file = _default_params()['flattener_file']
        flattener_file = modelsets.resolve(flattener_file, modelset_path)

        if type(params) == dict:
            params = SegmentationParameters(**params)
        if type(params) != SegmentationParameters:
            param_file = modelsets.resolve(params, modelset_path)
            with open(param_file, 'rt') as f:
                params = json.load(f, object_hook=as_python_object)
            if type(params) != SegmentationParameters:
                raise BadFile('Specified file does not contain a valid '
                              '`SegmentationParameters` object')

        if not params.fit_radial:
            raise BadParam('`BabyBrain` currently only works for '
                           '`SegmentationParameters` with `fit_radial=True`')

        self.params = params

        self.session = None
        self.graph = None
        if tf_version[0] == 1:
            if session is None:
                session = K.get_session()
            if graph is None:
                graph = tf.get_default_graph()
            self.session = session
            self.graph = graph

        self.print_info = print_info
        if self.print_info:
            print('Loading Keras model "{}"...'.format(morph_model_file))
        if tf_version[0] == 1:
            with self.graph.as_default():
                K.set_session(session)
                if self.print_info:
                    print('Loading model into session "{}"...'.format(
                        K.get_session()))
                self.morph_model = models.load_model(
                    morph_model_file,
                    custom_objects={
                        'bce_dice_loss': bce_dice_loss,
                        'dice_loss': dice_loss,
                        'dice_coeff': dice_coeff,
                        'AdamW': AdamW
                    })
        else:
            self.morph_model = models.load_model(
                morph_model_file,
                custom_objects={
                    'bce_dice_loss': bce_dice_loss,
                    'dice_loss': dice_loss,
                    'dice_coeff': dice_coeff
                })
            if tf_version[0] == 2 and tf_version[1] > 3:
                # TF 2.4 no longer supports reshaping using the functional API
                # on a new Input since it allows multiple inputs to a layer
                # Recommendation is to define generic Input layers with None
                # for variable dimensions, or to disable input checking with
                # the following code.
                # See release notes for version 2.4.0 at
                # https://github.com/tensorflow/tensorflow/blob/master/RELEASE.md
                self.morph_model.input_spec = None

        self.suppress_errors = suppress_errors
        self.error_dump_dir = error_dump_dir

        self.input_norm_dw = input_norm_dw
        self.flattener = SegmentationFlattening(flattener_file)

        self.morph_segmenter = MorphSegGrouped(self.flattener,
                                               params=params,
                                               return_masks=True,
                                               return_coords=True)

        self.pixel_size = pixel_size
        self.clogging_thresh = clogging_thresh

        if mmtracking:
            # Set the tracker engine to the model-free Mother Machine variant
            self.tracker = MMTracker(px_size=pixel_size)
        else:
            # Load tracker models and initialise Tracker
            if celltrack_model_file is None:
                celltrack_model_file = _default_params()['celltrack_model_file']
            celltrack_model_file = modelsets.resolve(celltrack_model_file, modelset_path)

            if celltrack_backup_model_file is None:
                celltrack_backup_model_file = _default_params()[
                    'celltrack_backup_model_file']
            celltrack_backup_model_file = modelsets.resolve(
                celltrack_backup_model_file, modelset_path)

            if budassign_model_file is None:
                budassign_model_file = _default_params()['budassign_model_file']
            budassign_model_file = modelsets.resolve(budassign_model_file, modelset_path)

            ctrack_args ={
                'model': celltrack_model_file,
                'bak_model': celltrack_backup_model_file,
                'nstepsback': nstepsback
            }
            btrack_args = {'model': budassign_model_file}
            self.tracker = MasterTracker(ctrack_args=ctrack_args,
                                         btrack_args=btrack_args,
                                         min_bud_tps=min_bud_tps,
                                         isbud_thresh=isbud_thresh,
                                         px_size=pixel_size)

        # Run prediction on mock image to load model for prediction
        _, x, y, z = self.morph_model.input.shape

        if default_image_size is not None:
            try:
                x, y = default_image_size
            except TypeError:
                x = default_image_size
                y = x

        self.morph_predict(np.zeros((1, x, y, z)))

    @property
    def depth(self):
        return self.morph_model.input.shape[3]

    def _predict(self, X, needs_context=True):
        if tf_version[0] == 1 and needs_context:
            with self.graph.as_default():
                K.set_session(self.session)
                return self._predict(X, needs_context=False)

        imdims = X.shape[1:3]
        # Current MSD model requires shape to be divisible by 8. Standard 5
        # layer U-Net requires shape to be divisible by 16.
        # CNN model for FRET is very sensitive to padding: we need images to
        # be at least 64px wide, the padding needs to be centred and the pad
        # value needs to be the image median. This basically makes the
        # following match the padding protocol in the Augmenter
        nndims = tuple(max(64, int(np.ceil(float(d) / 16.)) * 16)
                       for d in imdims)
        xpadoff, ypadoff = 0, 0
        if not all([n == i for n, i in zip(nndims, imdims)]):
            xpad, ypad = tuple(n - i for n, i in zip(nndims, imdims))
            xpadoff, ypadoff = xpad // 2, ypad // 2
            xpad = (xpadoff, xpad - xpadoff)
            ypad = (ypadoff, ypad - ypadoff)
            X = np.array([np.pad(Xi, (xpad, ypad, (0, 0)),
                                 mode='constant',
                                 constant_values=np.median(Xi))
                          for Xi in X])

        if nndims not in self.reshaped_models:
            i = layers.Input(shape=X.shape[1:])
            self.reshaped_models[nndims] = models.Model(
                i, self.morph_model(i))

        if tf_version[0] == 1 and self.print_info:
            print('Running prediction in session "{}"...'.format(
                K.get_session()))

        pred = self.reshaped_models[nndims].predict(X, verbose=0)

        return [p[:, xpadoff:xpadoff+imdims[0], ypadoff:ypadoff+imdims[1], :]
                for p in pred]

    def morph_predict(self, X, pixel_size=None, overlap_size=48,
                      yield_rescaling=False, keep_bb_pixel_size=False):
        # First preprocess each brightfield image in batch
        X, rescaling, inshape = _apply_preprocessing(
            X, self.input_norm_dw, pixel_size, self.pixel_size)

        if yield_rescaling:
            yield rescaling, inshape

        tilegen = _tile_generator(X, overlap_size=overlap_size)
        tiling_strategy = next(tilegen)
        predgen = chain(*map(lambda x: split_batch_pred(self._predict(x)),
                             _batch_generator(tilegen, 8)))
        
        for pred in _stitch_tiles(predgen, *tiling_strategy):
            if keep_bb_pixel_size:
                yield pred
            else:
                yield transform.resize(
                    pred, (len(pred),) + inshape, order=1)

    def segment(self,
                bf_img_batch,
                pixel_size=None,
                overlap_size=48,
                yield_edgemasks=False,
                yield_masks=False,
                yield_preds=False,
                yield_volumes=False,
                refine_outlines=False,
                yield_rescaling=False,
                keep_bb_pixel_size=False):
        '''Generator yielding segmented output for a batch of input images

        :param bf_img_batch: a list of ndarray with shape (X, Y, Z), or
            equivalently an ndarray with shape (N_images, X, Y, Z)
        :param yield_masks: if set to True, additionally yield filled cell
            masks in each output dict
        :param yield_edgemasks: if set to True, additionally yield edge masks
            in each output dict
        :param yield_preds: if set to True, additionally yield full prediction
            output from neural network

        :yields: for each image in `bf_img_batch` a dict with
            - centres: list of float pairs corresponding to (x, y) coords for
              each detected cell,
            - angles: list of lists of floats corresponding, for each cell, to
              angles (radians) used to form active contour outline in radial
              space
            - radii: list of lists of floats corresponding, for each cell, to
              radii used to form active contour outline in radial space
            - edgemasks: (optional) an ndarray of dtype "bool" with shape
              (N_cells, X, Y) specifying the rasterised edge for each
              segmented cell
            - volumes: (optional) list of floats corresponding, for each cell,
              to the conical section method for cell volume estimation 
        '''

        predgen = self.morph_predict(bf_img_batch,
                                     pixel_size=pixel_size,
                                     overlap_size=overlap_size,
                                     yield_rescaling=True,
                                     keep_bb_pixel_size=True)
        rescaling, inshape = next(predgen)
        if yield_rescaling:
            yield rescaling, inshape

        for cnn_output in predgen:
            segout = _segment(self.morph_segmenter, cnn_output,
                              refine_outlines, yield_volumes, yield_masks,
                              yield_preds, yield_edgemasks,
                              self.clogging_thresh, self.error_dump_dir,
                              self.suppress_errors)
            if not keep_bb_pixel_size:
                _rescale_output(segout, rescaling, inshape,
                                self.morph_segmenter.params.cartesian_spline)
            yield segout

    def run(self, bf_img_batch):
        '''Implementation of legacy runner function...
        '''
        output = []

        tnames = self.flattener.names()
        i_budneck = tnames.index('bud_neck')
        bud_target = 'sml_fill' if 'sml_fill' in tnames else 'sml_inte'
        i_bud = tnames.index(bud_target)

        for seg_output in self.segment(bf_img_batch,
                                       yield_masks=True,
                                       yield_preds=True):
            ba_probs = self.tracker.calc_mother_bud_stats(
                seg_output['preds'][i_budneck], seg['preds'][i_bud],
                seg_output['masks'])

            del seg_output['preds']
            del seg_output['masks']

            seg_output['ba_probs'] = ba_probs
            output.append(seg_output)

        return output

    def segment_and_track(self,
                          bf_img_batch,
                          tracker_states=None,
                          pixel_size=None,
                          yield_next=False,
                          yield_edgemasks=False,
                          assign_mothers=False,
                          return_baprobs=False,
                          refine_outlines=False,
                          yield_volumes=False):
        '''Generator yielding segmented and tracked output for a batch of input
        images

        :param bf_img_batch: a list of ndarray with shape (X, Y, Z), or
            equivalently an ndarray with shape (N_images, X, Y, Z)
        :param tracker_states: a generator of tracker states from the previous
            time point as yielded by this function when `yield_next` is True
        :param yield_next: whether to yield updated tracking states for
            subsequent calls to this function
        :param yield_edgemasks: whether to include edge masks in the output
        :param assign_mothers: whether to include mother assignments in the
            output
        :param return_baprobs: whether to include the bud assignment
            probability matrix in the output
        :param yield_volumes: whether to calculate and include volume
            estimates in the output

        :yields: for each image in `bf_img_batch` a dict with
            - centres: list of float pairs corresponding to (x, y) coords for
              each detected cell,
            - angles: list of lists of floats corresponding, for each cell, to
              angles (radians) used to form active contour outline in radial
              space
            - radii: list of lists of floats corresponding, for each cell, to
              radii used to form active contour outline in radial space
            - cell_label: list of int corresponding to tracked global ID for
              each cell detected in this image (indexed from 1)
            - mother_assign: (optional) list of int specifying for each
              (global) cell label ID, the cell label ID of the corresponding
              mother (0 if no mother was found)
            - p_bud_assign: (optional) matrix as a list of lists of floats,
              specifying the probability that a cell (outer list) is a mother
              to another cell (inner lists) in this image
            - edgemasks: (optional) an ndarray of dtype "bool" with shape
              (N_cells, X, Y) specifying the rasterised edge for each
              segmented cell

            If `yield_next` is True, yields the dict described above and
            tracker states for this time point as a tuple
        '''

        if tracker_states is None:
            tracker_states = repeat(None)

        tnames = self.flattener.names()
        bud_target = 'sml_fill' if 'sml_fill' in tnames else 'sml_inte'
        assign_buds = 'bud_neck' in tnames and bud_target in tnames
        if assign_buds:
            i_budneck = tnames.index('bud_neck')
            i_bud = tnames.index(bud_target)
        else:
            i_budneck = None
            i_bud = None

        segment_gen = self.segment(bf_img_batch,
                                   pixel_size=pixel_size,
                                   yield_masks=True,
                                   yield_edgemasks=True,
                                   yield_preds=True,
                                   yield_volumes=yield_volumes,
                                   refine_outlines=refine_outlines,
                                   yield_rescaling=True,
                                   keep_bb_pixel_size=True)

        rescaling, inshape = next(segment_gen)

        for seg, state in zip(segment_gen, tracker_states):
            trackout = _track(self.tracker, seg, state, i_budneck, i_bud,
                         assign_mothers, return_baprobs, yield_edgemasks,
                         yield_next, self.error_dump_dir,
                         self.suppress_errors)
            if yield_next:
                segout, state = trackout
                _rescale_output(segout, rescaling, inshape,
                                self.morph_segmenter.params.cartesian_spline)
                yield segout, state
            else:
                _rescale_output(trackout, rescaling, inshape,
                                self.morph_segmenter.params.cartesian_spline)
                yield trackout

    def segment_and_track_parallel(self,
                                   bf_img_batch,
                                   tracker_states=None,
                                   pixel_size=None,
                                   yield_next=False,
                                   yield_edgemasks=False,
                                   assign_mothers=False,
                                   return_baprobs=False,
                                   refine_outlines=False,
                                   yield_volumes=False,
                                   njobs=-2):
        '''Segment and track a batch of input images using joblib

        :param bf_img_batch: a list of ndarray with shape (X, Y, Z), or
            equivalently an ndarray with shape (N_images, X, Y, Z)
        :param tracker_states: a generator of tracker states from the previous
            time point as yielded by this function when `yield_next` is True
        :param yield_next: whether to yield updated tracking states for
            subsequent calls to this function
        :param yield_edgemasks: whether to include edge masks in the output
        :param assign_mothers: whether to include mother assignments in the
            output
        :param return_baprobs: whether to include the bud assignment
            probability matrix in the output
        :param yield_volumes: whether to calculate and include volume
            estimates in the output

        :returns: a list containing for each image in `bf_img_batch` a dict with
            - centres: list of float pairs corresponding to (x, y) coords for
              each detected cell,
            - angles: list of lists of floats corresponding, for each cell, to
              angles (radians) used to form active contour outline in radial
              space
            - radii: list of lists of floats corresponding, for each cell, to
              radii used to form active contour outline in radial space
            - cell_label: list of int corresponding to tracked global ID for
              each cell detected in this image (indexed from 1)
            - mother_assign: (optional) list of int specifying for each
              (global) cell label ID, the cell label ID of the corresponding
              mother (0 if no mother was found)
            - p_bud_assign: (optional) matrix as a list of lists of floats,
              specifying the probability that a cell (outer list) is a mother
              to another cell (inner lists) in this image
            - edgemasks: (optional) an ndarray of dtype "bool" with shape
              (N_cells, X, Y) specifying the rasterised edge for each
              segmented cell

            If `yield_next` is True, returns the dict described above and
            tracker states for this time point as a tuple
        '''

        # Do not run the CNN in parallel
        predgen = self.morph_predict(bf_img_batch,
                                     pixel_size=pixel_size,
                                     yield_rescaling=True,
                                     keep_bb_pixel_size=True)
        rescaling, inshape = next(predgen)
        preds = list(predgen)

        if tracker_states is None:
            tracker_states = repeat(None)

        trackout = _segment_and_track_parallel(
            self.morph_segmenter, self.tracker, self.flattener, preds,
            tracker_states, rescaling, inshape, refine_outlines,
            yield_volumes, yield_edgemasks, self.clogging_thresh,
            assign_mothers, return_baprobs, yield_next, njobs,
            self.error_dump_dir, self.suppress_errors)
        return trackout
