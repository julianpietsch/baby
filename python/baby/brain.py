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
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
from os.path import dirname, join, isfile, isdir
from itertools import repeat, chain

import numpy as np
import pickle

import tensorflow as tf
from tensorflow.keras import models, layers
from tensorflow.keras import backend as K

from .losses import bce_dice_loss, dice_loss, dice_coeff
from .preprocessing import robust_norm, SegmentationFlattening
from .morph_thresh_seg import MorphSegGrouped
from .tracker.core import MasterTracker
from .utils import batch_iterator, split_batch_pred
from .brain_util import _segment, _track, _segment_and_track_parallel

models_path = join(dirname(__file__), 'models')

tf_version = [int(v) for v in tf.__version__.split('.')]

# Default to optimal segmentation parameters found in Jupyter notebook
# segmentation-190906.ipynb:
default_params = {
    'interior_threshold': (0.7, 0.5, 0.5),
    'nclosing': (1, 0, 0),
    'nopening': (1, 0, 0),
    'connectivity': (2, 2, 1),
    'pedge_thresh': 0.001,
    'fit_radial': True,
    'edge_sub_dilations': 1,
    'use_group_thresh': True,
    'group_thresh_expansion': 0.1
}


class BabyBrain(object):
    '''Coordinator for the segmentation and tracking models

    Models are specified by file name, which can either be the path to a model
    file or the name of the file in the default "models" dir shipped with this
    package. If any are left unspecified, then default models will be loaded.

    :param morph_model_file: neural network model taking a stack of images and
        outputting predictions according to the paired flattener model.
    :param flattener_file: a saved `SegmentationFlattening` model specifying
        the trained target types for each output layer of the neural network.
    :param celltrack_model_file:
    :param budassign_model_file:
    :param default_image_size:
    :param params: dict of keyword parameters to be passed to the
        `morph_seg_grouped` function when segmenting.
    :param session: optionally specify the Tensorflow session to load the
        neural network model into (useful only for Tensorflow versions <2)
    :param graph: optionally specify the Tensorflow graph to load the neural
        network model into (useful only for Tensorflow versions <2)
    :param suppress_errors: whether or not to catch Exceptions raised during
        segmentation or tracking. If True, then any Exceptions will be logged
        using standard Python logging.
    :param error_dump_dir: optionally specify a directory in which to dump
        input parameters when an error is caught.
    '''

    def __init__(self,
                 morph_model_file=None,
                 flattener_file=None,
                 celltrack_model_file=None,
                 budassign_model_file=None,
                 pixel_size=0.263,
                 default_image_size=None,
                 params=default_params,
                 clogging_thresh=0.75,
                 min_bud_tps=3,
                 isbud_thresh=0.5,
                 session=None,
                 graph=None,
                 print_info=False,
                 suppress_errors=False,
                 error_dump_dir=None):

        self.reshaped_models = {}

        if morph_model_file is None:
            morph_model_file = join(models_path, 'I5_msd_d80_20190916.hdf5')
        elif not isfile(morph_model_file):
            morph_model_file = join(models_path, morph_model_file)

        if flattener_file is None:
            flattener_file = join(models_path, 'flattener_v2_20190905.json')
        elif not isfile(flattener_file):
            flattener_file = join(models_path, flattener_file)

        if celltrack_model_file is None:
            celltrack_model_file = join(models_path, 'ct_svc_20201106_6.pkl')
        elif not isfile(celltrack_model_file):
            celltrack_model_file = join(models_path, celltrack_model_file)

        if budassign_model_file is None:
            budassign_model_file = join(models_path,
                                        'baby_randomforest_20190906.pkl')
        elif not isfile(budassign_model_file):
            budassign_model_file = join(models_path, budassign_model_file)

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
                        'dice_coeff': dice_coeff
                    })
        else:
            self.morph_model = models.load_model(
                morph_model_file,
                custom_objects={
                    'bce_dice_loss': bce_dice_loss,
                    'dice_loss': dice_loss,
                    'dice_coeff': dice_coeff
                })

        self.suppress_errors = suppress_errors
        self.error_dump_dir = error_dump_dir

        self.flattener = SegmentationFlattening(flattener_file)
        self.params = params

        if ('use_group_thresh' not in self.params and
                'group_thresh_expansion' in self.params):
            self.params['use_group_thresh'] = True
        self.morph_segmenter = MorphSegGrouped(self.flattener,
                                               fit_radial=True,
                                               return_masks=True,
                                               return_coords=True,
                                               **self.params)

        self.pixel_size = pixel_size
        self.clogging_thresh = clogging_thresh

        # Load tracker models and initialise Tracker
        with open(celltrack_model_file, 'rb') as f:
            celltrack_model = pickle.load(f)
        with open(budassign_model_file, 'rb') as f:
            budassign_model = pickle.load(f)
        self.tracker = MasterTracker(ctrack_args={'model': celltrack_model},
                                     btrack_args={'model': budassign_model},
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

    def morph_predict(self, X, needs_context=True):
        if tf_version[0] == 1 and needs_context:
            with self.graph.as_default():
                K.set_session(self.session)
                return self.morph_predict(X, needs_context=False)

        imdims = X.shape[1:3]
        # Current MSD model requires shape to be divisible by 8
        nndims = tuple([int(np.ceil(float(d) / 8.)) * 8 for d in imdims])
        if not all([n == i for n, i in zip(nndims, imdims)]):
            xpad, ypad = tuple(n - i for n, i in zip(nndims, imdims))
            X = np.pad(X, ((0, 0), (0, xpad), (0, ypad), (0, 0)), 'edge')

        if nndims not in self.reshaped_models:
            base_input_shape = self.morph_model.input.shape[1:3]
            if all([n == m for n, m in zip(nndims, base_input_shape)]):
                self.reshaped_models[nndims] = self.morph_model
            else:
                i = layers.Input(shape=X.shape[1:])
                self.reshaped_models[nndims] = models.Model(
                    i, self.morph_model(i))

        if tf_version[0] == 1 and self.print_info:
            print('Running prediction in session "{}"...'.format(
                K.get_session()))

        pred = self.reshaped_models[nndims].predict(X)

        return [p[:, :imdims[0], :imdims[1], :] for p in pred]

    def segment(self,
                bf_img_batch,
                yield_edgemasks=False,
                yield_masks=False,
                yield_preds=False,
                yield_volumes=False,
                refine_outlines=False):
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
        # First preprocess each brightfield image in batch
        bf_img_batch = np.stack(
            [robust_norm(img, {}) for img in bf_img_batch])

        for batch in batch_iterator(bf_img_batch):
            morph_preds = split_batch_pred(self.morph_predict(batch))

            for cnn_output in morph_preds:
                yield _segment(self.morph_segmenter, cnn_output,
                               refine_outlines, yield_volumes, yield_masks,
                               yield_preds, yield_edgemasks,
                               self.clogging_thresh, self.error_dump_dir,
                               self.suppress_errors)

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
        i_budneck = tnames.index('bud_neck')
        bud_target = 'sml_fill' if 'sml_fill' in tnames else 'sml_inte'
        i_bud = tnames.index(bud_target)

        segment_gen = self.segment(bf_img_batch,
                                   yield_masks=True,
                                   yield_edgemasks=True,
                                   yield_preds=True,
                                   yield_volumes=yield_volumes,
                                   refine_outlines=refine_outlines)

        for seg, state in zip(segment_gen, tracker_states):
            yield _track(self.tracker, seg, state, i_budneck, i_bud,
                         assign_mothers, return_baprobs, yield_edgemasks,
                         yield_next, self.error_dump_dir,
                         self.suppress_errors)

    def segment_and_track_parallel(self,
                                   bf_img_batch,
                                   tracker_states=None,
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

        # First preprocess each brightfield image in batch
        bf_img_batch = np.stack(
            [robust_norm(img, {}) for img in bf_img_batch])

        # Do not run the CNN in parallel
        morph_preds = list(
            chain(*(split_batch_pred(self.morph_predict(batch))
                    for batch in batch_iterator(bf_img_batch))))

        if tracker_states is None:
            tracker_states = repeat(None)

        trackout = _segment_and_track_parallel(
            self.morph_segmenter, self.tracker, self.flattener, morph_preds,
            tracker_states, refine_outlines, yield_volumes, yield_edgemasks,
            self.clogging_thresh, assign_mothers, return_baprobs, yield_next,
            njobs, self.error_dump_dir, self.suppress_errors)
        return trackout
