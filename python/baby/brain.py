from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
from os.path import dirname, join, isfile, isdir
from itertools import repeat
import logging
from time import strftime
from uuid import uuid1

import numpy as np
import pickle

import tensorflow as tf
from tensorflow.keras import models, layers
from tensorflow.keras import backend as K

from .losses import bce_dice_loss, dice_loss, dice_coeff
from .segmentation import morph_seg_grouped
from .tracker.core import MasterTracker
from .preprocessing import robust_norm, SegmentationFlattening
from .utils import batch_iterator, split_batch_pred
from .morph_thresh_seg import MorphSegGrouped, SegmentationOutput
from .io import save_tiled_image

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
            celltrack_model_file = join(models_path,
                                        'ctrack_randomforest_20200513.pkl')
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

        self.morph_segmenter = MorphSegGrouped(self.flattener,
                                               return_masks=True,
                                               return_coords=True,
                                               **self.params)

        self.pixel_size = pixel_size

        # Load tracker models and initialise Tracker
        with open(celltrack_model_file, 'rb') as f:
            celltrack_model = pickle.load(f)
        with open(budassign_model_file, 'rb') as f:
            budassign_model = pickle.load(f)
        self.tracker = MasterTracker(
            ctrack_args={'model': celltrack_model},
            btrack_args={'model': budassign_model},
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
                output = {}

                try:
                    seg_result = self.morph_segmenter.segment(
                        cnn_output,
                        refine_outlines=refine_outlines,
                        return_volume=yield_volumes)
                except:
                    # Log errors
                    err_id = _generate_error_dump_id()
                    if (self.error_dump_dir is not None and
                            isdir(self.error_dump_dir)):
                        save_tiled_image(
                            np.uint16((2**16 - 1) * cnn_output),
                            join(self.error_dump_dir,
                                 err_id + '_cnn_out.png'))
                    if self.suppress_errors:
                        err_msg = dict(ID=err_id,
                                       refine_outlines=refine_outlines,
                                       return_volume=yield_volumes)
                        err_msg = [
                            '{}: {}'.format(k, v) for k, v in err_msg.items()
                        ]
                        err_msg.insert(0, 'Segmentation error')
                        logging.exception('\n'.join(err_msg))
                        seg_result = SegmentationOutput([])
                        output['error'] = 'Segmentation error: ' + err_id
                    else:
                        raise

                if len(seg_result.coords) > 0:
                    centres, radii, angles = zip(*seg_result.coords)
                else:
                    centres, radii, angles = 3 * [[]]

                # Return output as a dict
                output.update({
                    'centres': centres,
                    'angles': [a.tolist() for a in angles],
                    'radii': [r.tolist() for r in radii]
                })

                _0xy = (0,) + cnn_output.shape[1:3]
                if yield_masks:
                    if len(seg_result.masks) > 0:
                        output['masks'] = np.stack(seg_result.masks)
                    else:
                        output['masks'] = np.zeros(_0xy, dtype='bool')
                if yield_edgemasks:
                    if len(seg_result.edges) > 0:
                        output['edgemasks'] = np.stack(seg_result.edges)
                    else:
                        output['edgemasks'] = np.zeros(_0xy, dtype='bool')
                if yield_preds:
                    output['preds'] = cnn_output
                if yield_volumes:
                    output['volumes'] = seg_result.volume

                yield output

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
            try:
                tracking = self.tracker.step_trackers(
                    seg['masks'],
                    seg['preds'][i_budneck],
                    seg['preds'][i_bud],
                    state=state,
                    assign_mothers=assign_mothers,
                    return_baprobs=return_baprobs)
            except:
                # Log errors
                err_id = _generate_error_dump_id()
                if (self.error_dump_dir is not None and
                        isdir(self.error_dump_dir)):
                    fprefix = join(self.error_dump_dir, err_id)
                    masks = seg.get('masks', np.zeros((0, 0, 0)))
                    if masks.size > 0:
                        save_tiled_image(
                            seg['masks'].transpose((2, 0, 1)).astype('uint8'),
                            fprefix + '_masks.png')
                    save_tiled_image(
                        np.uint16((2**16 - 1) *
                                  seg['preds'][i_budneck, :, :, None]),
                        fprefix + '_budneck_pred.png')
                    save_tiled_image(
                        np.uint16(
                            (2**16 - 1) * seg['preds'][i_bud, :, :, None]),
                        fprefix + '_bud_pred.png')
                    with open(fprefix + '_state.pkl', 'wb') as f:
                        pickle.dump(state, f)
                if self.suppress_errors:
                    err_msg = dict(ID=err_id,
                                   assign_mothers=assign_mothers,
                                   return_baprobs=return_baprobs)
                    err_msg = [
                        '{}: {}'.format(k, v) for k, v in err_msg.items()
                    ]
                    err_msg.insert(0, 'Tracking error')
                    logging.exception('\n'.join(err_msg))
                    ncells = len(masks)
                    tracking = {
                        'cell_label': list(range(ncells)),
                        'mother_assign': [],
                        'p_bud_assign': np.zeros((ncells, ncells)).tolist(),
                        'state': state,
                    }

                    seg['error'] = 'Tracking error: ' + err_id
                else:
                    raise

            del seg['preds']
            del seg['masks']
            if not yield_edgemasks:
                del seg['edgemasks']

            seg['cell_label'] = tracking['cell_label']
            state = tracking['state']

            # Use the last added "previous features" in the tracker state to
            # obtain the major and minor ellipsoid axes
            if state is None:
                feats = np.zeros((0, len(self.tracker.outfeats)))
            else:
                feats = state['prev_feats'][-1]
            ellip_inds = tuple(
                self.tracker.outfeats.index(p)
                for p in ('major_axis_length', 'minor_axis_length'))
            if feats.shape[0] > 0:
                seg['ellipse_dims'] = feats[:, ellip_inds].tolist()
            else:
                seg['ellipse_dims'] = []

            if assign_mothers:
                seg['mother_assign'] = tracking['mother_assign']

            if return_baprobs:
                seg['p_bud_assign'] = tracking['p_bud_assign']

            if yield_next:
                yield seg, state
            else:
                yield seg


def _generate_error_dump_id():
    return strftime('%Y-%m-%d_%H-%M-%S_') + str(uuid1())
