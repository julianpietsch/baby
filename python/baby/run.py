from __future__ import absolute_import, division, print_function, unicode_literals
import numpy as np
from numpy import newaxis as nax
import pickle

from os.path import dirname, join, isfile

import tensorflow as tf
from tensorflow.keras import models, layers
from tensorflow.keras import backend as K

from scipy.ndimage import minimum_filter
from skimage.measure import regionprops

from .models import bce_dice_loss, dice_loss, dice_coeff
from .segmentation import (
    morph_seg_grouped,
    morph_thresh_masks, morph_radial_thresh_fit, unique_masks,
    squareconn, mask_containment, draw_radial
)
from .tracking import get_mother_bud_stats
from .preprocessing import robust_norm, SegmentationFlattening
from .utils import batch_iterator, split_batch_pred

models_path = join(dirname(__file__),'..','..','models')

tf_version = [int(v) for v in tf.version.VERSION.split('.')]

class BabyRunner(object):
    def __init__(self, morph_model_file=None, flattener_file=None,
                 budassign_model_file=None, default_image_size=None,
                 session=None, graph=None):
        self.reshaped_models = {}

        if morph_model_file is None:
            morph_model_file = join(
                #models_path, 'msd_d32r2_d16_grps_tv2_20190905.hdf5')
                models_path, 'I5_msd_d80_20190916.hdf5')
        elif not isfile(morph_model_file):
            morph_model_file = join(models_path, morph_model_file)

        if flattener_file is None:
            flattener_file = join(
                models_path, 'flattener_v2_20190905.json')
        elif not isfile(flattener_file):
            flattener_file = join(models_path, flattener_file)

        if budassign_model_file is None:
            budassign_model_file = join(
                models_path, 'baby_randomforest_20190906.pkl')
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

        print('Loading Keras model "{}"...'.format(morph_model_file))
        if tf_version[0] == 1:
            with self.graph.as_default():
                K.set_session(session)
                print('Loading model into session "{}"...'.format(
                    K.get_session()))
                self.morph_model = models.load_model(
                    morph_model_file, custom_objects={
                        'bce_dice_loss': bce_dice_loss,
                        'dice_loss': dice_loss,
                        'dice_coeff': dice_coeff
                    })
        else:
            self.morph_model = models.load_model(
                morph_model_file, custom_objects={
                    'bce_dice_loss': bce_dice_loss,
                    'dice_loss': dice_loss,
                    'dice_coeff': dice_coeff
                })

        self.flattener = SegmentationFlattening(flattener_file)

        with open(budassign_model_file, 'rb') as f:
            self.budassign_model = pickle.load(f)

        # Run prediction on mock image to load model for prediction
        _, x, y, z = self.morph_model.input.shape

        if default_image_size is not None:
            try:
                x, y = default_image_size
            except TypeError:
                x = default_image_size
                y = x

        self.morph_predict(np.zeros((1,x,y,z)))


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
        nndims = tuple([int(np.ceil(float(d)/8.))*8 for d in imdims])
        if not all([n==i for n, i in zip(nndims, imdims)]):
            xpad, ypad = tuple(n-i for n,i in zip(nndims,imdims))
            X = np.pad(X,((0,0),(0,xpad),(0,ypad),(0,0)),'edge')

        if nndims not in self.reshaped_models:
            base_input_shape = self.morph_model.input.shape[1:3]
            if all([n == m for n, m in zip(nndims, base_input_shape)]):
                self.reshaped_models[nndims] = self.morph_model
            else:
                i = layers.Input(shape=X.shape[1:])
                self.reshaped_models[nndims] = models.Model(i, self.morph_model(i))

        if tf_version[0] == 1:
            print('Running prediction in session "{}"...'.format(
                K.get_session()))

        pred = self.reshaped_models[nndims].predict(X)

        return [p[:,:imdims[0],:imdims[1],:] for p in pred]


    def run(self, bf_img_batch):
        # Choose optimal segmentation parameters found in Jupyter notebook
        # segmentation-190906.ipynb:
        params = {
            'interior_threshold': (0.7,0.5,0.5),
            'nclosing': (1,0,0),
            'nopening': (1,0,0),
            'connectivity': (2,2,1),
            'pedge_thresh': 0.001, 'fit_radial': True,
            'ingroup_edge_segment': True,
            'use_group_thresh': True,
            'group_thresh_expansion': 0.1
        }

        output = []

        # First preprocess each brightfield image in batch
        bf_img_batch = np.stack([robust_norm(img, {}) for img in bf_img_batch])

        for batch in batch_iterator(bf_img_batch):
            morph_preds = split_batch_pred(self.morph_predict(batch))

            for cnn_output in morph_preds:
                _, masks, coords = morph_seg_grouped(
                    cnn_output, self.flattener, return_masks=True,
                    return_coords=True, **params
                )

                if len(coords) > 0:
                    centres, radii, angles = zip(*coords)
                else:
                    centres, radii, angles = 3 * [[]]

                ncells = len(masks)

                # Assign mothers to daughters (masks and rprops will be updated)
                ba_prob_mat = np.nan * np.ones((ncells, ncells))
                mb_stats = get_mother_bud_stats(
                    cnn_output, self.flattener, None, masks=masks)
                mb_stats = np.hstack([s.flatten()[:,nax] for s in mb_stats])
                good_stats = ~np.isnan(mb_stats).any(axis=1)
                ba_probs = np.nan * np.ones(ncells**2)
                if good_stats.any():
                    ba_probs[good_stats] = self.budassign_model.predict_proba(
                        mb_stats[good_stats,:])[:,1]
                ba_probs = ba_probs.reshape((ncells,)*2)

                ba_prob_mat = ba_probs

                # Save output as a dict
                output.append({
                    'centres': centres,
                    'angles': [a.tolist() for a in angles],
                    'radii': [r.tolist() for r in radii],
                    'ba_probs': ba_prob_mat.tolist()
                })

        return output

