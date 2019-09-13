import numpy as np
from numpy import newaxis as nax
import pickle

from os.path import dirname, join

import tensorflow
from tensorflow.python.keras import models, layers

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

class BabyRunner(object):
    def __init__(self, morph_model_file=None, flattener_file=None,
                 budassign_model_file=None):
        self.reshaped_models = {}

        if morph_model_file is None:
            morph_model_file = join(
                models_path, 'msd_d32r2_d16_grps_tv2_20190905.hdf5')

        if flattener_file is None:
            flattener_file = join(
                models_path, 'flattener_v2_20190905.json')

        if budassign_model_file is None:
            budassign_model_file = join(
                models_path, 'baby_randomforest_20190906.pkl')

        self.morph_model = models.load_model(
            morph_model_file, custom_objects={
                'bce_dice_loss': bce_dice_loss,
                'dice_loss': dice_loss,
                'dice_coeff': dice_coeff
            })

        self.flattener = SegmentationFlattening(flattener_file)

        with open(budassign_model_file, 'rb') as f:
            self.budassign_model = pickle.load(f)

    def morph_predict(self, X):
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

        pred = self.reshaped_models[nndims].predict(X)
        return [p[:,:imdims[0],:imdims[1],:] for p in pred]

    def run(self, bf_img_batch):
        # Choose optimal segmentation parameters found in Jupyter notebook
        # segmentation-190906.ipynb:
        params = {
            'interior_threshold': (0.7,0.5,0.5),
            'nclosing': (1,0,0),
            'nopening': (1,0,0),
            'pedge_thresh': 0.001, 'fit_radial': True,
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

