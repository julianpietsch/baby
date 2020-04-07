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

class BabyCrawler(object):
    def __init__(self, baby_brain):
        self.baby_brain = baby_brain

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

