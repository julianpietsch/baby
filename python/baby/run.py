import numpy as np
from numpy import newaxis as nax
import pickle

from os.path import dirname, join

import tensorflow
from tensorflow.python.keras import models, layers

from scipy.ndimage import minimum_filter
from skimage.measure import regionprops

from .models import bce_dice_loss, dice_loss
from .segmentation import (
    morph_thresh_masks, morph_radial_thresh_fit, unique_masks, connect_filt,
    mask_containment, draw_radial
)
from .tracking import get_mother_bud_stats
from .io import preprocess_brightfield
from .utils import batch_iterator, split_batch_pred

models_path = join(dirname(__file__),'..','..','models')

class BabyRunner(object):
    def __init__(self, morph_model_file=None, budassign_model_file=None):
        self.reshaped_models = {}

        if morph_model_file is None:
            morph_model_file = join(
                models_path, 'morphviz_msd_d80_bn_20190624.hdf5')

        if budassign_model_file is None:
            budassign_model_file = join(
                models_path, 'baby_randomforest_20190720.pkl')

        self.morph_model = models.load_model(
            morph_model_file, custom_objects={
                'bce_dice_loss': bce_dice_loss, 'dice_loss': dice_loss})

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
        # Choose optimal segmentation parameters found in Jupyter notebook for
        # validation data:
        interior_threshold = 0.90
        # interior_threshold = 0.85
        overlap_threshold = 0.8
        bud_threshold = 0.8
        bud_overlap = True
        # isbud_threshold = 0.3
        isbud_threshold = None

        output = []

        # First preprocess each brightfield image in batch
        bf_img_batch = np.stack([preprocess_brightfield(img) for img in bf_img_batch])

        for batch in batch_iterator(bf_img_batch):
            morph_preds = split_batch_pred(self.morph_predict(batch))

            for cnn_output in morph_preds:
                p_edge, _, p_interior, p_overlap, p_budneck, p_bud = cnn_output
                shape = p_interior.shape

                if isbud_threshold is None and bud_threshold is not None:
                    p_interior *= 1 - p_bud

                p_interior *= 1 - p_edge

                masks = morph_thresh_masks(
                    p_interior, interior_threshold=interior_threshold,
                    p_overlap=p_overlap, overlap_threshold=overlap_threshold)

                budmasks = morph_thresh_masks(
                    p_bud, interior_threshold=bud_threshold, dilate=False,
                    p_overlap=p_overlap, overlap_threshold=overlap_threshold)

                if isbud_threshold is not None:
                    # Omit interior masks if they overlap with bud masks
                    masks = unique_masks(masks, budmasks, iou_func=mask_containment,
                                        threshold=isbud_threshold) + budmasks
                else:
                    masks = masks + budmasks

                # Need mask outlines and region properties
                mseg = [minimum_filter(m, footprint=connect_filt) != m for m in masks]
                rprops = [regionprops(m.astype('int'), coordinates='rc')[0] for m in masks]

                outlines = []
                all_radii = []
                all_angles = []
                for mask, outline, rp in zip(masks, mseg, rprops):
                    try:
                        radii, angles = morph_radial_thresh_fit(outline, mask, rp)
                    except:
                        all_radii.append(None)
                        all_angles.append(None)
                    finally:
                        all_radii.append(radii.tolist())
                        all_angles.append(angles.tolist())

                    try:
                        outline = draw_radial(radii, angles, rp.centroid, shape)
                    except:
                        outlines.append(None)
                    finally:
                        outlines.append(outline)

                ncells = len(outlines)

                # Assign mothers to daughters (masks and rprops will be updated)
                ba_prob_mat = np.nan * np.ones((ncells, ncells))
                good_ol = np.flatnonzero([o is not None for o in outlines])
                outlines = [o for o in outlines if o is not None]

                mb_stats = get_mother_bud_stats(cnn_output, outlines)
                mb_stats = np.hstack([s.flatten()[:,nax] for s in mb_stats])
                good_stats = ~np.isnan(mb_stats).any(axis=1)
                ba_probs = np.nan * np.ones(len(outlines)**2)
                if good_stats.any():
                    ba_probs[good_stats] = self.budassign_model.predict_proba(
                        mb_stats[good_stats,:])[:,1]
                ba_probs = ba_probs.reshape((len(outlines),)*2)

                ba_prob_mat[good_ol[:,None], good_ol] = ba_probs

                # Save output as a dict
                output.append({
                    'centres': [rp.centroid for rp in rprops],
                    'angles': all_angles,
                    'radii': all_radii,
                    'ba_probs': ba_prob_mat.tolist()
                })

        return output

