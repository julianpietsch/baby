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
import pickle
from collections import namedtuple, Counter
from os.path import isfile
import json
from pathlib import Path

import baby
import numpy as np
import pytest
from baby import modelsets
from baby.io import load_paired_images, load_tiled_image
from baby.morph_thresh_seg import MorphSegGrouped, SegmentationParameters
from baby.preprocessing import raw_norm, SegmentationFlattening
from baby.tracker.core import MasterTracker
from baby.utils import as_python_object, load_sklearn_model
from baby.performance import calc_IoUs, best_IoU


TrackerEnv = namedtuple('TrackerEnv', ['masks', 'p_budneck', 'p_bud', 'target'])
DEFAULT_MODELSET = 'yeast-alcatras-brightfield-EMCCD-60x-5z'


def get_tracker_model(mset, mset_name):
    # Load the celltrack and budassign models
    ctm_file = modelsets.resolve(mset['celltrack_model_file'], mset_name)
    if ctm_file.suffix == 'pkl':
        with open(ctm_file, 'rb') as f:
            ctm = pickle.load(f)
    else:
        ctm = load_sklearn_model(ctm_file)
    bam_file = modelsets.resolve(mset['budassign_model_file'], mset_name)
    if bam_file.suffix == 'pkl':
        with open(bam_file, 'rb') as f:
            bam = pickle.load(f)
    else:
        bam = load_sklearn_model(bam_file)

    # Set up a tracker for this model set
    return MasterTracker(ctrack_args={'model': ctm},
                         btrack_args={'model': bam},
                         min_bud_tps=mset.get('min_bud_tps', 3),
                         isbud_thresh=mset.get('isbud_thresh', 0.5),
                         px_size=mset.get('pixel_size', 0.263))


def segment_from_preds(mset, mset_name, impair_groups):
    # Load flattener
    ff = modelsets.resolve(mset['flattener_file'], mset_name)
    flattener = SegmentationFlattening(ff)

    tnames = flattener.names()
    i_budneck = tnames.index('bud_neck')
    bud_target = 'sml_fill' if 'sml_fill' in tnames else 'sml_inte'
    i_bud = tnames.index(bud_target)

    # Load BabyBrain param defaults
    params = mset['params']
    if type(params) == dict:
        params = SegmentationParameters(**params)
    if type(params) != SegmentationParameters:
        param_file = modelsets.resolve(mset['params'], mset_name)
        with open(param_file, 'rt') as f:
            params = json.load(f, object_hook=as_python_object)
    assert type(params) == SegmentationParameters

    segmenter = MorphSegGrouped(flattener, params=params, return_masks=True)

    # Segment and add to list of input data
    trks = {}
    for k, impairs in impair_groups.items():
        trks[k] = []
        for impair in impairs.values():
            cnn_out = raw_norm(*impair['preds']).transpose((2, 0, 1))
            seg_output = segmenter.segment(cnn_out, refine_outlines=True)
            _0xy = (0,) + cnn_out.shape[1:3]
            if len(seg_output.masks) > 0:
                masks = np.stack(seg_output.masks)
            else:
                masks = np.zeros(_0xy, dtype='bool')
            trks[k].append(
                TrackerEnv(masks, cnn_out[i_budneck], cnn_out[i_bud],
                           impair.get('segoutlines')))
    return trks


@pytest.fixture(scope='module')
def evolve60env(modelsets, image_dir):
    mset_name = DEFAULT_MODELSET
    mset = modelsets.get_params(mset_name)
    tracker = get_tracker_model(mset, mset_name)
    impair_groups = {
        'trkF': load_paired_images(image_dir.glob('evolve_testF_tp*.png'),
                                   typeA='preds'),
        'trkG': load_paired_images(image_dir.glob('evolve_testG_tp*.png'),
                                   typeA='preds')
    }
    assert all([len(impairs) > 0 for impairs in impair_groups.values()])
    trks = segment_from_preds(mset, mset_name, impair_groups)
    return tracker, trks


@pytest.fixture(scope='module')
def prime60env(modelsets, image_dir):
    mset_name = 'yeast-alcatras-brightfield-sCMOS-60x-5z'
    mset = modelsets.get_params(mset_name)
    tracker = get_tracker_model(mset, mset_name)
    impair_groups = {
        'trkC': load_paired_images(image_dir.glob('prime95b_testC_tp*.png'),
                                   typeA='preds')
    }
    assert all([len(impairs) > 0 for impairs in impair_groups.values()])
    trks = segment_from_preds(mset, mset_name, impair_groups)
    return tracker, trks


def test_bad_track(evolve60env):
    tracker, tracker_inputs = evolve60env
    tracker_inputs = tracker_inputs['trkF']
    nstepsback = 2
    state = {}
    for masks, p_budneck, p_bud, _ in tracker_inputs:
        ncells = len(masks)

        # Check feature calculation
        features = tracker.calc_feats_from_mask(masks)
        assert len(features) == ncells
        assert ncells == 0 or features.any()

        # Check label assignment
        prev_lbls = state.get('cell_lbls', [])[-nstepsback:]
        prev_feats = state.get('prev_feats', [])[-nstepsback:]
        if features.any() and prev_feats:
            counts = Counter(
                [lbl for lbl_set in prev_lbls for lbl in lbl_set])
            print(counts)
            lbls_order = list(counts.keys())
            max_prob = np.zeros((len(lbls_order), 1, len(features)),
                                dtype=float)
            new_lbls = tracker.cell_tracker.assign_lbls(max_prob, lbls_order)
            assert len(new_lbls) == ncells
        else:
            new_max = len(features)
            new_lbls = [*range(1, new_max + 1)]
            assert len(new_lbls) == ncells

        # Check get_new_lbls method
        new_lbls, _, _ = tracker.cell_tracker.get_new_lbls(None,
                                              prev_lbls,
                                              prev_feats,
                                              state.get('max_lbl', 0),
                                              new_feats=features)
        assert len(new_lbls) == ncells
        assert type(new_lbls) == list
        assert all([type(l) == int for l in new_lbls])

        tracking = tracker.step_trackers(masks, p_budneck, p_bud, state=state)
        assert len(tracking['cell_label']) == ncells
        state = tracking['state']


def test_bud_assignment(evolve60env):
    tracker, tracker_inputs = evolve60env
    trkG = tracker_inputs['trkG']
    mother_lbl = None
    bud_lbl = None
    state = {}
    for tp, (masks, p_budneck, p_bud, _) in enumerate(trkG):
        assert len(masks) == 2

        tracking = tracker.step_trackers(masks,
                                         p_budneck,
                                         p_bud,
                                         state=state,
                                         assign_mothers=True,
                                         return_baprobs=True)
        state = tracking['state']
        print(tracking['p_bud_assign'])

        assert len(tracking['cell_label']) == 2
        mother_ind = masks.sum((1, 2)).argmax()
        bud_ind = 1 - mother_ind
        if not mother_lbl:
            mother_lbl = tracking['cell_label'][mother_ind]
        if not bud_lbl:
            bud_lbl = tracking['cell_label'][bud_ind]

        print(tracking['mother_assign'])

        if tp < 2:
            assert mother_lbl not in tracking['mother_assign']
        else:
            assert mother_lbl in tracking['mother_assign']
            assert tracking['mother_assign'][bud_lbl - 1]


def test_prime60_bud_assignment(prime60env):
    tracker, tracker_inputs = prime60env
    trkC = tracker_inputs['trkC']
    state = {}
    MAX_LABEL = 10
    assigned_targets = np.full(MAX_LABEL, -MAX_LABEL-2, dtype='int')
    mother_targets = np.full(MAX_LABEL, -MAX_LABEL-2, dtype='int')
    for tp, (masks, p_budneck, p_bud, target) in enumerate(trkC):
        target_masks, target_info = target
        target_masks = target_masks.transpose((2, 0, 1))
        target_labels = np.array(target_info['cellLabels'])
        assert len(masks) == len(target_masks)
        _, assignments = best_IoU(calc_IoUs(masks, target_masks))

        tracking = tracker.step_trackers(masks,
                                         p_budneck,
                                         p_bud,
                                         state=state,
                                         assign_mothers=True,
                                         return_baprobs=True)
        state = tracking['state']
        labels = np.array(tracking['cell_label'])
        assert len(labels) == len(target_labels)

        # print(tp, target_labels, labels, assigned_targets)
        isnew = assigned_targets[labels] < 0
        new_labels = labels[isnew]
        assigned_targets[new_labels] = target_labels[assignments][isnew]
        assert np.all(target_labels[assignments] == assigned_targets[labels])
        
        print(assigned_targets)

        if 'buds' in target_info:
            tgt_lbl_max = target_labels.max() + 1
            tgt_lbl_map = np.full(tgt_lbl_max, -tgt_lbl_max-1, dtype='int')
            tgt_lbl_map[target_labels] = np.arange(target_labels.size)
            tgt_buds = np.array(target_info['buds'])
            tgt_mthrs = np.zeros_like(target_labels)
            tgt_mthrs[tgt_lbl_map[tgt_buds[tgt_buds > 0]]] = target_labels[tgt_buds > 0]
            isnew = tgt_mthrs[assignments] > 0
            # print(isnew, tgt_mthrs, tgt_lbl_map)
            mother_targets[labels[isnew]] = tgt_mthrs[assignments][isnew]

        print(mother_targets)

        if tp > 2:
            mothers = np.array(tracking['mother_assign'])
            bud_inds = np.flatnonzero(mothers > 0)
            print(mothers)
            assert np.all(mother_targets[bud_inds] ==
                          assigned_targets[mothers[bud_inds - 1]])


def test_cyclic_bud_assignment(bb_prime60, image_dir):
    input_img_files = image_dir.glob('prime95b_testB_tp?_Brightfield.png')
    input_img_files = sorted(input_img_files)

    tracker_states = [None]

    for img_file in input_img_files:
        img, _ = load_tiled_image(img_file)
        seg_trk_gen = bb_prime60.segment_and_track(
            [img], tracker_states=tracker_states, yield_next=True,
            assign_mothers=True, refine_outlines=True)
        for i, (seg, state) in enumerate(seg_trk_gen):
            tracker_states[i] = state
