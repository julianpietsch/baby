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
from baby.io import load_paired_images
from baby.morph_thresh_seg import MorphSegGrouped, SegmentationParameters
from baby.preprocessing import raw_norm, SegmentationFlattening
from baby.tracker.core import MasterTracker
from baby.utils import as_python_object


TrackerEnv = namedtuple('TrackerEnv', ['masks', 'p_budneck', 'p_bud'])
DEFAULT_MODELSET = 'yeast-alcatras-brightfield-EMCCD-60x-5z'


@pytest.fixture(scope='module')
def evolve60env(modelsets, image_dir):
    mset = modelsets.get_params(DEFAULT_MODELSET)

    # Load flattener
    ff = modelsets.resolve(mset['flattener_file'], DEFAULT_MODELSET)
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
        param_file = modelsets.resolve(mset['params'], DEFAULT_MODELSET)
        with open(param_file, 'rt') as f:
            params = json.load(f, object_hook=as_python_object)
    assert type(params) == SegmentationParameters

    segmenter = MorphSegGrouped(flattener, params=params, return_masks=True)

    # Load CNN outputs
    impairs = load_paired_images(image_dir.glob('evolve_test[FG]_tp*.png'),
                                 typeA='preds')
    assert len(impairs) > 0
    tpkeys = (sorted([
        k for k in impairs.keys() if k.startswith('evolve_testF')
    ]), sorted([k for k in impairs.keys() if k.startswith('evolve_testG')]))

    # Segment and add to list of input data
    trks = ([], [])
    for i in range(len(tpkeys)):
        for k in tpkeys[i]:
            impair = impairs[k]
            cnn_out = raw_norm(*impair['preds']).transpose((2, 0, 1))
            seg_output = segmenter.segment(cnn_out, refine_outlines=True)
            _0xy = (0,) + cnn_out.shape[1:3]
            if len(seg_output.masks) > 0:
                masks = np.stack(seg_output.masks)
            else:
                masks = np.zeros(_0xy, dtype='bool')
            trks[i].append(
                TrackerEnv(masks, cnn_out[i_budneck], cnn_out[i_bud]))
    trkF, trkG = trks

    # Load the celltrack and budassign models
    ctm_file = modelsets.resolve(mset['celltrack_model_file'], DEFAULT_MODELSET)
    with open(ctm_file, 'rb') as f:
        ctm = pickle.load(f)
    bam_file = modelsets.resolve(mset['budassign_model_file'], DEFAULT_MODELSET)
    with open(bam_file, 'rb') as f:
        bam = pickle.load(f)

    # Set up a tracker for this model set
    tracker = MasterTracker(ctrack_args={'model': ctm},
                            btrack_args={'model': bam},
                            min_bud_tps=mset.get('min_bud_tps',3),
                            isbud_thresh=mset.get('isbud_thresh',0.5),
                            px_size=0.263)

    return tracker, trkF, trkG


def test_bad_track(evolve60env):
    tracker, input_args, _ = evolve60env
    nstepsback = 2
    state = {}
    for masks, p_budneck, p_bud in input_args:
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
    tracker, _, trkG = evolve60env
    mother_lbl = None
    bud_lbl = None
    state = {}
    for tp, (masks, p_budneck, p_bud) in enumerate(trkG):
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
