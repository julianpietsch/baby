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
import pickle
from collections import namedtuple, Counter
from os.path import isfile

import baby
import numpy as np
import pytest
from baby.brain import default_params
from baby.io import load_paired_images
from baby.morph_thresh_seg import MorphSegGrouped
from baby.preprocessing import raw_norm, SegmentationFlattening
from baby.tracker.core import MasterTracker

from joblib import Parallel, delayed, parallel_backend

MODEL_DIR = baby.model_path()


def resolve_file(filename):
    if not isfile(filename):
        filename = MODEL_DIR / filename
    assert isfile(filename)
    return filename


@pytest.fixture(scope='module')
def evolve60env(modelsets, image_dir):
    mset = modelsets['evolve_brightfield_60x_5z']

    # Load flattener
    ff = resolve_file(mset['flattener_file'])
    flattener = SegmentationFlattening(ff)

    tnames = flattener.names()
    i_budneck = tnames.index('bud_neck')
    bud_target = 'sml_fill' if 'sml_fill' in tnames else 'sml_inte'
    i_bud = tnames.index(bud_target)

    # Load BabyBrain param defaults
    params = default_params.copy()
    params.update(mset.get('params', {}))

    segmenter = MorphSegGrouped(flattener, return_masks=True, **params)

    # Load CNN outputs
    impairs = load_paired_images(image_dir.glob('evolve_test[FG]_tp*.png'),
                                 typeA='preds')
    assert len(impairs) > 0
    tpkeys = (sorted([
        k for k in impairs.keys() if k.startswith('evolve_testF')
    ]), sorted([k for k in impairs.keys() if k.startswith('evolve_testG')]))

    # Load the celltrack and budassign models
    ctm_file = resolve_file(mset['celltrack_model_file'])
    with open(ctm_file, 'rb') as f:
        ctm = pickle.load(f)
    bam_file = resolve_file(mset['budassign_model_file'])
    with open(bam_file, 'rb') as f:
        bam = pickle.load(f)

    # Set up a tracker for this model set
    tracker = MasterTracker(ctrack_args={'model': ctm},
                            btrack_args={'model': bam},
                            min_bud_tps=mset.get('min_bud_tps',3),
                            isbud_thresh=mset.get('isbud_thresh',0.5),
                            px_size=0.263)

    return impairs, tpkeys, segmenter, tracker, i_budneck, i_bud


def test_parallel_segment(evolve60env):
    impairs, tpkeys, segmenter, _, _, _ = evolve60env
    
    morph_preds = [raw_norm(*p['preds']).transpose((2, 0, 1)) for p in
            impairs.values()]
    seg_ser = [_segment(segmenter, cnn_out) for cnn_out in morph_preds]
    seg_par = Parallel(n_jobs=6)(delayed(_segment)(segmenter, cnn_out)
            for cnn_out in morph_preds)
    for ss, sp in zip(seg_ser, seg_par):
        assert (ss == sp).all()


def test_parallel_track(evolve60env):
    impairs, tpkeys, segmenter, tracker, i_budneck, i_bud = evolve60env
    
    morph_preds = [raw_norm(*p['preds']).transpose((2, 0, 1)) for p in
            impairs.values()]
    seg_ser = [_segment(segmenter, cnn_out) for cnn_out in morph_preds]
    seg_par = Parallel(n_jobs=-1)(delayed(_segment)(segmenter, cnn_out)
            for cnn_out in morph_preds)

    props_ser = [_tracker_props(tracker, m) for m in seg_ser]
    props_par = Parallel(n_jobs=-1)(delayed(_tracker_props)(tracker, m) for m
            in seg_par)

    print(props_ser)
    print(props_par)
    # NB: it seems that set serialisation causes the set of features to assume
    # a random order upon loading...
    for ps, pp in zip(props_ser, props_par):
        assert list(ps[2]) == list(pp[2])

    feats_ser = [_feats(tracker, m) for m in seg_ser]
    feats_par = Parallel(n_jobs=-1)(delayed(_feats)(tracker, m)
            for m in seg_par)

    for fs, fp in zip(feats_ser, feats_par):
        assert (fs == fp).all()

    p_budneck = [cnn_out[i_budneck] for cnn_out in morph_preds]
    p_bud = [cnn_out[i_bud] for cnn_out in morph_preds]
    # for pbn, pb in zip(p_budneck, p_bud):
    #



def _segment(segmenter, cnn_out):
    segout = segmenter.segment(cnn_out, refine_outlines=True)
    _0xy = (0,) + cnn_out.shape[1:3]
    if len(segout.masks) > 0:
        masks = np.stack(segout.masks)
    else:
        masks = np.zeros(_0xy, dtype='bool')
    return masks


def _tracker_props(tracker, masks):
    return len(masks), tracker.px_size, tracker.feats2use, tracker.trapfeats


def _feats(tracker, masks):
    return tracker.calc_feats_from_mask(masks)
