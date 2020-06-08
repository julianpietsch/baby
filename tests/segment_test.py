import pytest

import json
from os.path import isfile
import numpy as np
from scipy.ndimage import binary_fill_holes
from itertools import chain

from baby.io import load_paired_images
from baby.errors import BadParam
from baby.preprocessing import (
    raw_norm, seg_norm, dwsquareconn, SegmentationFlattening
)
from baby.brain import default_params
from baby.morph_thresh_seg import MorphSegGrouped
from baby.segmentation import morph_seg_grouped
from baby.performance import (
    calc_IoUs, best_IoU, calc_AP, flattener_seg_probs
)

from .conftest import MODEL_DIR, IMAGE_DIR


@pytest.fixture(scope='module')
def bbparams(modelsets):
    mset = modelsets['evolve_brightfield_60x_5z']
    params = default_params.copy()
    params.update(mset.get('params', {}))
    return params


@pytest.fixture(scope='module')
def bbparams_func(bbparams):
    params = bbparams.copy()
    del params['edge_sub_dilations']
    params['ingroup_edge_segment'] = True
    return params


@pytest.fixture(scope='module')
def flattener_evolve60(modelsets):
    f = modelsets['evolve_brightfield_60x_5z']['flattener_file']
    if not isfile(f):
        f = MODEL_DIR / f
    assert isfile(f)
    return SegmentationFlattening(f)


@pytest.fixture(scope='module')
def segmenter_evolve60(bbparams, flattener_evolve60):
    return MorphSegGrouped(flattener_evolve60, return_masks=True,
                           return_coords=True, **bbparams)


@pytest.fixture(scope='module')
def impairs_evolve60():
    impairs = load_paired_images(IMAGE_DIR.glob('evolve_*.png'), typeA='preds')
    assert len(impairs) > 0
    return impairs


@pytest.fixture(scope='module')
def preds_evolve60(impairs_evolve60):
    return [raw_norm(*v['preds']).transpose((2, 0, 1))
            for v in impairs_evolve60.values()]


@pytest.fixture(scope='module')
def truth_evolve60(impairs_evolve60):
    return [seg_norm(*v['segoutlines'])[0].transpose((2, 0, 1))
            for v in impairs_evolve60.values()]


def test_match_targets_exceptions(flattener_evolve60):
    ntargets = len(flattener_evolve60.names())
    with pytest.raises(BadParam, match=r'.*does not match.*flattener'):
        morph_seg_grouped(np.zeros((ntargets + 1, 81, 81)), flattener_evolve60)
    segmenter = MorphSegGrouped(flattener_evolve60)
    with pytest.raises(BadParam, match=r'.*does not match.*flattener'):
        segmenter.segment(np.zeros((ntargets + 1, 81, 81)))


def test_segfunc_dfltpar_empty(flattener_evolve60):
    ntargets = len(flattener_evolve60.names())
    out = morph_seg_grouped(np.zeros((ntargets, 81, 81)), flattener_evolve60,
                            return_masks=True, return_coords=True)
    assert tuple(len(o) for o in out) == (0, 0, 0)


def test_segmenter_dfltpar_empty(flattener_evolve60):
    ntargets = len(flattener_evolve60.names())
    segmenter = MorphSegGrouped(flattener_evolve60, return_masks=True,
                                return_coords=True)
    out = segmenter.segment(np.zeros((ntargets, 81, 81)))
    assert tuple(len(o) for o in out) == (0, 0, 0)


def test_segfunc_bbparams_empty(bbparams_func, flattener_evolve60):
    ntargets = len(flattener_evolve60.names())
    out = morph_seg_grouped(np.zeros((ntargets, 81, 81)), flattener_evolve60,
                            return_masks=True, return_coords=True,
                            **bbparams_func)
    assert tuple(len(o) for o in out) == (0, 0, 0)


def test_segmenter_bbparams_empty(segmenter_evolve60, flattener_evolve60):
    ntargets = len(flattener_evolve60.names())
    out = segmenter_evolve60.segment(np.zeros((ntargets, 81, 81)))
    assert tuple(len(o) for o in out) == (0, 0, 0)


def compare_edges_and_masks(edges, masks):
    if len(edges) == 0 and len(masks) == 0:
        return 1, 1
    edges, masks = np.dstack(edges), np.dstack(masks)
    filled_edges = binary_fill_holes(edges, dwsquareconn)
    nmatching = (filled_edges == masks).sum()
    ntotal = 0.99 * np.prod(masks.shape)
    return nmatching, ntotal


def get_performance(cnn_out, flattener, edges, truth):
    all_IoUs = calc_IoUs(truth, edges)
    IoUs, truth_assign = best_IoU(all_IoUs)
    nFN = (truth_assign < 0).sum()
    nTP = len(truth_assign)
    probs = flattener_seg_probs(cnn_out, flattener, edges)
    AP, pred_assign = calc_AP(all_IoUs, probs=probs)
    nFP = (pred_assign < 0).sum()
    return IoUs, AP, nTP, nFN, nFP


def run_performance_checks(seg_outputs, cnn_outputs, flattener, truth):
    edge_mask_similarity = []
    performance = []
    for s_out, nn_out, t in zip(seg_outputs, cnn_outputs, truth):
        assert tuple(len(o) for o in s_out) == (len(s_out[0]),) * 3
        nmatching, ntotal = compare_edges_and_masks(*s_out[:2])
        edge_mask_similarity.append(nmatching / ntotal)
        performance.append(get_performance(nn_out, flattener, s_out[0], t))

    edge_mask_similarity = np.array(edge_mask_similarity)
    IoUs, APs, nTPs, nFNs, nFPs = zip(*performance)
    IoUs = np.concatenate(IoUs)
    APs = np.array(APs)
    fracFN = np.sum(nFNs) / np.sum(nTPs)
    fracFP = np.sum(nFPs) / np.sum(nTPs)

    return edge_mask_similarity, IoUs, APs, fracFN, fracFP


def test_segfunc_dfltpar_preds(flattener_evolve60, preds_evolve60,
                               truth_evolve60):
    params = {'return_masks': True, 'return_coords': True}
    seg_outputs = [morph_seg_grouped(pred, flattener_evolve60, **params)
                   for pred in preds_evolve60]
    edge_mask_sim, IoUs, APs, fracFN, fracFP = run_performance_checks(
        seg_outputs, preds_evolve60, flattener_evolve60, truth_evolve60)

    assert edge_mask_sim.min() > 0.999
    assert fracFN < 0.2 and fracFP < 0.4
    assert IoUs.mean() > 0.5 and APs.mean() >= 0.4


def test_segmenter_dfltpar_preds(flattener_evolve60, preds_evolve60,
                                 truth_evolve60):
    segmenter = MorphSegGrouped(
        flattener_evolve60, return_masks=True, return_coords=True)
    seg_outputs = [segmenter.segment(pred) for pred in preds_evolve60]

    edge_mask_sim, IoUs, APs, fracFN, fracFP = run_performance_checks(
        seg_outputs, preds_evolve60, flattener_evolve60, truth_evolve60)
    assert edge_mask_sim.min() > 0.999
    assert fracFN < 0.2 and fracFP < 0.4
    assert IoUs.mean() > 0.5 and APs.mean() >= 0.4


def test_segfunc_bbparams_preds(bbparams_func, flattener_evolve60,
                                preds_evolve60, truth_evolve60):
    params = bbparams_func.copy()
    params.update({'return_masks': True, 'return_coords': True})
    seg_outputs = [morph_seg_grouped(pred, flattener_evolve60, **params)
                   for pred in preds_evolve60]

    edge_mask_sim, IoUs, APs, fracFN, fracFP = run_performance_checks(
        seg_outputs, preds_evolve60, flattener_evolve60, truth_evolve60)
    assert edge_mask_sim.min() > 0.999
    assert fracFN < 0.3 and fracFP < 0.3
    assert IoUs.mean() > 0.6 and APs.mean() >= 0.4


def test_segmenter_bbparams_preds(segmenter_evolve60, flattener_evolve60,
                                  preds_evolve60, truth_evolve60):
    seg_outputs = [segmenter_evolve60.segment(pred)
                   for pred in preds_evolve60]

    edge_mask_sim, IoUs, APs, fracFN, fracFP = run_performance_checks(
        seg_outputs, preds_evolve60, flattener_evolve60, truth_evolve60)
    assert edge_mask_sim.min() > 0.999
    assert fracFN < 0.3 and fracFP < 0.3
    assert IoUs.mean() > 0.6 and APs.mean() >= 0.4


def test_segfunc_refined_preds(bbparams_func, flattener_evolve60,
                               preds_evolve60, truth_evolve60):
    params = bbparams_func.copy()
    params.update({'return_masks': True, 'return_coords': True,
                   'refine_outlines': True})
    seg_outputs = [morph_seg_grouped(pred, flattener_evolve60, **params)
                   for pred in preds_evolve60]

    edge_mask_sim, IoUs, APs, fracFN, fracFP = run_performance_checks(
        seg_outputs, preds_evolve60, flattener_evolve60, truth_evolve60)
    assert edge_mask_sim.min() > 0.999
    assert fracFN < 0.3 and fracFP < 0.3
    assert IoUs.mean() > 0.6 and APs.mean() >= 0.4


def test_segmenter_refined_preds(segmenter_evolve60, flattener_evolve60,
                                 preds_evolve60, truth_evolve60):
    seg_outputs = [segmenter_evolve60.segment(pred, refine_outlines=True)
                   for pred in preds_evolve60]

    edge_mask_sim, IoUs, APs, fracFN, fracFP = run_performance_checks(
        seg_outputs, preds_evolve60, flattener_evolve60, truth_evolve60)
    assert edge_mask_sim.min() > 0.999
    assert fracFN < 0.3 and fracFP < 0.3
    assert IoUs.mean() > 0.6 and APs.mean() >= 0.4
