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
import pytest

from os.path import isfile
from pathlib import Path
import json
import inspect
import numpy as np
from scipy.ndimage import binary_fill_holes
from itertools import chain
from collections import namedtuple

from baby.io import load_paired_images, save_tiled_image
from baby.errors import BadParam
from baby.preprocessing import (raw_norm, seg_norm, dwsquareconn,
                                SegmentationFlattening)
from baby.morph_thresh_seg import (MorphSegGrouped, SegmentationOutput,
                                   SegmentationParameters)
from baby.utils import as_python_object
from baby import segmentation
from baby.segmentation import morph_seg_grouped
from baby.performance import (calc_IoUs, best_IoU, calc_AP,
                              flattener_seg_probs)


DEFAULT_MODELSET = 'yeast-alcatras-brightfield-EMCCD-60x-5z'


# Tuple for variables needed to test segmentation
SegmentationEnv = namedtuple(
    'SegmentationEnv',
    ['flattener', 'cparams', 'fparams', 'cnn_out', 'truth', 'imnames'])


# Old default parameters
DEFAULT_PARAMETERS = SegmentationParameters(
    interior_threshold=(0.7, 0.5, 0.5),
    nclosing=(1, 0, 0),
    nopening=(1, 0, 0),
    connectivity=(2, 2, 1),
    pedge_thresh=0.001,
    fit_radial=True,
    edge_sub_dilations=1,
    use_group_thresh=True,
    group_thresh_expansion=0.1)


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
        if isinstance(s_out, SegmentationOutput):
            assert tuple(len(o) for o in s_out[:-1]) == (len(s_out[0]),) * 3
        else:
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


@pytest.fixture(scope='module')
def evolve60env(modelsets, image_dir):
    mset = modelsets.get_params(DEFAULT_MODELSET)

    # Load flattener
    ff = modelsets.resolve(mset['flattener_file'], DEFAULT_MODELSET)
    flattener = SegmentationFlattening(ff)

    # Load BabyBrain param defaults
    params = mset['params']
    if type(params) == dict:
        params = SegmentationParameters(**params)
    if type(params) != SegmentationParameters:
        param_file = modelsets.resolve(mset['params'], DEFAULT_MODELSET)
        with open(param_file, 'rt') as f:
            params = json.load(f, object_hook=as_python_object)
    assert type(params) == SegmentationParameters
    cparams = params

    # Convert to params compatible with morph_seg_grouped
    msg_args = inspect.signature(morph_seg_grouped).parameters.keys()
    fparams = cparams._asdict()
    fparams = {k: v for k, v in fparams.items() if k in msg_args}
    fparams['ingroup_edge_segment'] = True
    fparams['containment_func'] = getattr(segmentation,
                                          fparams['containment_func'])
    if fparams['cellgroups'] is None:
        fparams['cellgroups'] = ['large', 'medium', 'small']

    # Load CNN outputs
    impairs = load_paired_images(image_dir.glob('evolve_*.png'),
                                 typeA='preds')
    assert len(impairs) > 0

    cnn_out = [
        raw_norm(*v['preds']).transpose((2, 0, 1)) for v in impairs.values()
    ]

    truth = [
        seg_norm(*v['segoutlines'])[0].transpose((2, 0, 1))
        for v in impairs.values()
    ]

    return SegmentationEnv(flattener, cparams, fparams, cnn_out, truth,
                           impairs.keys())


@pytest.fixture(scope='function')
def save_segoutlines(tmp_path, save_segment_outlines):

    def savefn(segoutputs, imnames):
        for segoutput, l in zip(segoutputs, imnames):
            edgemasks = segoutput.edges
            if len(edgemasks) == 0:
                continue
            save_tiled_image(np.dstack(edgemasks).astype('uint8'),
                             tmp_path / '{}_segoutlines.png'.format(l),
                             layout=(1, None))

    if save_segment_outlines:
        return savefn
    else:
        return lambda segoutputs, imnames: None


def test_match_targets_exceptions(evolve60env):
    flattener = evolve60env.flattener
    ntargets = len(flattener.names())
    with pytest.raises(BadParam, match=r'.*does not match.*flattener'):
        morph_seg_grouped(np.zeros((ntargets + 1, 81, 81)), flattener)
    segmenter = MorphSegGrouped(flattener)
    with pytest.raises(BadParam, match=r'.*does not match.*flattener'):
        segmenter.segment(np.zeros((ntargets + 1, 81, 81)))


def test_segfunc_dfltpar_empty(evolve60env):
    flattener = evolve60env.flattener
    ntargets = len(flattener.names())
    out = morph_seg_grouped(np.zeros((ntargets, 81, 81)),
                            flattener,
                            return_masks=True,
                            return_coords=True,)
    assert tuple(len(o) for o in out) == (0, 0, 0)


def test_segmenter_dfltpar_empty(evolve60env):
    flattener = evolve60env.flattener
    ntargets = len(flattener.names())
    segmenter = MorphSegGrouped(flattener,
                                return_masks=True,
                                return_coords=True)
    out = segmenter.segment(np.zeros((ntargets, 81, 81)))
    assert tuple(len(o) for o in out) == (0, 0, 0, 0)


def test_segfunc_bbparams_empty(evolve60env):
    flattener = evolve60env.flattener
    params = evolve60env.fparams
    ntargets = len(flattener.names())
    out = morph_seg_grouped(np.zeros((ntargets, 81, 81)),
                            flattener,
                            return_masks=True,
                            return_coords=True,
                            **params)
    assert tuple(len(o) for o in out) == (0, 0, 0)


def test_segmenter_bbparams_empty(evolve60env):
    flattener = evolve60env.flattener
    params = evolve60env.cparams
    ntargets = len(flattener.names())
    segmenter = MorphSegGrouped(flattener,
                                params=params,
                                return_masks=True,
                                return_coords=True)
    out = segmenter.segment(np.zeros((ntargets, 81, 81)))
    assert tuple(len(o) for o in out) == (0, 0, 0, 0)


def test_segfunc_dfltpar_preds(evolve60env, save_segoutlines):
    flattener, _, _, cnn_out, truth, imnames = evolve60env
    params = {'return_masks': True, 'return_coords': True}
    seg_outputs = [
        morph_seg_grouped(pred, flattener, **params) for pred in cnn_out
    ]
    save_segoutlines(seg_outputs, imnames)

    edge_mask_sim, IoUs, APs, fracFN, fracFP = run_performance_checks(
        seg_outputs, cnn_out, flattener, truth)

    assert edge_mask_sim.min() > 0.999
    assert fracFN < 0.4 and fracFP < 0.6
    assert IoUs.mean() > 0.5 and APs.mean() >= 0.3


def test_segmenter_dfltpar_preds(evolve60env, save_segoutlines):
    flattener, _, _, cnn_out, truth, imnames = evolve60env
    segmenter = MorphSegGrouped(flattener,
                                return_masks=True,
                                return_coords=True)
    seg_outputs = [segmenter.segment(pred) for pred in cnn_out]
    save_segoutlines(seg_outputs, imnames)

    edge_mask_sim, IoUs, APs, fracFN, fracFP = run_performance_checks(
        seg_outputs, cnn_out, flattener, truth)
    assert edge_mask_sim.min() > 0.999
    assert fracFN < 0.4 and fracFP < 0.6
    assert IoUs.mean() > 0.5 and APs.mean() >= 0.3


def test_segfunc_bbparams_preds(evolve60env, save_segoutlines):
    flattener, _, params, cnn_out, truth, imnames = evolve60env
    params = params.copy()
    params.update({'return_masks': True, 'return_coords': True})
    seg_outputs = [
        morph_seg_grouped(pred, flattener, **params) for pred in cnn_out
    ]
    save_segoutlines(seg_outputs, imnames)

    edge_mask_sim, IoUs, APs, fracFN, fracFP = run_performance_checks(
        seg_outputs, cnn_out, flattener, truth)
    assert edge_mask_sim.min() > 0.999
    assert fracFN < 0.4 and fracFP < 0.6
    assert IoUs.mean() > 0.5 and APs.mean() >= 0.25


def test_segmenter_bbparams_preds(evolve60env, save_segoutlines):
    flattener, params, _, cnn_out, truth, imnames = evolve60env
    segmenter = MorphSegGrouped(flattener,
                                params=params,
                                return_masks=True,
                                return_coords=True)
    seg_outputs = [segmenter.segment(pred) for pred in cnn_out]
    save_segoutlines(seg_outputs, imnames)

    edge_mask_sim, IoUs, APs, fracFN, fracFP = run_performance_checks(
        seg_outputs, cnn_out, flattener, truth)
    assert edge_mask_sim.min() > 0.999
    assert fracFN < 0.4 and fracFP < 0.3
    assert IoUs.mean() > 0.5 and APs.mean() >= 0.3


def test_segfunc_refined_preds(evolve60env, save_segoutlines):
    flattener, _, params, cnn_out, truth, imnames = evolve60env
    params = params.copy()
    params.update({
        'return_masks': True,
        'return_coords': True,
        'refine_outlines': True
    })
    seg_outputs = [
        morph_seg_grouped(pred, flattener, **params) for pred in cnn_out
    ]
    save_segoutlines(seg_outputs, imnames)

    edge_mask_sim, IoUs, APs, fracFN, fracFP = run_performance_checks(
        seg_outputs, cnn_out, flattener, truth)
    assert edge_mask_sim.min() > 0.999
    assert fracFN < 0.4 and fracFP < 0.4
    assert IoUs.mean() > 0.5 and APs.mean() >= 0.3


def test_segmenter_refined_preds(evolve60env, save_segoutlines):
    flattener, params, _, cnn_out, truth, imnames = evolve60env
    segmenter = MorphSegGrouped(flattener,
                                params=params,
                                return_masks=True,
                                return_coords=True)
    seg_outputs = [
        segmenter.segment(pred, refine_outlines=True) for pred in cnn_out
    ]
    save_segoutlines(seg_outputs, imnames)

    edge_mask_sim, IoUs, APs, fracFN, fracFP = run_performance_checks(
        seg_outputs, cnn_out, flattener, truth)
    assert edge_mask_sim.min() > 0.999
    assert fracFN < 0.4 and fracFP < 0.3
    assert IoUs.mean() > 0.5 and APs.mean() >= 0.3
