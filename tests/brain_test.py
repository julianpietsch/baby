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

import re
import inspect
import numpy as np
import json

import baby
from baby.brain import BabyBrain
from baby.morph_thresh_seg import MorphSegGrouped, SegmentationParameters

#from .conftest import BASE_DIR

DEFAULT_MODELSET = 'yeast-alcatras-brightfield-EMCCD-60x-5z'
TEST_MODELSETS = ['yeast-alcatras-brightfield-EMCCD-60x-5z',
                  'yeast-alcatras-brightfield-sCMOS-60x-5z',
                  'ecoli-mothermachine-phase-sCMOS-100x-1z']


eqlen_outkeys = {
    'angles', 'radii', 'cell_label', 'edgemasks', 'ellipse_dims', 'volumes',
    'ba_probs'
}


@pytest.fixture(scope='module')
def bb(modelsets, tf_session_graph):
    # Attempt to load default evolve model
    tf_session, tf_graph = tf_session_graph
    return modelsets.get(DEFAULT_MODELSET, session=tf_session, graph=tf_graph)


@pytest.fixture(scope='module')
def imgstack(imgs_evolve60):
    return np.stack([v['Brightfield'][0] for v in imgs_evolve60.values()])


def test_modelsets(modelsets, verify_all_modelsets):
    # Ensure that at least the test model sets are present
    all_mset_ids = modelsets.ids()
    assert all([mset_id in all_mset_ids for mset_id in TEST_MODELSETS])

    bb_args = inspect.getfullargspec(BabyBrain.__init__).args
    msg_args = set(SegmentationParameters()._fields)

    if verify_all_modelsets:
        modelsets.update('all', force=False)
    else:
        modelsets.update(TEST_MODELSETS, force=False)

    for mset_info in modelsets.specifications().values():
        # Make sure all parameters match the BabyBrain and MorphSegGrouped
        # signatures
        mset = mset_info['brain_params']
        assert set(mset.keys()).issubset(bb_args)
        params = mset.get('params', {})
        if type(params) == dict:
            assert set(params.keys()).issubset(msg_args)

    share_path = modelsets.LOCAL_MODELSETS_PATH / modelsets.SHARE_PATH
    for mset_id, mset_info in modelsets.specifications(local=True).items():
        mset_path = modelsets.LOCAL_MODELSETS_PATH / mset_id
        mset = mset_info['brain_params']

        assert mset_path.is_dir()
        params = mset.get('params', {})
        if type(params) != dict and type(params) != SegmentationParameters:
            assert ((mset_path / params).is_file() or
                    (share_path / params).is_file())

        # Make sure all model files exist
        for k, v in mset.items():
            if k.endswith('_file'):
                assert ((mset_path / v).is_file() or
                        (share_path / v).is_file())


def test_init(bb, imgstack):
    # Attempt to load default evolve model (in fixture above)...
    # ...and ensure that the model can segment
    output = bb.segment(imgstack)
    # NB: `segment` returns a generator
    for o in output:
        assert all(
            [len(o['centres']) == len(o[k]) for k in o if k in eqlen_outkeys])


def test_evolve_segment(bb, imgstack):
    # Test segment with all options enabled
    output = bb.segment(imgstack,
                        yield_edgemasks=True,
                        yield_masks=True,
                        yield_preds=True,
                        yield_volumes=True,
                        refine_outlines=True)
    for o in output:
        assert all(
            [len(o['centres']) == len(o[k]) for k in o if k in eqlen_outkeys])


def test_prime_segment(bb_prime60, bfimgs_prime60):
    imgstack = np.stack([v[0] for v in bfimgs_prime60.values()])
    # Test segment with all options enabled
    output = bb_prime60.segment(imgstack,
                                yield_edgemasks=True,
                                yield_masks=True,
                                yield_preds=True,
                                yield_volumes=True,
                                refine_outlines=True)
    for o in output:
        assert all(
            [len(o['centres']) == len(o[k]) for k in o if k in eqlen_outkeys])


def test_mm_segment(bb_mmscmos, imgs_mmscmos):
    # The sample mother machine sCMOS images have different shapes so cannot
    # be stacked, so segment each image separately
    for imgpair in imgs_mmscmos.values():
        img = imgpair['Brightfield'][0]
        # Test segment with all options enabled except refine_outlines, which is
        # not yet available for the cartesian splines used for E. coli
        o = bb_mmscmos.segment(img[None, ...],
                               yield_edgemasks=True,
                               yield_masks=True,
                               yield_preds=True,
                               yield_volumes=True,
                               refine_outlines=False)
        # Expand generator and select first image
        o = list(o)[0]
        assert all(
            [len(o['centres']) == len(o[k]) for k in o if k in eqlen_outkeys])


def test_evolve_segment_and_track(bb, imgstack, imgs_evolve60):
    # Test stateless version
    output = bb.segment_and_track(imgstack,
                                  yield_volumes=True,
                                  refine_outlines=True,
                                  return_baprobs=True)
    for o in output:
        # print(json.dumps(o, indent=2))
        assert all(
            [len(o['centres']) == len(o[k]) for k in o if k in eqlen_outkeys])

    # Test stateful version
    output0, state = zip(*bb.segment_and_track(
        imgstack, yield_next=True, yield_volumes=True, refine_outlines=True))

    # Step once to the same image (cells should track exactly)
    output1, state = zip(*bb.segment_and_track(imgstack,
                                               tracker_states=state,
                                               yield_next=True,
                                               yield_volumes=True,
                                               refine_outlines=True))
    for o1, o0 in zip(output1, output0):
        assert np.all(
            np.array(o0['cell_label']) == np.array(o1['cell_label']))

    # Test stateful version stepping between real time points
    tp3 = imgs_evolve60['evolve_testF_tp3']['Brightfield'][0][None, ...]
    tp4 = imgs_evolve60['evolve_testF_tp4']['Brightfield'][0][None, ...]
    output3, state = zip(*bb.segment_and_track(
        tp3, yield_next=True, yield_volumes=True, refine_outlines=True))
    output4, state = zip(*bb.segment_and_track(tp4,
                                               tracker_states=state,
                                               yield_next=True,
                                               yield_volumes=True,
                                               refine_outlines=True))
    list(output4)  # NB: required since output is from a generator


def test_evolve_segment_and_track_parallel(bb, imgstack, imgs_evolve60):
    # Compare non-parallel with parallel
    output_np, state_np = zip(*bb.segment_and_track(imgstack,
                                                    yield_next=True,
                                                    yield_volumes=True,
                                                    refine_outlines=True,
                                                    yield_edgemasks=True))

    output0, state = zip(*bb.segment_and_track_parallel(imgstack,
                                                        yield_next=True,
                                                        yield_volumes=True,
                                                        refine_outlines=True,
                                                        yield_edgemasks=True))

    for onp, o0 in zip(output_np, output0):
        assert (np.array(onp['centres']) == np.array(o0['centres'])).all()
        assert (np.array(onp['edgemasks']) == np.array(o0['edgemasks'])).all()

    for snp, s0 in zip(state_np, state):
        assert (np.array(snp['max_lbl']) == np.array(s0['max_lbl'])).all()
        assert (np.array(snp['cell_lbls']) == np.array(s0['cell_lbls'])).all()
        assert (np.array(snp['lifetime']) == np.array(s0['lifetime'])).all()
        # assert np.array(snp['ba_cum']).sum() == np.array(s0['ba_cum']).sum()
        # assert (np.array(snp['prev_feats']) == np.array(
        #     s0['prev_feats'])).all()

    # Step once to the same image (cells should track exactly)
    output1, _ = zip(*bb.segment_and_track_parallel(imgstack,
                                                    tracker_states=state,
                                                    yield_next=True,
                                                    yield_volumes=True,
                                                    refine_outlines=True))
    for o1, o0 in zip(output1, output0):
        assert np.all(
            np.array(o0['cell_label']) == np.array(o1['cell_label']))

    # Test stateful version stepping between real time points
    tp3 = imgs_evolve60['evolve_testF_tp3']['Brightfield'][0][None, ...]
    tp4 = imgs_evolve60['evolve_testF_tp4']['Brightfield'][0][None, ...]
    output3, state = zip(*bb.segment_and_track_parallel(
        tp3, yield_next=True, yield_volumes=True, refine_outlines=True))
    output4, state = zip(*bb.segment_and_track_parallel(tp4,
                                                        tracker_states=state,
                                                        yield_next=True,
                                                        yield_volumes=True,
                                                        refine_outlines=True))
    list(output4)  # NB: required since output is from a generator
