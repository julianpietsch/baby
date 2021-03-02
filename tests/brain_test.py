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
import pytest

import re
import inspect
import numpy as np
import json

import baby
from baby.brain import BabyBrain
from baby.morph_thresh_seg import MorphSegGrouped

#from .conftest import BASE_DIR

#MODEL_PATH = BASE_DIR / 'models'
MODEL_PATH = baby.model_path()
DEFAULT_MODELSET = 'evolve_brightfield_60x_5z'

eqlen_outkeys = {
    'angles', 'radii', 'cell_label', 'edgemasks', 'ellipse_dims', 'volumes',
    'ba_probs'
}


@pytest.fixture(scope='module')
def bb(modelsets, tf_session_graph):
    # Attempt to load default evolve model
    tf_session, tf_graph = tf_session_graph
    return BabyBrain(session=tf_session,
                     graph=tf_graph,
                     **modelsets[DEFAULT_MODELSET])


@pytest.fixture(scope='module')
def imgstack(imgs_evolve60):
    return np.stack([v['Brightfield'][0] for v in imgs_evolve60.values()])


def test_modelsets(modelsets):
    bb_args = inspect.getfullargspec(BabyBrain.__init__).args
    msg_args = inspect.getfullargspec(MorphSegGrouped.__init__).args

    for mset in modelsets.values():
        # Make sure all parameters match the BabyBrain and MorphSegGrouped
        # signatures
        assert set(mset.keys()).issubset(bb_args)
        params = mset.get('params', {})
        assert set(params.keys()).issubset(msg_args)

        # Make sure all model files exist
        for k, v in mset.items():
            if k.endswith('_file'):
                assert (MODEL_PATH / v).is_file()

    # Ensure that the default test model is present
    assert DEFAULT_MODELSET in modelsets


def test_init(bb, imgstack):
    # Attempt to load default evolve model (in fixture above)...
    # ...and ensure that the model can segment
    output = bb.segment(imgstack)
    # NB: `segment` returns a generator
    for o in output:
        assert all(
            [len(o['centres']) == len(o[k]) for k in o if k in eqlen_outkeys])


def test_segment(bb, imgstack):
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
