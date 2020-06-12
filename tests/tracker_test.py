import pytest

from os.path import isfile
from collections import namedtuple, Counter
import numpy as np

from baby.io import load_paired_images, save_tiled_image
from baby.preprocessing import raw_norm, seg_norm, SegmentationFlattening
from baby.brain import default_params
from baby.morph_thresh_seg import MorphSegGrouped
from baby.tracker import Tracker

from .conftest import MODEL_DIR, IMAGE_DIR

TrackerEnv = namedtuple('TrackerEnv', ['masks', 'p_budneck', 'p_bud'])


@pytest.fixture(scope='module')
def evolve60env(modelsets):
    mset = modelsets['evolve_brightfield_60x_5z']

    # Load flattener
    ff = mset['flattener_file']
    if not isfile(ff):
        ff = MODEL_DIR / ff
    assert isfile(ff)
    flattener = SegmentationFlattening(ff)

    tnames = flattener.names()
    i_budneck = tnames.index('bud_neck')
    i_bud = tnames.index('sml_fill')

    # Load BabyBrain param defaults
    params = default_params.copy()
    params.update(mset.get('params', {}))

    segmenter = MorphSegGrouped(flattener, return_masks=True, **params)

    # Load CNN outputs
    impairs = load_paired_images(IMAGE_DIR.glob('evolve_testF_tp*.png'),
                                 typeA='preds')
    assert len(impairs) > 0
    tpkeys = sorted(impairs.keys())

    # Segment and add to list of input data
    trkin = []
    for k in tpkeys:
        impair = impairs[k]
        cnn_out = raw_norm(*impair['preds']).transpose((2, 0, 1))
        _, masks = segmenter.segment(cnn_out, refine_outlines=True)
        trkin.append(TrackerEnv(masks, cnn_out[i_budneck], cnn_out[i_bud]))

    # Set up a tracker for this model set
    tracker = Tracker(ctrack_model=mset['celltrack_model_file'],
                      ba_model=mset['budassign_model_file'])

    return tracker, trkin


def test_bad_track(evolve60env):
    tracker, input_args = evolve60env
    nstepsback = 2
    state = {}
    for masks, p_budneck, p_bud in input_args:
        ncells = len(masks)

        # Check feature calculation
        features = tracker.calc_feats_from_masks(masks)
        assert len(features) == ncells
        assert ncells == 0 or features.any()

        # Check label assignment
        prev_lbls = state.get('cell_lbls', [])[-nstepsback:]
        prev_feats = state.get('prev_feats', [])[-nstepsback:]
        if features.any() and prev_feats:
            counts = Counter([lbl for lbl_set in prev_lbls for lbl in
                lbl_set])
            print(counts)
            lbls_order = list(counts.keys())
            max_prob = np.zeros((len(lbls_order), len(features)), dtype=float)
            new_lbls = tracker.assign_lbls(max_prob, lbls_order)
            assert len(new_lbls) == ncells
        else:
            new_max = len(features)
            new_lbls = [*range(1, new_max + 1)]
            assert len(new_lbls) == ncells

        tracking = tracker.step_trackers(masks, p_budneck, p_bud, state=state)
        assert len(tracking['cell_label']) == ncells
        state = tracking['state']
