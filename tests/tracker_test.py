import pytest

from os.path import isfile
import pickle
from collections import namedtuple, Counter
import numpy as np

from baby.io import load_paired_images, save_tiled_image
from baby.preprocessing import raw_norm, seg_norm, SegmentationFlattening
from baby.brain import default_params
from baby.morph_thresh_seg import MorphSegGrouped
from baby.tracker import Tracker

from .conftest import MODEL_DIR, IMAGE_DIR

TrackerEnv = namedtuple('TrackerEnv', ['masks', 'p_budneck', 'p_bud'])


def resolve_file(filename):
    if not isfile(filename):
        filename = MODEL_DIR / filename
    assert isfile(filename)
    return filename


@pytest.fixture(scope='module')
def evolve60env(modelsets):
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

    # Load the celltrack and budassign models
    ctm_file = resolve_file(mset['celltrack_model_file'])
    with open(ctm_file, 'rb') as f:
        ctm = pickle.load(f)
    bam_file = resolve_file(mset['budassign_model_file'])
    with open(bam_file, 'rb') as f:
        bam = pickle.load(f)

    # Set up a tracker for this model set
    tracker = Tracker(ctrack_model=ctm, ba_model=bam)

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
            counts = Counter(
                [lbl for lbl_set in prev_lbls for lbl in lbl_set])
            print(counts)
            lbls_order = list(counts.keys())
            max_prob = np.zeros((len(lbls_order), len(features)), dtype=float)
            new_lbls = tracker.assign_lbls(max_prob, lbls_order)
            assert len(new_lbls) == ncells
        else:
            new_max = len(features)
            new_lbls = [*range(1, new_max + 1)]
            assert len(new_lbls) == ncells

        # Check get_new_lbls method
        new_lbls, _, _ = tracker.get_new_lbls(None,
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
