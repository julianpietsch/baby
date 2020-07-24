import pytest

import re
import inspect
import numpy as np

import baby
from baby.brain import BabyBrain
from baby.morph_thresh_seg import MorphSegGrouped

#from .conftest import BASE_DIR

#MODEL_PATH = BASE_DIR / 'models'
MODEL_PATH = baby.model_path()
DEFAULT_MODELSET = 'evolve_brightfield_60x_5z'


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


def test_init(modelsets, tf_session_graph, imgs_evolve60):
    # Attempt to load default evolve model
    tf_session, tf_graph = tf_session_graph
    bb = BabyBrain(session=tf_session,
                   graph=tf_graph,
                   **modelsets[DEFAULT_MODELSET])

    # And ensure that the model can segment
    imgstack = np.stack([v['Brightfield'] for v in imgs_evolve60.values()])
    bb.segment(imgstack)
