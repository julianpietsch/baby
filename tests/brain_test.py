import pytest

import re
import inspect

from baby.brain import BabyBrain
from baby.morph_thresh_seg import MorphSegGrouped

from .conftest import BASE_DIR

MODEL_PATH = BASE_DIR / 'models'
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
                assert (MODEL_PATH / v).is_file())

    # Ensure that the default test model is present
    assert DEFAULT_MODELSET in modelsets


def test_init(modelsets):
    # Attempt to load default evolve model
    BabyBrain(**modelsets[DEFAULT_MODELSET])
