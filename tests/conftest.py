import pytest

from pathlib import Path
import json

from baby.io import load_paired_images
from baby.brain import BabyBrain

BASE_DIR = Path(__file__).parents[1]
MODEL_DIR = BASE_DIR / 'models'
IMAGE_DIR = Path(__file__).parent / 'images'


def pytest_addoption(parser):
    parser.addoption(
        "--save-cnn-predictions", action="store_true", default=False,
        help="When running CNN tests, save the predictions to image dir"
    )


@pytest.fixture(scope='session')
def save_cnn_predictions(request):
    return request.config.getoption("--save-cnn-predictions")


@pytest.fixture(scope='session')
def imgs_evolve60():
    return load_paired_images(IMAGE_DIR.glob('evolve_*.png'))


@pytest.fixture(scope='session')
def modelsets():
    with open(BASE_DIR / 'modelsets.json') as f:
        msets = json.load(f)
    return msets


@pytest.fixture(scope='session')
def bb_evolve60(modelsets):
    return BabyBrain(**modelsets['evolve_brightfield_60x_5z'])


@pytest.fixture(scope='module')
def bb_prime60(modelsets):
    return BabyBrain(**modelsets['prime95b_brightfield_60x_5z'])
