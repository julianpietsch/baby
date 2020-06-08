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
def tf_session_graph():
    import tensorflow as tf

    # Compensate for bug in tensorflow + RTX series NVidia GPUs
    tf_version = tuple(int(v) for v in tf.version.VERSION.split('.'))
    if tf_version[0] == 1:
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        tf_session = tf.Session(config=config)
        tf_graph = tf.get_default_graph()
    elif tf_version[0] == 2:
        tf_session, tf_graph = 2 * (None,)
        gpus = tf.config.experimental.list_physical_devices('GPU')
        if gpus:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logical_gpus = tf.config.experimental.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,", len(logical_gpus),
                    "Logical GPUs")
    else:
        raise Exception(
            'Unsupported version of tensorflow encountered ({})'.format(
                tf.version.VERSION
            )
        )

    return tf_session, tf_graph

@pytest.fixture(scope='session')
def bb_evolve60(modelsets, tf_session_graph):
    tf_session, tf_graph = tf_session_graph
    return BabyBrain(session=tf_session, graph=tf_graph,
                     **modelsets['evolve_brightfield_60x_5z'])


@pytest.fixture(scope='module')
def bb_prime60(modelsets, tf_session_graph):
    tf_session, tf_graph = tf_session_graph
    return BabyBrain(session=tf_session, graph=tf_graph,
                     **modelsets['prime95b_brightfield_60x_5z'])
