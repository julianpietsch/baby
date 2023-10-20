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
from pathlib import Path
import os

import baby
import pytest
from baby import modelsets as modelsets_module
from baby.brain import BabyBrain
from baby.io import load_paired_images

IMAGE_DIR = Path(__file__).parent / 'images'


def pytest_addoption(parser):
    parser.addoption(
        "--save-cnn-predictions", action="store_true", default=False,
        help="When running CNN tests, save the predictions to image dir"
    )
    parser.addoption(
        "--save-segment-outlines", action="store_true", default=False,
        help="When running segment tests, save the outlines to tmp dir"
    )
    parser.addoption(
        "--save-crawler-output", action="store_true", default=False,
        help="When running crawler tests, save the predictions to tmp dir"
    )
    parser.addoption(
        "--verify-all-modelsets", action="store_true", default=False,
        help="Download all models to confirm model set files exist"
    )


@pytest.fixture(scope='session')
def image_dir():
    return IMAGE_DIR


@pytest.fixture(scope='session')
def save_cnn_predictions(request):
    return request.config.getoption("--save-cnn-predictions")


@pytest.fixture(scope='session')
def save_crawler_output(request):
    return request.config.getoption("--save-crawler-output")


@pytest.fixture(scope='session')
def save_segment_outlines(request):
    return request.config.getoption("--save-segment-outlines")


@pytest.fixture(scope='session')
def verify_all_modelsets(request):
    return request.config.getoption("--verify-all-modelsets")


@pytest.fixture(scope='session')
def imgs_evolve60():
    return load_paired_images(IMAGE_DIR.glob('evolve_*.png'))


@pytest.fixture(scope='session')
def imgs_prime60():
    return load_paired_images(IMAGE_DIR.glob('prime95b_*.png'))


@pytest.fixture(scope='session')
def imgs_mmscmos():
    return load_paired_images(IMAGE_DIR.glob('mmsCMOS_*.png'))


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
def modelsets(tmp_path_factory):
    envpath = os.environ.get(modelsets_module.ENV_VAR_MODELSETS_PATH)
    if not envpath:
        envpath = tmp_path_factory.mktemp('modelsets')
        print(envpath)
        # Patch modelsets module with temporary storage path
        modelsets_module.ENV_LOCAL_MODELSETS_PATH = envpath
        modelsets_module.LOCAL_MODELSETS_PATH = envpath
        modelsets_module.LOCAL_MODELSETS_CACHE = (
            envpath / modelsets_module.MODELSETS_FILENAME)
    return modelsets_module


@pytest.fixture(scope='session')
def bb_evolve60(modelsets, tf_session_graph):
    tf_session, tf_graph = tf_session_graph
    return modelsets.get('yeast-alcatras-brightfield-EMCCD-60x-5z',
                         session=tf_session, graph=tf_graph)


@pytest.fixture(scope='module')
def bb_prime60(modelsets, tf_session_graph):
    tf_session, tf_graph = tf_session_graph
    return modelsets.get('yeast-alcatras-brightfield-sCMOS-60x-5z',
                         session=tf_session, graph=tf_graph)


@pytest.fixture(scope='module')
def bb_mmscmos(modelsets, tf_session_graph):
    tf_session, tf_graph = tf_session_graph
    return modelsets.get('ecoli-mm-phase-sCMOS-100x-1z',
                         session=tf_session, graph=tf_graph)
