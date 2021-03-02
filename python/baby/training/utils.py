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
from contextlib import contextmanager
from typing import NamedTuple, Any, Tuple, Union

import tensorflow as tf
from baby.augmentation import Augmenter
from baby.generator import ImageLabel
from baby.utils import EncodableNamedTuple


def fix_tf_rtx_gpu_bug():
    """Run to set up TensorFlow session with RTX series NVidia GPUs

    Compensates for cuDNN bug in tensorflow + RTX series NVidia GPUs.
    """
    if tf.version.VERSION.startswith('1'):
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        tf_session = tf.Session(config=config)
        tf.keras.backend.set_session(tf_session)
    elif tf.version.VERSION.startswith('2'):
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
                    tf.version.VERSION))


class TrainValProperty(NamedTuple):
    train: Any
    val: Any


class TrainValTestProperty(NamedTuple):
    train: Any
    val: Any
    test: Any


@EncodableNamedTuple
class BabyTrainerParameters(NamedTuple):
    """Global parameters for training BABY models
    """
    train_val_test_pairs_file: str = 'train_val_pairs.json'
    tracker_tvt_pairs_file: str = 'tracker_tvt_pairs.json'
    smoothing_sigma_stats_file: str = 'smoothing_sigma_stats.csv'
    smoothing_sigma_model_file: str = 'smoothing_sigma_model.json'
    flattener_stats_file: str = 'flattener_stats.json'
    flattener_file: str = 'flattener.json'
    segmentation_stats_file: str = 'segmentation_stats.csv'
    segmentation_param_file: str = 'segmentation_params.json'
    mother_bud_props_file: str = 'mother_bud_props.csv'
    mother_bud_model_file: str = 'mother_bud_model.pkl'
    cnn_set: Tuple[str, ...] = ('msd_d80', 'unet_4s')
    cnn_fn: Union[None, str] = None
    batch_size: int = 8
    in_memory: bool = True
    xy_out: int = 80
    target_pixel_size: float = 0.263
    substacks: Union[None, int] = None



@contextmanager
def augmented_generator(gen: ImageLabel, aug: Augmenter):
    # Save the previous augmenter if any
    saved_aug = gen.aug
    gen.aug = aug
    try:
        yield gen
    # Todo: add except otherwise there might be an issue of there is an error?
    finally:
        gen.aug = saved_aug

