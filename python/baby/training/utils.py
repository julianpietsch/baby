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


@EncodableNamedTuple
class BabyTrainerParameters(NamedTuple):
    """Global parameters for training BABY models
    """
    train_val_pairs_file: str = 'train_val_pairs.json'
    smoothing_sigma_stats_file: str = 'smoothing_sigma_stats.csv'
    smoothing_sigma_model_file: str = 'smoothing_sigma_model.json'
    flattener_stats_file: str = 'flattener_stats.json'
    flattener_file: str = 'flattener.json'
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

