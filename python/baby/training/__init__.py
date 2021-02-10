"""Utilities for training the steps of BABY separately or sequentially.

The training module combines all of the utilities and structures needed to
train a BABY segmentation framework from scratch.
It includes the following trainers
* `SmoothingModelTrainer`: hyper-parameters for smooth data augmentation
* `FlattenerTrainer`: hyper-parameters for distinguishing CNN outputs
* `HyperParameterTrainer`: CNN hyper-parameters
* `CNNTrainer`: CNN using gradient descent to optimize for a given loss
* `SegmentationTrainer`: hyper-parameters for post-processing of CNN Output
into cell instances and attributes

Given the appropriate inputs, each of these can be trained separately.
This is useful for fine-tuning or re-training parts separately.

For training the entire framework at once, it is recommended to use the
`BabyTrainer` class, which is also aliased as `Nursery`.
"""
from .smoothing_model_trainer import SmoothingModelTrainer
from .flattener_trainer import FlattenerTrainer
import tensorflow as tf
if tf.__version__.startswith('1'):
    from .v1_hyper_parameter_trainer import HyperParamV1 \
        as HyperParameterTrainer
else:
    from .hyper_parameter_trainer import HyperParameterTrainer
from .cnn_trainer import CNNTrainer

from .training import *
from .utils import fix_tf_rtx_gpu_bug

