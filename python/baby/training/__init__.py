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

