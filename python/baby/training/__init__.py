from .smoothing_model_trainer import SmoothingModelTrainer
from .flattener_trainer import FlattenerTrainer
from .bud_trainer import BudTrainer
from .track_trainer import TrackTrainer
# Todo: separate based on TFv1 or TFv2
import tensorflow as tf
if tf.__version__.startswith('1'):
    from .v1_hyper_parameter_trainer import HyperParamV1 \
        as HyperParameterTrainer
else:
    from .hyper_parameter_trainer import HyperParameterTrainer
from .cnn_trainer import CNNTrainer

from .training import *

