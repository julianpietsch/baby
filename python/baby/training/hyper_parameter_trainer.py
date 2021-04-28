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
import json
from pathlib import Path
from typing import Union

from baby.training.hypermodels import get_hypermodel
from kerastuner import RandomSearch, Hyperband, BayesianOptimization, Tuner, \
    HyperModel

from .utils import augmented_generator


def instantiate_tuner(model, method='random', **kwargs):
    method = method.lower()
    if method == 'random':
        return RandomSearch(model, **kwargs, project_name=model.name)
    elif method == 'hyperband':
        return Hyperband(model, **kwargs, project_name=model.name)
    elif method == 'bayesian':
        return BayesianOptimization(model, **kwargs, project_name=model.name)
    else:
        raise (ValueError, 'Method {} is not supported.'.format(method))


class HyperParameterTrainer:
    """Optimises hyperparameters for different CNN architectures.

    Note: uses Keras-tuner Hypermodels -- requires tensorflow 2

    Outputs: a set of parameters for that form of model, into a file.
    If using tensorflow 1: default parameters are used but they can be set
    by the user under "hyperparameters.json" under each CNN architecture's
    dedicated directory.
    """

    def __init__(self, save_dir: Path, cnn_set, gen, aug, outputs,
                 tuner_params: Union[Tuner, None, dict, str] = None):
        self.save_dir = save_dir
        self.aug = aug
        self.gen = gen
        self.outputs = outputs
        self._init_hypermodels(cnn_set)
        self._tuners = dict()

        self._cnn = None
        self.cnn = self.cnn_set[0]

        if tuner_params is None:
            self._tuner_params = dict(method='random',
                                      objective='val_loss',
                                      max_trials=3,
                                      directory=self.save_dir,
                                      overwrite=False)
        elif isinstance(tuner_params, str):
            with open(tuner_params, 'r') as fd:
                self._tuner_params = json.load(fd)
        elif isinstance(tuner_params, dict):
            # Todo: save tuner to file
            self._tuner_params = tuner_params
        self._best_parameters = dict()

    def _init_hypermodels(self, cnn_set):
        self.cnn_set = [None] * len(cnn_set)
        for i, cnn in enumerate(cnn_set):
            if isinstance(cnn, str):
                # Get from Hypermodel file by name
                shapes = self.gen.train.shapes
                self.cnn_set[i] = get_hypermodel(cnn, shapes.input[1:],
                                                 self.outputs)
            elif not isinstance(cnn, HyperModel):
                raise TypeError("Non-standard CNNs must be in the form of a "
                                "keras.Hypermodel, "
                                "received {}".format(type(cnn)))

    @property
    def cnn(self):
        return self._cnn

    @cnn.setter
    def cnn(self, cnn):
        if isinstance(cnn, HyperModel):
            self._cnn = cnn
        else:
            raise TypeError("CNN instance must be of type "
                            "kerastuner.Hypermodel.")

    @property
    def cnn_dir(self):
        cnn_dir = self.save_dir / self.cnn.name
        if not cnn_dir.exists():
            cnn_dir.mkdir()
        return cnn_dir

    @property
    def tuner(self):
        if self.cnn.name not in self._tuners:
            cnn_tuner_params = self._tuner_params.copy()
            cnn_tuner_params['directory'] = cnn_tuner_params['directory'] / self.cnn.name
            self._tuners[self.cnn.name] = instantiate_tuner(self.cnn,
                                                            **cnn_tuner_params)
        return self._tuners[self.cnn.name]

    # Todo: set tuner parameters, not tuner itself
    @tuner.setter
    def tuner(self, tuner):
        if isinstance(tuner, str):
            with open(tuner, 'r') as fd:
                params = json.load(fd)
            self._tuner_params = params
        elif isinstance(tuner, dict):
            self._tuner_params = tuner
        # Invalidate current tuners
        self._tuners = dict()

    @property
    def best_parameters(self):
        if self.cnn.name not in self._best_parameters:
            self._best_parameters[self.cnn.name] = \
                self.tuner.get_best_hyperparameters()[0].values
        return self._best_parameters[self.cnn.name]

    def save_best_parameters(self, filename):
        with open(filename, 'w') as fd:
            json.dump(self.best_parameters, fd)

    def use_defaults(self):
        filename = self.cnn_dir / 'hyperparameters.json'
        with open(filename, 'w') as fd:
            json.dump(self.cnn.defaults, fd)

    def search(self, epochs=100, steps_per_epoch=10, validation_steps=10,
               **kwargs):
        """
        Runs search with the instance's generator and tuner.

        Keyword arguments are those you would normally use in a `model.fit` call.
        For instance:
        ```python
        tuner.search(generator,
                     steps_per_epoch=train_steps,
                     epochs=args.nb_epochs,
                     callbacks=[early_stopping, checkpointer, tensor_board],
                     validation_data=val_generator,
                     validation_steps=val_steps,
                     verbose=1,
                     workers=args.nb_workers,
                     class_weight=class_weight)
        ```
        :param kwargs:
        :return:
        """
        # TODO: introduce a tensorflow logs file by default
        with augmented_generator(self.gen.train, self.aug.train) as train_gen:
            with augmented_generator(self.gen.val, self.aug.val) as val_gen:
                self.tuner.search(train_gen,
                                  steps_per_epoch=steps_per_epoch,
                                  epochs=epochs,
                                  validation_data=val_gen,
                                  validation_steps=validation_steps,
                                  **kwargs)
        # Get best parameters
        self.save_best_parameters(self.cnn_dir / 'hyperparameters.json')

    def history(self, cnn_name):
        # TODO extract data from tensorboard events and aggregate into a
        # history for each loss for each trial
        # TODO save as a parameter
        pass
    
    def plot_hyperparameter_training(self, cnn_name):
        # TODO get tensorboard output, aggregate, plot
        pass

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


def extract(dpath, subpath):
    scalar_accumulators = [EventAccumulator(str(dpath / dname /
    subpath)).Reload().scalars for dname in filter(Path.is_dir, dpath.iterdir()) if 'plugins' not in str(dname)]

    # Filter non event files
    scalar_accumulators = [scalar_accumulator for
    scalar_accumulator in scalar_accumulators if
    scalar_accumulator.Keys()]

    # Get and validate all scalar keys
    all_keys = [tuple(scalar_accumulator.Keys()) for
            scalar_accumulator in scalar_accumulators]
    assert len(set(all_keys)) == 1, "All runs need to have the same scalar keys. There are mismatches in {}".format(all_keys)
    keys = all_keys[0]
            
    all_scalar_events_per_key = [[scalar_accumulator.Items(key) 
        for scalar_accumulator in scalar_accumulators] 
        for key in keys]
    
    # Get and validate all steps per  key
    all_steps_per_key = [[tuple(scalar_event.step 
         for scalar_event in scalar_events) 
         for scalar_events in all_scalar_events]
         for all_scalar_events in all_scalar_events_per_key]
        
    for i, all_steps in enumerate(all_steps_per_key):
        assert len(set(all_steps)) == 1, "For scalar {} the step numbering or count doesn't match. Step count for all runs: {}".format( keys[i], [len(steps) for steps in all_steps])
    steps_per_key = [all_steps[0] for all_steps in all_steps_per_key]
    # Get values per step per key
    values_per_key = [[[scalar_event.value for scalar_event in scalar_events] 
        for scalar_events in all_scalar_events]
        for all_scalar_events in all_scalar_events_per_key]
    all_per_key = dict(zip(keys, zip(steps_per_key, values_per_key)))
    return all_per_key

