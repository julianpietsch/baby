import json
from pathlib import Path
from typing import Union

from baby.hypermodels import get_hypermodel
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
    """
    Class that chooses the best hyperparameters for a specific model-type.

    Note: uses Keras-tuner Hypermodels -- requires tensorflow 2

    Outputs: a set of parameters for that form of model, into a file.
    If using tensorflow 1: these parameters need to be set by the user by
    default.
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
                                      directory='./',
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
            self._tuners[self.cnn.name] = instantiate_tuner(self.cnn,
                                                            **self._tuner_params)
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
