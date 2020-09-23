import json
from pathlib import Path


class DummyHyperModel:
    """
    Dummy version of HyperModel for TFv1 with which kerastuner cannot be used
    """
    unet_default_params = dict(depth=4, layer_size=4, batchnorm=True,
                               dropout=0.)
    msd_default_params = dict(depth=80, width=1, n_dilations=4, dilation=1,
                              batchnorm=True)

    default_parameters = dict(unet=unet_default_params,
                              msd=msd_default_params)

    def __init__(self, name, parameters=None):
        self.name = name
        if parameters is None:
            self.parameters = self.default_parameters[self.name]
        else:
            self.parameters = parameters


class HyperParamV1:
    """
    Class that chooses the best hyperparameters for a specific model-type.

    """
    def __init__(self, save_dir: Path, cnn_set, *args, **kwargs):
        self.save_dir = save_dir
        self._tuners = dict()
        self._best_parameters = dict()

        self.cnn_set = cnn_set
        self.cnn = self.cnn_set[0]

    @property
    def cnn(self):
        return self._cnn

    @cnn.setter
    def cnn(self, cnn):
        hyper_param_file = cnn + '/hyperparameters.json'
        hyper_param_file = self.save_dir / hyper_param_file
        try:
            with open(hyper_param_file, 'r') as fd:
                params = json.load(fd)
            self._cnn = DummyHyperModel(name=cnn, parameters=params)
        except FileNotFoundError:
            self._cnn = DummyHyperModel(name=cnn)

    @property
    def cnn_dir(self):
        cnn_dir = self.save_dir / self.cnn.name
        if not cnn_dir.exists():
            cnn_dir.mkdir()
        return cnn_dir

    @property
    def best_parameters(self):
        return self.cnn.parameters

    def save_best_parameters(self, filename):
        with open(filename, 'w') as fd:
            json.dump(self.best_parameters, fd)

    def search(self, **kwargs):
        """
        Dummy search that just saves the default parameters to file.
        ```
        :param kwargs:
        :return:
        """
        # Get best parameters
        self.save_best_parameters(self.cnn_dir / 'hyperparameters.json')
