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


class DummyHyperModel:
    """
    Dummy version of HyperModel for TFv1 with which kerastuner cannot be used
    """
    unet_default_params = dict(depth=4, layer_size=8, batchnorm=True,
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
