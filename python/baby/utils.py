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
import time
import json
from functools import partial
import warnings
from pathlib import Path
from typing import NamedTuple, Any
from importlib import import_module
from packaging.version import Version
import numpy as np
import sklearn
from sklearn.base import BaseEstimator
from sklearn.tree._tree import Tree
from baby.errors import BadParam, BadFile, BadModel


class Timer(object):
    """Based on code from sourceforge..."""
    def __init__(self, name=None):
        self.name = name

    def __enter__(self):
        self.tstart = time.time()

    def __exit__(self, type, value, traceback):
        if self.name:
            print('[%s]' % self.name,)
        print('Elapsed: %s' % (time.time() - self.tstart))


class PathEncoder(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, Path):
            return str(o)
        else:
            return json.JSONEncoder.default(self, o)


def named_obj(name):
    def wrap(obj):
        obj._baby_name = name
        return obj
    return wrap


def get_name(obj):
    return getattr(obj, '_baby_name', obj.__name__)


def jsonify(obj):
    if hasattr(obj, 'toJSON'):
        return obj.toJSON()
    elif hasattr(obj, 'dtype') and hasattr(obj, 'tolist'):
        return obj.tolist()
    elif isinstance(obj, tuple):
        return {'_python_tuple': [jsonify(v) for v in obj]}
    if isinstance(obj, set):
        return {'_python_set': list(obj)}
    elif isinstance(obj, dict):
        return {k: jsonify(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [jsonify(v) for v in obj]
    else:
        return obj


def NamedTupleToJSON(self):
    return {
        '_python_NamedTuple': jsonify(self._asdict()),
        '__module__': self.__class__.__module__,
        '__class__': self.__class__.__name__
    }


def EncodableNamedTuple(obj):
    obj.toJSON = NamedTupleToJSON
    return obj


def as_python_object(obj):
    if '_python_NamedTuple' in obj:
        obj_class = getattr(import_module(obj['__module__']), obj['__class__'])
        return obj_class(**obj['_python_NamedTuple'])
    if '_python_tuple' in obj:
        return tuple(obj['_python_tuple'])
    elif '_python_set' in obj:
        return set(obj['_python_set'])
    else:
        return obj


ACCEPTED_MODELS = (
    'RandomForestClassifier', 
    'DecisionTreeClassifier',
    'Tree'
)


def jsonify_sklearn_model(obj, arrays=None):
    if arrays is None:
        arrays = {}
        return jsonify_sklearn_model(obj, arrays), arrays
    recurse = partial(jsonify_sklearn_model, arrays=arrays)
    
    if isinstance(obj, np.ndarray):
        if obj.dtype == 'O' and all([type(o) == str for o in obj]):
            return {'_numpy_ndarray': obj.tolist()}
        else:
            name = f'array{len(arrays)}'
            arrays[name] = obj
            return {'_numpy_ndarray': name}
    if isinstance(obj, np.int64):
        return int(obj)
    elif isinstance(obj, (BaseEstimator, Tree)):
        T = type(obj)
        return {
            f'_sklearn_object': recurse(obj.__getstate__()),
            '__module__': T.__module__,
            '__name__': T.__name__
        }
    elif isinstance(obj, tuple):
        return {'_python_tuple': [recurse(v) for v in obj]}
    elif isinstance(obj, dict):
        return {k: recurse(v) for k, v in obj.items() if not k.startswith('base_estimator')}
    elif isinstance(obj, list):
        return [recurse(v) for v in obj]
    else:
        return obj


def as_sklearn_model(obj, arrays={}):
    if '_numpy_ndarray' in obj:
        array = obj['_numpy_ndarray']
        if type(array) == str:
            return arrays[obj['_numpy_ndarray']]
        elif type(array) == list:
            return np.array(array, dtype='O')
        else:
            raise BadFile('Invalid model file, or file is corrupt')
    if '_sklearn_object' in obj:
        mod = obj['__module__']
        if not mod.startswith('sklearn.'):
            raise BadFile('Invalid model file, or file is corrupt')
        cls = obj['__name__']
        if cls not in ACCEPTED_MODELS:
            raise BadFile('Invalid model file, or file is corrupt')
        skl_obj = getattr(import_module(mod), cls)
        obj = obj['_sklearn_object']
        if cls == 'Tree':
            n_features = obj['nodes']['feature'].max() + 1
            _, n_outputs, n_classes = obj['values'].shape
            skl_obj = skl_obj(n_features, np.array([n_classes]), n_outputs)

            # Adapt nodes dtype to the current version of sklearn
            skl_nodes = skl_obj.__getstate__()['nodes']
            obj_nodes = obj['nodes']
            if not set(obj_nodes.dtype.names).issubset(skl_nodes.dtype.names):
                raise BadModel('tree dtypes are not a subset of those for '
                               'current version of scikit-learn')
            obj_dtfields = {k: v for k, v in obj_nodes.dtype.fields.items()}
            skl_dtfields = {k: v for k, v in skl_nodes.dtype.fields.items()
                            if k in obj_dtfields}
            if obj_dtfields != skl_dtfields:
                raise BadModel('tree dtypes are not a subset of those for '
                               'current version of scikit-learn')
            
            new_nodes = np.zeros_like(skl_nodes, shape=obj_nodes.shape)
            for fname in obj_nodes.dtype.names:
                new_nodes[fname] = obj_nodes[fname]
            obj['nodes'] = new_nodes

            if Version(sklearn.__version__) >= Version("1.4.1"):
                # Need to renormalise the ._value attribute from weighted
                # absolute count of number of samples to weighted fraction of
                # total number of samples.
                # See https://scikit-learn.org/stable/whats_new/v1.4.html
                norm = obj['values'].sum(axis=2, keepdims=True)
                obj['values'] = obj['values'] / norm
        else:
            skl_obj = skl_obj()
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="Trying to unpickle estimator",
                category=UserWarning
            )
            skl_obj.__setstate__(obj)
        return skl_obj
    elif '_python_tuple' in obj:
        return tuple(obj['_python_tuple'])
    else:
        return obj


def save_sklearn_model(model, filename):
    """Save a Scikit-learn model in NumPy .npz format

    Avoids pickling the model, but potentially requires additional maintenance
    to ensure compatibilty with future Scikit-learn updates.
    """
    if not Path(filename).suffix == '.npz':
        raise BadParam('model file name must end in ".npz"')
    npz_dict, npz_arrays = jsonify_sklearn_model(model)
    np.savez_compressed(filename,
                        model_json=json.dumps(npz_dict),
                        **npz_arrays)


def load_sklearn_model(filename):
    """Load a Scikit-learn saved in NumPy .npz format

    Loads models as saved by :py:func:`save_sklearn_model`.
    """
    loader = np.load(filename)
    object_hook = partial(as_sklearn_model, arrays=loader)
    return json.loads(str(loader['model_json']),
                      object_hook=object_hook)


def find_file(filename, default_dir, argname=None):
    filepath = Path(filename)
    if not filepath.is_file():
        filepath = Path(default_dir) / filepath
    if not filepath.is_file():
        msg = 'Could not find file "{}"'.format(filename)
        if argname:
            msg += ' specified for "{}"'.format(argname)
        raise BadParam(msg)
    return filepath


def batch_iterator(a, batch_size=8):
    for i in range(np.ceil(len(a) / batch_size).astype('int')):
        yield a[batch_size*i:batch_size*(i+1)]


def split_batch_pred(outputs):
    outputs = np.stack(outputs)
    return [o.squeeze() for o in np.split(outputs, outputs.shape[1], axis=1)]


def schedule_steps(epoch, steps):
    for step in steps:
        if step[1] > epoch:
            print("Setting learning rate to {}".format(step[0]))
            return step[0]
    print("Setting learning rate to {}".format(steps[-1][0]))
    return steps[-1][0]
