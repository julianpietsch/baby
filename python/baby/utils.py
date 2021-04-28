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
import time
import json
from pathlib import Path
from typing import NamedTuple, Any
from importlib import import_module
import numpy as np
from baby.errors import BadParam


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


def NamedTupleToJSON(self):
    return {
        '_python_NamedTuple': self._asdict(),
        '__module__': self.__class__.__module__,
        '__class__': self.__class__.__name__
    }


def EncodableNamedTuple(obj):
    obj.toJSON = NamedTupleToJSON
    return obj


def jsonify(obj):
    if hasattr(obj, 'toJSON'):
        return obj.toJSON()
    elif hasattr(obj, 'dtype') and hasattr(obj, 'tolist'):
        return obj.tolist()
    elif isinstance(obj, tuple):
        return {'_python_tuple': list(obj)}
    if isinstance(obj, set):
        return {'_python_set': list(obj)}
    elif isinstance(obj, dict):
        return {k: jsonify(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [jsonify(v) for v in obj]
    else:
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
