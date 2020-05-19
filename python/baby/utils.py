import time
import json
from pathlib import Path
import numpy as np


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


class JSONencodable:
    def toJSON(self):
        pass

    @staticmethod
    def fromJSON(obj):
        pass


class ExtendedEncoder(json.JSONEncoder):
    def _iterencode(self, obj, markers=None):
        if isinstance(obj, NamedTuple):
            gen = self._iterencode_dict({
                '_python_NamedTuple': obj._asdict(),
                '__class__': obj.__class__
            })
        elif isinstance(obj, tuple):
            return {'_python_tuple': list(obj)}
        elif isinstance(obj, set):
            return {'_python_set': list(obj)}
        else:
            return json.JSONEncoder.default(self, obj)

def as_python_object(obj):
    if '_python_tuple' in obj:
        return tuple(obj['_python_tuple'])
    elif '_python_set' in obj:
        return set(obj['_python_set'])
    else:
        return obj


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
