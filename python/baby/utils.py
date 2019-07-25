import time
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


def batch_iterator(a, batch_size=8):
    for i in range(np.ceil(len(a) / batch_size).astype('int')):
        yield a[batch_size*i:batch_size*(i+1)]


def split_batch_pred(outputs):
    outputs = np.stack(outputs)
    return [o.squeeze() for o in np.split(outputs, outputs.shape[1], axis=1)]
