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
from pathlib import Path
import numpy as np
from functools import namedtuple
from tensorflow.python.keras.utils.data_utils import Sequence
from tqdm import tqdm

from .io import load_tiled_image
from .augmentation import Augmenter
from .preprocessing import standard_norm

# Following are used by ImgLblInMem and should be deprecated
from .preprocessing import robust_norm as preprocess_brightfield
from .preprocessing import segoutline_flattening as preprocess_segim


ImageLabelShapes = namedtuple('ImageLabelShapes', ('input', 'output'))

class ImageLabel(Sequence):
    def __init__(self, paths, batch_size, aug, preprocess=None,
                 in_memory=False):
        """Generator for training image-label pairs.

        Arguments:
            paths: an iterable generating tuples of filename pairs
                specifying source image and target image
            batch_size: integer specifying desired size for each batch
            aug: an `Augmenter` object for transforming the images
            preprocess: a single callable or tuple of callables (one for each
                file of pair); None specifies the default `standard_norm`
            in_memory: whether or not to load all images into memory
        """

        self.batch_size = batch_size

        assert callable(aug), '"aug" must be a callable'
        self.aug = aug

        # Check that all required images exist
        self.paths = [(Path(img), Path(lbl)) for img, lbl in paths]
        assert all([all([p.exists() for p in pair]) for pair in self.paths]), \
            'some of the specified images do not exist'

        if preprocess is None:
            self.preprocess = 2 * (standard_norm,)
        elif callable(preprocess):
            self.preprocess = 2 * (preprocess,)
        else:
            self.preprocess = preprocess

        assert (len(self.preprocess) == 2 and
                all([callable(f) for f in self.preprocess])), \
            'preprocess argument specified incorrectly'

        self.in_memory = in_memory
        if self.in_memory:
            self.images = [
                [ppf(*load_tiled_image(img)) for ppf, img
                 in zip(self.preprocess, imgs)] for imgs in tqdm(self.paths)
            ]

        # Initialise ordering
        self.on_epoch_end()

    def __len__(self):
        return int(np.ceil(len(self.paths) / float(self.batch_size)))

    @property
    def shapes(self):
        if not hasattr(self, '_shapes') or not self._shapes:
            if len(self.paths) == 0:
                return ImageLabelShapes(tuple(), tuple())

            img, lbl = self.get_by_index(0)
            Nbatch = (self.batch_size,)
            self._shapes = ImageLabelShapes(Nbatch + img.shape,
                                          Nbatch + lbl.shape)
        return self._shapes

    def on_epoch_end(self):
        # Shuffle samples for next epoch
        Nsamples = len(self.paths)
        self.ordering = np.random.choice(Nsamples, Nsamples, replace=False)

    @property
    def n_pairs(self):
        return len(self.paths)

    def get_by_index(self, i, aug=None):
        if self.in_memory:
            img, lbl = self.images[i]
        else:
            img, lbl = [ppf(*load_tiled_image(im)) for ppf, im
                        in zip(self.preprocess, self.paths[i])]

        if aug is None:
            return self.aug(img, lbl)
        else:
            return aug(img, lbl)

    def __getitem__(self, idx):
        Nbatch = self.batch_size
        current_batch = self.ordering[idx * Nbatch:(idx + 1) * Nbatch]

        img_batch = []
        lbl_batch = []

        for i in current_batch:
            img, lbl = self.get_by_index(i)
            lbl = np.dsplit(lbl, lbl.shape[2])

            img_batch.append(img)
            lbl_batch.append(lbl)

        img_batch = np.array(img_batch)
        lbl_batch = [np.array(lw) for lw in zip(*lbl_batch)]

        return img_batch, lbl_batch


class ImgLblInMem(Sequence):

    def __init__(self, imglbl_paths, batch_size, aug):
        self.batch_size = batch_size

        assert isinstance(aug, Augmenter), '"aug" must be an "Augmenter" object'
        self.aug = aug

        # Load all of the source images into memory
        self.img_lbl_info = []
        for img_path, lbl_path in imglbl_paths:
            img, _ = load_tiled_image(img_path)
            lbl, info = load_tiled_image(lbl_path)
            img = preprocess_brightfield(img)
            lbl = lbl > 0
            self.img_lbl_info.append((img, lbl, info))

        # Initialise ordering
        self.on_epoch_end()

    def __len__(self):
        return int(np.ceil(len(self.img_lbl_info) / float(self.batch_size)))

    def __getitem__(self, idx):
        Nbatch = self.batch_size
        current_batch = self.ordering[idx*Nbatch : (idx+1)*Nbatch]

        img_batch = []
        lbl_batch = []

        for i in current_batch:
            img, lbl, info = self.img_lbl_info[i]
            img, lbl = self.aug.apply(img, lbl)
            lbl = preprocess_segim(lbl, info)
            img_batch.append(img)
            lbl_batch.append(lbl)

        img_batch = np.array(img_batch)
        lbl_batch = np.array(lbl_batch)
        lbl_batch = np.split(lbl_batch, lbl_batch.shape[-1], axis=-1)

        return img_batch, lbl_batch

    def on_epoch_end(self):
        # Shuffle samples for next epoch
        Nsamples = len(self.img_lbl_info)
        self.ordering = np.random.choice(Nsamples, Nsamples, replace=False)
