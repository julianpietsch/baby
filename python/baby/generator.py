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
import json
from pathlib import Path
from contextlib import contextmanager
import numpy as np
from functools import namedtuple
from itertools import repeat
from tensorflow.keras.utils import Sequence
from tqdm import tqdm
from matplotlib import pyplot as plt
from PIL import Image

from .io import load_tiled_image
from .augmentation import Augmenter
from .preprocessing import standard_norm
from .visualise import colour_segstack

# Following are used by ImgLblInMem and should be deprecated
from .preprocessing import robust_norm as preprocess_brightfield
from .preprocessing import segoutline_flattening as preprocess_segim


ImageLabelShapes = namedtuple('ImageLabelShapes', ('input', 'output'))

class ImageLabel(Sequence):
    def __init__(self, paths, batch_size, aug, preprocess=None,
                 in_memory=False, balanced_sampling=False,
                 use_sample_weights=False, n_jobs=4):
        """Generator for training image-label pairs.

        Arguments:
            paths: an iterable generating tuples of filename pairs
                specifying source image and target image
            batch_size: integer specifying desired size for each batch
            aug: an `Augmenter` object for transforming the images
            preprocess: a single callable or tuple of callables (one for each
                file of pair); None specifies the default `standard_norm`
            in_memory: whether or not to load all images into memory
            balanced_sampling: whether to increase sampling of images based on
                their shape relative to the output crop
        """

        self.batch_size = batch_size

        assert callable(aug), '"aug" must be a callable'
        self.aug = aug
        self.n_jobs = n_jobs

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
        
        self.balanced_sampling = balanced_sampling
        self.use_sample_weights = use_sample_weights

        # Initialise ordering
        self.on_epoch_end()

    def __len__(self):
        return int(np.ceil(sum(self.nsamples) / float(self.batch_size)))

    @property
    def shapes(self):
        if len(self.paths) == 0:
            return ImageLabelShapes(tuple(), tuple())

        img, lbl = self.get_by_index(0)
        if type(lbl) == tuple and len(lbl) == 2:
            lbl, _ = lbl
        Nbatch = (self.batch_size,)
        return ImageLabelShapes(Nbatch + img.shape, Nbatch + lbl.shape)

    def _collect_size_info(self):
        self._rawshapes = []
        self._pixelsizes = []
        self._weights = []
        for imgfile, lblfile in self.paths:
            img = Image.open(imgfile)
            img_info = json.loads(img.info.get('Description', '{}'))
            if 'tilesize' in img_info and 'ntiles' in img_info:
                img_shape = tuple(img_info['tilesize']) + (img_info['ntiles'],)
            else:
                img_shape = img.size + (1,)
            self._rawshapes.append(img_shape)

            lbl = Image.open(lblfile)
            lbl_info = json.loads(lbl.info.get('Description', '{}'))
            self._pixelsizes.append(lbl_info.get('pixel_size'))
            self._weights.append(lbl_info.get('weight',1))
        self._weights = np.array(self._weights)

    @property
    def rawshapes(self):
        if not hasattr(self, '_rawshapes') or self._rawshapes is None:
            self._collect_size_info()
        return self._rawshapes

    @property
    def pixelsizes(self):
        if not hasattr(self, '_pixelsizes') or self._pixelsizes is None:
            self._collect_size_info()
        return self._pixelsizes

    @property
    def weights(self):
        if not hasattr(self, '_weights') or self._weights is None:
            self._collect_size_info()
        return self._weights

    @property
    def target_pixel_size(self):
        val = None
        if hasattr(self.aug, 'target_pixel_size'):
            val = self.aug.target_pixel_size
        return val

    @property
    def nsamples(self):
        if ~self.balanced_sampling:
            return np.ones(len(self.rawshapes), dtype='int').tolist()

        aug_shape = self.shapes.input[1:]
        usepxsz = self.target_pixel_size is not None
        if usepxsz:
            aug_size = np.array(aug_shape, dtype=float)
            aug_size[:2] *= self.target_pixel_size
        nsamples = []
        for in_shape, pxsz in zip(self.rawshapes, self.pixelsizes):
            if pxsz and usepxsz:
                in_size = np.array(in_shape, dtype=float)
                in_size[:2] *= pxsz
                szratio = np.floor_divide(in_size, aug_size)
            else:
                szratio = np.floor_divide(in_shape, aug_shape)
            szratio = szratio.astype(int)
            nsamples.append(np.prod(np.maximum(szratio, 1)))
        return nsamples

    def on_epoch_end(self):
        # Shuffle samples for next epoch
        self.ordering = np.repeat(np.arange(len(self.nsamples)),
                                  self.nsamples)
        np.random.shuffle(self.ordering)

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

    def parallel_get_indices(self, inds, n_jobs=None):
        if n_jobs is None:
            n_jobs = self.n_jobs
        passthrough = lambda img, lbl: (img, lbl)
        img_lbl_pairs = [self.get_by_index(i, aug=passthrough) for i in inds]
        from joblib import Parallel, delayed
        return Parallel(n_jobs=n_jobs)(
            delayed(self.aug)(img, lbl) for img, lbl in img_lbl_pairs)

    def __getitem__(self, idx):
        Nbatch = self.batch_size
        current_batch = self.ordering[idx * Nbatch:(idx + 1) * Nbatch]

        if self.n_jobs > 1:
            img_batch, lbl_batch = zip(*self.parallel_get_indices(current_batch))
        else:
            img_batch, lbl_batch = zip(*[self.get_by_index(i) for i in
                                         current_batch])

        lbl_batch = [np.dsplit(lbl, lbl.shape[2]) for lbl in lbl_batch]

        img_batch = np.array(img_batch)
        lbl_batch = [np.array(lw) for lw in zip(*lbl_batch)]

        if self.use_sample_weights:
            return img_batch, lbl_batch, self.weights[current_batch]
        else:
            return img_batch, lbl_batch

    def plot_sample(self, i=0, figsize=3):
        """Plot a sample batch from the generator

        This function assumes that the assigned Augmenter produces label
        images that can be concatenated along a new axis.
        """
        img_batch, lbl_batch = self[i][:2]
        lbl_batch = np.concatenate(lbl_batch, axis=3)

        n_sections = img_batch.shape[3]
        n_targets = lbl_batch.shape[3]

        target_names = repeat(None, n_targets)
        edge_inds = None
        if hasattr(self.aug, 'targetgenfunc'):
            if hasattr(self.aug.targetgenfunc, 'names'):
                target_names = self.aug.targetgenfunc.names()
            if hasattr(self.aug.targetgenfunc, 'targets'):
                edge_inds = np.flatnonzero(
                    [t.prop == 'edge' for t in self.aug.targetgenfunc.targets])

        ncol = len(img_batch)
        nrow =  n_sections + n_targets
        fig, axs = plt.subplots(nrow, ncol,
                                figsize=(figsize * ncol, figsize * nrow))

        # Plot img sections first...
        for axrow, section in zip(axs[:n_sections],
                                  np.split(img_batch, n_sections, axis=3)):
            for ax, img, lbl in zip(axrow, section, lbl_batch):
                ax.imshow(img, cmap='gray')
                if edge_inds is not None:
                    ax.imshow(colour_segstack(lbl[..., edge_inds], dw=True))
                ax.grid(False)
                ax.set(xticks=[], yticks=[])

        # ...then plot targets
        for axrow, target, name in zip(axs[n_sections:],
                                       np.split(lbl_batch, n_targets, axis=3),
                                       target_names):
            for ax, lbl in zip(axrow, target):
                ax.imshow(lbl, cmap='gray')
                ax.grid(False)
                ax.set(xticks=[], yticks=[])
                if name is not None:
                    ax.set_title(name)

        return fig, axs



@contextmanager
def augmented_generator(gen: ImageLabel, aug: Augmenter):
    # Save the previous augmenter if any
    saved_aug = gen.aug
    gen.aug = aug
    try:
        yield gen
    # Todo: add except otherwise there might be an issue of there is an error?
    finally:
        gen.aug = saved_aug


class AugmentedGenerator(Sequence):
    """Wraps a generator with an alternative augmenter.

    Args:
        gen (ImageLabel): Generator to wrap.
        aug (augmentation.Augmenter): Augmenter to use.
    """
    def __init__(self, gen, aug):
        self._gen = gen
        self._aug = aug
        self.on_epoch_end()

    def __len__(self):
        with augmented_generator(self._gen, self._aug) as g:
            return len(g)

    @property
    def batch_size(self):
        return self._gen.batch_size

    @property
    def shapes(self):
        with augmented_generator(self._gen, self._aug) as g:
            return g.shapes

    @property
    def nsamples(self):
        with augmented_generator(self._gen, self._aug) as g:
            return g.nsamples

    @property
    def rawshapes(self):
        return self._gen.rawshapes

    @property
    def pixelsizes(self):
        return self._gen.pixelsizes

    def on_epoch_end(self):
        with augmented_generator(self._gen, self._aug) as g:
            g.on_epoch_end()

    @property
    def ordering(self):
        return self._gen.ordering

    @property
    def n_pairs(self):
        return self._gen.n_pairs

    def get_by_index(self, i, aug=None):
        if aug is None:
            with augmented_generator(self._gen, self._aug) as g:
                return g.get_by_index(i)
        else:
            return self._gen.get_by_index(i, aug=aug)

    def parallel_get_indices(self, inds, **kwargs):
        with augmented_generator(self._gen, self._aug) as g:
            return g.parallel_get_indices(inds, **kwargs)

    def __getitem__(self, idx):
        with augmented_generator(self._gen, self._aug) as g:
            return g[idx]

    def plot_sample(self, *args, **kwargs):
        with augmented_generator(self._gen, self._aug) as g:
            return g.plot_sample(*args, **kwargs)


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
