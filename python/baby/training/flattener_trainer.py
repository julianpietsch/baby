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
"""Optimising the hyper-parameters of the `SegmentationFlattener`"""
import json
import pathlib
from itertools import chain

import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import binary_erosion

from baby.augmentation import Augmenter
from baby.errors import BadProcess, BadType, BadParam
from baby.generator import ImageLabel
from baby.preprocessing import dwsquareconn, SegmentationFlattening
from baby.utils import find_file
from baby.visualise import colour_segstack

from .utils import TrainValProperty, standard_augmenter
from .smoothing_model_trainer import SmoothingModelTrainer


def _generate_flattener_stats_inner(segs, nerode, keep_zero):
    segs = segs > 0
    # First remove any sections that may have been augmented away
    # NB: the ScalingAugmenter does remove most of these at the rough_crop
    # step, but there will still be many excluded after the second crop
    segs = segs[..., segs.any(axis=(0,1))]
    s_sizes = segs.sum(axis=(0,1)).tolist()
    nsegs = segs.shape[2]
    esizes = [[] for s in range(nsegs)]
    overlap_sizes = [[] for _ in nerode]
    for e in nerode:
        for s0 in range(nsegs):
            seg0 = segs[..., s0]
            n0 = int(seg0.sum())
            esizes[s0].append(n0)
            if n0 == 0:
                continue
            for s1 in range(s0 + 1, nsegs):
                seg1 = segs[..., s1]
                # Calculate number of overlapping pixels
                nO = np.sum(seg0 & seg1)
                # Calculate fraction overlap
                fO = float(nO / np.sum(seg0 | seg1))
                if fO > 0 or keep_zero:
                    sizes = tuple(sorted([s_sizes[s0], s_sizes[s1]]))
                    if keep_zero:
                        overlap_sizes[e].append(sizes + (fO, nO))
                    else:
                        overlap_sizes[e].append(sizes + (fO,))
        segs = binary_erosion(segs, dwsquareconn)
    return esizes, overlap_sizes


def _batch_parallel_generator(gen, sample_inds, batch_size=8, n_jobs=4):
    for i in range(np.ceil(len(sample_inds) / batch_size).astype('int')):
        yield gen.parallel_get_indices(
            sample_inds[batch_size*i:batch_size*(i+1)], n_jobs=n_jobs)


def _generate_flattener_stats(gen: ImageLabel,
                              max_erode: int,
                              keep_zero=False,
                              batch_size=16,
                              n_jobs=4) -> dict:
    """ Generates flattener statistics of the data output by the generator.

    This function measures the size (in pixels) of segmentation mask
    overlaps and the number of binary erosions necessary to remove the overlap
    between any pairs of cells in the data contained in the generator.

    :param gen: the data generator on which to compute statistics
    :param max_erode: the maximum number of erosions to consider
    :return: the statistics in a dictionary with keys `overlap_sizes` and
    `erosion_sizes`. `overlap_sizes` is a list over 0 to `max_erode`
    applications of binary erosion, where each element contains a list of cell
    pairs that have non-zero overlap. Each pair is expressed as a tuple
    specifying the area of each cell (smaller first) and the fraction overlap.
    `erosion_sizes` is a list of lists, specifying, for each cell mask, its
    size after applying the indexed number of erosions.
    """
    nerode = list(range(max_erode + 1))

    sample_inds = np.repeat(np.arange(len(gen.nsamples)), gen.nsamples)
    sample_gen = _batch_parallel_generator(gen, sample_inds, n_jobs=n_jobs)
    n_batches = np.ceil(len(sample_inds) / batch_size).astype('int')
    erosion_sizes = []
    overlap_sizes = [[] for _ in range(len(nerode))]
    from joblib import Parallel, delayed
    from tqdm import trange
    for i in trange(n_batches):
        gen_batch = gen.parallel_get_indices(
            sample_inds[batch_size*i:batch_size*(i+1)], n_jobs=n_jobs)
        e_sizes, o_sizes = zip(*Parallel(n_jobs=n_jobs)(
            delayed(_generate_flattener_stats_inner)(segs, nerode, keep_zero)
            for _, segs in gen_batch))
        erosion_sizes.extend(chain(*e_sizes))
        o_sizes = [chain(*e) for e in zip(*o_sizes)]
        for e, o in zip(o_sizes, overlap_sizes):
            o.extend(e)

    return {'overlap_sizes': overlap_sizes, 'erosion_sizes': erosion_sizes}


def _group_sizes(es, thresh, pad=0):
    """Returns the upper and lower bounds for cell group sizes.

    #TODO not sure what the parameters mean
    :param es:
    :param thresh:
    :param pad:
    :return:
    """
    return ([s for s in es if s[0] < thresh + pad],
            [s for s in es if s[0] >= thresh - pad])


def _group_overlapping(os, thresh, pad=0):
    """Filter overlap stats into groups according to a fuzzy size threshold

    :param os: overlap statistics  for for particular level of erosion, that
    is, a list of tuples specifying, for each pair of overlapping outlines,
    the area of cell 1, area of cell 2 (always at least as large as cell 1)
    and fraction overlap (overlap area divided by union area).
    :param thresh: size threshold to apply
    :param pad: padding around threshold for double-grouping pairs
    :return: A tuple of two lists, the first containing stats for all pairs
    where both cells are smaller than the threshold + padding, the second
    containing stats for all pairs where both cells are larger than the
    threshold - padding.
    """
    return ([
        (x, y, w) for x, y, w in os if x < thresh + pad and y < thresh + pad
    ], [(x, y, w) for x, y, w in os if x >= thresh - pad and y >= thresh - pad
       ])


def _best_overlapping(overlapping, erosion_sizes, min_size, min_size_frac):
    """Return overlap stats for highest level of erosion without losing cells

    Binary erosion is valid if it does not reduce the area of any cells below
    `min_size`. Any elements failing this test will be removed from
    `overlapping`. The first element in erosion lists is never removed (it is
    assumed to contain stats before erosion is applied). The return value is
    the element of overlapping for which erosion is maximised.

    #TODO not sure what these parameters mean
    :param overlapping:
    :param erosion_sizes:
    :param min_size:
    :return:
    """
    # Rearrange `erosion_sizes` by number of applied erosions
    sz_erosions = list(zip(*erosion_sizes))
    min_median_sz = np.median(sz_erosions[0]) * min_size_frac
    # Erosions are invalid if any cells drop below the minimum allowed size,
    # or if the median size drops below a fraction of the original median
    e_invalid = [np.any(np.array(e) < min_size)
                 or np.median(e) < min_median_sz
                 for e in sz_erosions[:0:-1]]
    # Return only overlap stats for valid numbers of erosions
    o_valid = [o for o, e in zip(overlapping[:0:-1], e_invalid) if not e]
    o_valid += [overlapping[0]]
    return o_valid[0]


def _sum_group_overlapping(os, thresh, pad=0):
    """#TODO not sure what this function does

    #TODO not sure what these parameters mean
    :param os:
    :param thresh:
    :param pad:
    :return:
    """
    return tuple(
        sum([w
             for _, _, w in og])
        for og in _group_overlapping(os, thresh, pad=pad))


def _find_best_fgroup_split(os, edges, pad=0):
    """#TODO not sure what this function does

    #TODO not sure what these parameters mean
    :param os:
    :param edges:
    :param pad:
    :return:
    """
    overlaps = [
        _sum_group_overlapping(os, thresh, pad=pad) for thresh in edges
    ]
    return min(zip(edges, overlaps), key=lambda x: sum(x[1]))


def _best_nerode(szg, min_size):
    """#TODO not sure what this function does

    #TODO not sure what these parameters mean
    :param szg:
    :param min_size:
    :return:
    """
    ne = [
        n for n, e in list(enumerate(zip(*szg)))[:0:-1]
        if not any([c < min_size for c in e])
    ]
    return ne[0] if len(ne) > 0 else 0


class FlattenerTrainer:
    """Optimises the hyper-parameters for the `SegmentationFlattener`.

    #TODO describe method for optimisation

    Args:
        shared_params (utils.SharedParameterContainer): Training and
            segmentation parameters as provided by
            :py:class:`utils.SharedParameterContainer`.
        shared_data (utils.SharedDataContainer): Access to training data.
        ssm_trainer (SmoothingModelTrainer): Trainer from which to obtain a
            smoothing sigma model.
    """
    def __init__(self, shared_params, shared_data, ssm_trainer):
        self._shared_params = shared_params
        self._shared_data = shared_data
        self._ssm_trainer = ssm_trainer
        self._flattener = None
        self._stats = None

    @property
    def save_dir(self):
        return self._shared_params.save_dir

    @property
    def stats_file(self):
        """File in which to save derived data for training flattener"""
        return (self.save_dir /
                self._shared_params.parameters.flattener_stats_file)

    @property
    def flattener_file(self):
        return (self.save_dir /
                self._shared_params.parameters.flattener_file)

    def generate_flattener_stats(self, max_erode: int = 5, n_jobs=None):
        """Generate overlap and erosion statistics for augmented data in input.

        Saves results to file specified in :py:attr:`stats_file`.

        Args:
            max_erode: the maximum allowed number of erosions used to
        generate erosion values
        :return: None, saves results to `self.stats_file`
        """
        dummy_flattener = lambda x, y: x
        ssm = self._ssm_trainer.model
        params = self._shared_params.parameters
        # NB: use isval=True als for training aug since we do not need extra
        # augmentations for calibrating the flattener
        aug = standard_augmenter(ssm, dummy_flattener, params, isval=True)
        train_gen, val_gen, _ = self._shared_data.gen_with_aug(aug)

        if n_jobs is None:
            n_jobs = params.n_jobs
        fs_train = _generate_flattener_stats(train_gen, max_erode,
                                             n_jobs=n_jobs)
        fs_val = _generate_flattener_stats(val_gen, max_erode,
                                           n_jobs=n_jobs)
        with open(self.stats_file, 'wt') as f:
            json.dump({'train': fs_train, 'val': fs_val}, f)
        self._stats = None  # trigger reload of property

    @property
    def stats(self) -> TrainValProperty:
        """The last statistics computed, loaded from :py:attr:`stats_file`.

        Raises:
            BadProcess: If the file does not exist.
        """
        if self._stats is None:
            if not self.stats_file.exists():
                raise BadProcess('flattener stats have not been generated')
            with open(self.stats_file, 'rt') as f:
                self._stats = json.load(f)
        # Fixme: this recreates an object at each call, can we just save the
        #  object?
        return TrainValProperty(self._stats.get('train', {}),
                                self._stats.get('val', {}))

    def plot_stats(self, nbins=30, nrows=1, sqrt_area=False):
        """Plot a histogram of cell overlap statistics of the training set.

        # TODO describe what the plot means
        # TODO add an image as an example

        :param nbins: binning of data, passed to `matplotlib.pyplot.hist2d`
        :return: None, saves the resulting figure under `self.save_dir /
        "flattener_stats.png"`
        """
        overlapping = self.stats.train.get('overlap_sizes', []).copy()
        if sqrt_area:
            # Transform all areas by sqrt
            for i, o in enumerate(overlapping):
                x, y, f = zip(*o)
                overlapping[i] = list(zip(np.sqrt(x), np.sqrt(y), f))
        max_erode = len(overlapping)
        ncols = int(np.ceil(max_erode / nrows))
        fig, axs = plt.subplots(nrows, ncols, figsize=(4 * ncols, 4 * nrows))
        axs = axs.flatten()
        x, y, _ = zip(*overlapping[0])
        max_size = max(x + y)
        for ax, (e, os) in zip(axs, enumerate(overlapping)):
            if len(os) > 0:
                x, y, w = zip(*os)
            else:
                x, y, w = 3 * [[]]
            ax.hist2d(x,
                      y,
                      bins=nbins,
                      weights=w,
                      range=[[0, max_size], [0, max_size]])
            ax.plot((0, max_size), (0, max_size), 'r')
            ax.set_title('nerosions = {:d}'.format(e))
        fig.savefig(self.save_dir / 'flattener_stats.png')

    @property
    def flattener(self) -> SegmentationFlattening:
        """The last flattener saved to file.
        :return: a segmentation flattener
        :raise: BadProcess, if no flattener has been assigned.
        """
        if self._flattener is None:
            if self.flattener_file.is_file():
                f = SegmentationFlattening()
                f.load(self.flattener_file)
                self._flattener = f
            else:
                raise BadProcess('The "flattener" has not been assigned yet')
        return self._flattener

    @flattener.setter
    def flattener(self, f):
        if isinstance(f, str):
            flattener_file = find_file(f, self.save_dir, 'flattener')
            f = SegmentationFlattening()
            f.load(flattener_file)
        if not isinstance(f, SegmentationFlattening):
            raise BadType(
                '"flattener" must be of type "baby.preprocessing.SegmentationFlattening"'
            )
        f.save(self.flattener_file)
        self._flattener = f

    def fit(self, nbins=30, min_size=10, min_size_frac=0., pad_frac=0.03,
            bud_max=200, sqrt_area=False, overlaps=None):
        """Optimise the parameters of the `SegmentationFlattener` based on
        previously computed statistics.

        # TODO define optimisation method

        # TODO missing some unknown parameters
        :param nbins:
        :param min_size: the minimum size of a segmented object for it to be
        considered a cell.
        :param pad_frac:
        :param bud_max: the maximum size (in pixels) of cells considered buds.
        :return: None, saves the optimal parameters to the flattener file
        :raises: BadProcess if no statistics have been computed yet.
        """
        if pad_frac > 0.25 or pad_frac < 0:
            raise BadParam('"pad_frac" must be between 0 and 0.2')

        # Load generated stats for training data
        overlapping = self.stats.train.get('overlap_sizes', []).copy()
        erosion_sizes = self.stats.train.get('erosion_sizes', []).copy()
        if len(overlapping) == 0 or len(erosion_sizes) == 0 or \
                len(list(zip(*erosion_sizes))) != len(overlapping):
            raise BadProcess(
                '"flattener_stats.json" file appears to be corrupted')

        if sqrt_area:
            # Transform all areas by sqrt
            min_size = np.sqrt(min_size)
            erosion_sizes = np.sqrt(erosion_sizes).tolist()
            for i, o in enumerate(overlapping):
                x, y, f = zip(*o)
                overlapping[i] = list(zip(np.sqrt(x), np.sqrt(y), f))

        # Find the best single split point by brute force iteration over a
        # binned version of the training data

        # Use distribution of cell sizes to determine binning
        o_noerode = overlapping[0]
        x, y, _ = zip(*o_noerode)
        max_size = max(x + y)
        pad = max([pad_frac * max_size, min_size])
        edges = np.linspace(pad, max_size - pad, nbins)[1:-1]

        # Use overlap stats at maximum valid level of erosion
        o_maxerode = _best_overlapping(overlapping, erosion_sizes, min_size,
                                       min_size_frac)
        # Then iterate over the thresholds (edges) to find which split
        # minimises the overlap fraction
        split0, w0 = _find_best_fgroup_split(o_maxerode, edges, pad=pad)

        # Use the threshold to split all overlap and erosion stats into low
        # and high groups:
        ogL, ogH = zip(
            *[_group_overlapping(o, split0, pad=pad) for o in overlapping])
        szgL, szgH = _group_sizes(erosion_sizes, split0, pad=pad)

        # And again use the overlap stats at maximum valid level of erosion
        ogL = _best_overlapping(ogL, szgL, min_size, min_size_frac)
        ogH = _best_overlapping(ogH, szgH, min_size, min_size_frac)

        w_ogL = sum([w for _, _, w in ogL])
        w_ogH = sum([w for _, _, w in ogH])
        if w_ogL == w_ogH:
            w_ogL, w_ogH = w0

        if w_ogL > w_ogH:
            edges = np.linspace(pad, split0 - pad, nbins)
            og = ogL
        else:
            edges = np.linspace(split0 + pad, max_size, nbins)
            og = ogH

        split1, _ = _find_best_fgroup_split(og, edges, pad=pad)

        splits = list(sorted([split0, split1]))

        szg0, szg12 = _group_sizes(erosion_sizes, splits[0], pad=pad)
        szg1, szg2 = _group_sizes(szg12, splits[1], pad=pad)

        ne0 = _best_nerode(szg0, min_size)
        ne1 = _best_nerode(szg1, min_size)
        ne2 = _best_nerode(szg2, min_size)

        if sqrt_area:
            untransform = lambda x: int(np.round(np.square(x)))
        else:
            untransform = lambda x: int(np.round(x))

        flattener = SegmentationFlattening()

        flattener.addGroup('small', upper=untransform(splits[0] + pad))
        flattener.addGroup('medium',
                           lower=untransform(splits[0] - pad),
                           upper=untransform(splits[1] + pad))
        flattener.addGroup('large', lower=untransform(splits[1] - pad))
        flattener.addGroup('buds', upper=bud_max, budonly=True)
        if overlaps == 'all':
            flattener.addGroup('all')

        flattener.addTarget('lge_inte', 'large', 'interior', nerode=ne2)
        flattener.addTarget('lge_edge', 'large', 'edge')
        flattener.addTarget('mid_inte', 'medium', 'interior', nerode=ne1)
        flattener.addTarget('mid_edge', 'medium', 'edge')
        flattener.addTarget('sml_inte', 'small', 'filled', nerode=ne0)
        flattener.addTarget('sml_edge', 'small', 'edge')
        flattener.addTarget('bud_neck', 'buds', 'budneck')
        if overlaps == 'all':
            flattener.addTarget('all_ovlp', 'all', 'overlap')

        self.flattener = flattener

    @property
    def default_gen(self):
        """Get default data generators using the current flattener."""
        ssm = self._ssm_trainer.model
        params = self._shared_params.parameters
        t_aug = standard_augmenter(ssm, self.flattener, params, isval=False)
        v_aug = standard_augmenter(ssm, self.flattener, params, isval=True)
        return self._shared_data.gen_with_aug((t_aug, v_aug, v_aug))

    def plot_default_gen_sample(self, i=0, figsize=3, validation=False):
        g = self.default_gen.val if validation else self.default_gen.train
        img_batch, lbl_batch = g[i]
        lbl_batch = np.concatenate(lbl_batch, axis=3)

        f = self.flattener
        target_names = f.names()
        edge_inds = np.flatnonzero([t.prop == 'edge' for t in f.targets])

        ncol = len(img_batch)
        nrow = len(target_names) + 1
        fig = plt.figure(figsize=(figsize * ncol, figsize * nrow))
        for b, (bf, seg) in enumerate(zip(img_batch, lbl_batch)):
            plt.subplot(nrow, ncol, b + 0 * ncol + 1)
            plt.imshow(bf[:, :, 0], cmap='gray')
            plt.imshow(colour_segstack(seg[:, :, edge_inds], dw=True))
            plt.grid(False)
            plt.xticks([])
            plt.yticks([])

            for i, name in enumerate(target_names):
                plt.subplot(nrow, ncol, b + (i + 1) * ncol + 1)
                plt.imshow(seg[:, :, i], cmap='gray')
                plt.grid(False)
                plt.xticks([])
                plt.yticks([])
                plt.title(name)

        fig.savefig(self.save_dir / '{}_generator_sample.png'.format(
            'validation' if validation else 'training'))

