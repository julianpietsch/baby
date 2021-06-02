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
"""Optimising the hyper-parameters of the `SegmentationFlattener`"""
import json
import pathlib

import matplotlib.pyplot as plt
import numpy as np
from baby.augmentation import Augmenter
from baby.errors import BadProcess, BadType, BadParam
from baby.generator import ImageLabel
from baby.preprocessing import dwsquareconn, SegmentationFlattening
from baby.utils import find_file
from scipy.ndimage import binary_erosion

from .utils import augmented_generator, TrainValProperty


def _generate_flattener_stats(gen: ImageLabel,
                              max_erode: int,
                              keep_zero=False) -> dict:
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
    overlap_sizes = [[] for _ in nerode]
    erosion_sizes = []

    for t in range(len(gen.paths)):
        _, segs = gen.get_by_index(t)
        nsegs = segs.shape[2]
        segs = segs > 0
        s_sizes = [int(segs[..., s].sum()) for s in range(nsegs)]
        esizes = [[] for s in range(nsegs)]
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
        erosion_sizes.extend(esizes)

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


def _best_overlapping(overlapping, erosion_sizes, min_size):
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
    # Erosions are invalid if any cells drop below the minimum allowed size
    e_invalid = [any([c < min_size for c in e]) for e in sz_erosions[:0:-1]]
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

    def __init__(self, save_dir: pathlib.Path, stats_file: str,
                 flattener_file: str):
        """Optimises the hyper-parameters for the `SegmentationFlattener`.



        #TODO describe method for optimisation
        :param save_dir: the base directory in which to save outputs
        :param stats_file: the name of the file in which the stats are saved
        :param flattener_file: the name of the file defining the flattener
        """
        self.save_dir = save_dir
        self.stats_file = self.save_dir / stats_file
        self.flattener_file = self.save_dir / flattener_file
        self._flattener = None
        self._stats = None

    def generate_flattener_stats(self,
                                 train_gen: ImageLabel,
                                 val_gen: ImageLabel,
                                 train_aug: Augmenter,
                                 val_aug: Augmenter,
                                 max_erode: int = 5):
        """Generate overlap and erosion statistics for augmented data in input.

        :param train_gen: the generator of training images and their labels
        :param val_gen: the generator of validation images and their labels
        :param train_aug: the augmenter to use for training images
        :param val_aug: the augmenter to use for validation images
        :param max_erode: the maximum allowed number of erosions used to
        generate erosion values
        :return: None, saves results to `self.stats_file`
        """
        with augmented_generator(train_gen, train_aug) as gen:
            fs_train = _generate_flattener_stats(gen, max_erode)
        with augmented_generator(val_gen, val_aug) as gen:
            fs_val = _generate_flattener_stats(gen, max_erode)
        with open(self.stats_file, 'wt') as f:
            json.dump({'train': fs_train, 'val': fs_val}, f)
        self._stats = None  # trigger reload of property

    @property
    def stats(self) -> TrainValProperty:
        """The last statistics computed, loaded from `self.stats_file`

        :return: The last training and validation statistics computed.
        :raises: BadProcess error if the file does not exist.
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

    def fit(self, nbins=30, min_size=10, pad_frac=0.03, bud_max=200):
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
        overlapping = self.stats.train.get('overlap_sizes', [])
        erosion_sizes = self.stats.train.get('erosion_sizes', [])
        if len(overlapping) == 0 or len(erosion_sizes) == 0 or \
                len(list(zip(*erosion_sizes))) != len(overlapping):
            raise BadProcess(
                '"flattener_stats.json" file appears to be corrupted')

        # Find the best single split point by brute force iteration over a
        # binned version of the training data

        # Use distribution of cell sizes to determine binning
        o_noerode = overlapping[0]
        x, y, _ = zip(*o_noerode)
        max_size = max(x + y)
        pad = max([pad_frac * max_size, min_size])
        edges = np.linspace(pad, max_size - pad, nbins)[1:-1]

        # Use overlap stats at maximum valid level of erosion
        o_maxerode = _best_overlapping(overlapping, erosion_sizes, min_size)
        # Then iterate over the thresholds (edges) to find which split
        # minimises the overlap fraction
        split0, w0 = _find_best_fgroup_split(o_maxerode, edges, pad=pad)

        # Use the threshold to split all overlap and erosion stats into low
        # and high groups:
        ogL, ogH = zip(
            *[_group_overlapping(o, split0, pad=pad) for o in overlapping])
        szgL, szgH = _group_sizes(erosion_sizes, split0, pad=pad)

        # And again use the overlap stats at maximum valid level of erosion
        ogL = _best_overlapping(ogL, szgL, min_size)
        ogH = _best_overlapping(ogH, szgH, min_size)

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

        flattener = SegmentationFlattening()

        flattener.addGroup('small', upper=int(np.round(splits[0] + pad)))
        flattener.addGroup('medium',
                           lower=int(np.round(splits[0] - pad)),
                           upper=int(np.round(splits[1] + pad)))
        flattener.addGroup('large', lower=int(np.round(splits[1] - pad)))
        flattener.addGroup('buds', upper=bud_max, budonly=True)

        flattener.addTarget('lge_inte', 'large', 'interior', nerode=ne2)
        flattener.addTarget('lge_edge', 'large', 'edge')
        flattener.addTarget('mid_inte', 'medium', 'interior', nerode=ne1)
        flattener.addTarget('mid_edge', 'medium', 'edge')
        flattener.addTarget('sml_inte', 'small', 'filled', nerode=ne0)
        flattener.addTarget('sml_edge', 'small', 'edge')
        flattener.addTarget('bud_neck', 'buds', 'budneck')

        flattener.save(self.flattener_file)
        self._flattener = None

    def plot_stats(self, nbins=30):
        """Plot a histogram of cell overlap statistics of the training set.

        # TODO describe what the plot means
        # TODO add an image as an example

        :param nbins: binning of data, passed to `matplotlib.pyplot.hist2d`
        :return: None, saves the resulting figure under `self.save_dir /
        "flattener_stats.png"`
        """
        overlapping = self.stats.train.get('overlap_sizes', [])
        max_erode = len(overlapping)
        fig, axs = plt.subplots(1, max_erode, figsize=(16, 16 / max_erode))
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
