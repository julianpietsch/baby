import json
import matplotlib.pyplot as plt
import numpy as np
from baby.errors import BadProcess, BadType, BadParam
from baby.preprocessing import dwsquareconn, SegmentationFlattening
from baby.utils import find_file
from scipy.ndimage import binary_erosion

from .utils import augmented_generator, TrainValProperty


def _generate_flattener_stats(gen, max_erode):
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
                    o = float(np.sum(seg0 & seg1) / np.sum(seg0 | seg1))
                    if o > 0:
                        sizes = tuple(sorted([s_sizes[s0], s_sizes[s1]]))
                        overlap_sizes[e].append(sizes + (o,))
            segs = binary_erosion(segs, dwsquareconn)
        erosion_sizes.extend(esizes)

    return {'overlap_sizes': overlap_sizes, 'erosion_sizes': erosion_sizes}


def _group_sizes(es, thresh, pad=0):
    return ([s for s in es if s[0] < thresh + pad],
            [s for s in es if s[0] >= thresh - pad])


def _group_overlapping(os, thresh, pad=0):
    return ([
                (x, y, w) for x, y, w in os if
                x < thresh + pad and y < thresh + pad
            ], [(x, y, w) for x, y, w in os if
                x >= thresh - pad and y >= thresh - pad
                ])


def _best_overlapping(overlapping, erosion_sizes, min_size):
    sz_erosions = list(zip(*erosion_sizes))
    e_invalid = [any([c < min_size for c in e]) for e in
                 sz_erosions[:0:-1]]
    o_valid = [o for o, e in zip(overlapping[:0:-1], e_invalid) if not e]
    o_valid += [overlapping[0]]
    return o_valid[0]


def _sum_group_overlapping(os, thresh, pad=0):
    return tuple(
        sum([w
             for _, _, w in og])
        for og in _group_overlapping(os, thresh, pad=pad))


def _find_best_fgroup_split(os, edges, pad=0):
    overlaps = [
        _sum_group_overlapping(os, thresh, pad=pad) for thresh in edges
    ]
    return min(zip(edges, overlaps), key=lambda x: sum(x[1]))


def _best_nerode(szg, min_size):
    ne = [
        n for n, e in list(enumerate(zip(*szg)))[:0:-1]
        if not any([c < min_size for c in e])
    ]
    return ne[0] if len(ne) > 0 else 0



class FlattenerTrainer:
    def __init__(self, save_dir, stats_file, flattener_file):
        self.save_dir = save_dir
        self.stats_file = self.save_dir / stats_file
        self.flattener_file = self.save_dir / flattener_file
        self._flattener = None
        self._stats = None

    def generate_flattener_stats(self, train_gen, val_gen,
                                 train_aug, val_aug, max_erode=5):
        # Set up temporary flattener
        old_flattener = getattr(self, '_flattener', None)
        self.flattener = lambda x, y: x
        try:
            with augmented_generator(train_gen, train_aug) as gen:
                fs_train = _generate_flattener_stats(gen, max_erode)
            with augmented_generator(val_gen, val_aug) as gen:
                fs_val = _generate_flattener_stats(gen, max_erode)
        finally:
            self.flattener = old_flattener

        with open(self.stats_file, 'wt') as f:
            json.dump({'train': fs_train, 'val': fs_val}, f)
        self._stats = None  # trigger reload of property

    @property
    def stats(self):
        if self._stats is None:
            if not self.stats_file.exists():
                raise BadProcess(
                    'flattener stats have not been generated')
            with open(self.stats_file, 'rt') as f:
                self._stats = json.load(f)
        # Fixme: this recreates an object at each call, can we just save the
        #  object?
        return TrainValProperty(self._stats.get('train', {}),
                                self._stats.get('val', {}))

    @property
    def flattener(self):
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
        if pad_frac > 0.25 or pad_frac < 0:
            raise BadParam('"pad_frac" must be between 0 and 0.2')

        # Find the best split
        overlapping = self.stats.train.get('overlap_sizes', [])
        erosion_sizes = self.stats.train.get('erosion_sizes', [])
        if len(overlapping) == 0 or len(erosion_sizes) == 0 or \
                len(list(zip(*erosion_sizes))) != len(overlapping):
            raise BadProcess(
                '"flattener_stats.json" file appears to be corrupted')

        o_noerode = overlapping[0]
        x, y, _ = zip(*o_noerode)
        max_size = max(x + y)
        pad = max([pad_frac * max_size, min_size])
        edges = np.linspace(pad, max_size - pad, nbins)[1:-1]

        o_maxerode = _best_overlapping(overlapping, erosion_sizes, min_size)
        split0, w0 = _find_best_fgroup_split(o_maxerode, edges, pad=pad)

        ogL, ogH = zip(*[_group_overlapping(o, split0, pad=pad) for o in
                         overlapping])
        szgL, szgH = _group_sizes(erosion_sizes, split0, pad=pad)

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
        flattener.addGroup('medium', lower=int(np.round(splits[0] - pad)),
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
            ax.hist2d(x, y, bins=nbins, weights=w,
                      range=[[0, max_size], [0, max_size]])
            ax.plot((0, max_size), (0, max_size), 'r')
            ax.set_title('nerosions = {:d}'.format(e))
        fig.savefig(self.save_dir / 'flattener_stats.png')