import json
import re
from pathlib import Path
from fnmatch import translate as glob_to_re
from os import walk
from itertools import groupby
from functools import namedtuple
import numpy as np
from imageio import imread, imwrite
from PIL.PngImagePlugin import PngInfo
from sklearn.model_selection import train_test_split

from .utils import PathEncoder
from .errors import LayoutError, UnpairedImagesError

import pandas as pd
from collections import Counter
from itertools import chain

def load_tiled_image(filename):
    tImg = imread(filename)
    info = json.loads(tImg.meta.get('Description', '{}'))
    tw, th = info.get('tilesize', tImg.shape[0:2])
    nt = info.get('ntiles', 1)
    nr, nc = info.get('layout', (1, 1))
    nc_final_row = np.mod(nt, nc)
    img = np.zeros((tw, th, nt), dtype=tImg.dtype)
    for i in range(nr):
        i_nc = nc_final_row if i + 1 == nr and nc_final_row > 0 else nc
        for j in range(i_nc):
            ind = i * nc + j
            img[:, :, ind] = tImg[i * tw:(i + 1) * tw, j * th:(j + 1) * th]
    return img, info


def save_tiled_image(img, filename, info={}, layout=None):
    if layout is not None and len(layout) != 2:
        raise LayoutError('"layout" must a 2-tuple')

    tw, th, nt = img.shape
    if not layout or layout == (None, None):
        nr = int(np.floor(np.sqrt(nt)))
        nc = None
    else:
        nr, nc = layout

    if nr is None:
        nr = int(np.ceil(nt / nc))
    if nc is None:
        nc = int(np.ceil(nt / nr))

    info['tilesize'] = (tw, th)
    info['ntiles'] = nt
    info['layout'] = (nr, nc)

    nc_final_row = np.mod(nt, nc)
    tImg = np.zeros((tw * nr, th * nc), dtype=img.dtype)
    for i in range(nr):
        i_nc = nc_final_row if i + 1 == nr and nc_final_row > 0 else nc
        for j in range(i_nc):
            ind = i * nc + j
            tImg[i * tw:(i + 1) * tw, j * th:(j + 1) * th] = img[:, :, ind]

    meta = PngInfo()
    meta.add_text('Description', json.dumps(info))
    imwrite(filename, tImg, format='png', pnginfo=meta, prefer_uint8=False)


def load_paired_images(filenames, typeA='Brightfield', typeB='segoutlines'):
    re_imlbl = re.compile(r'^(.*)_(' + typeA + r'|' + typeB + r')\.png$')
    # For groupby to work, the list needs to be sorted; also has the side
    # effect of ensuring filenames is no longer a generator
    filenames = sorted(filenames)
    matches = [re_imlbl.match(f.name) for f in filenames]
    valid = filter(lambda m: m[0], zip(matches, filenames))
    grouped = {
        k: {m.group(2): f for m, f in v
           } for k, v in groupby(valid, key=lambda m: m[0].group(1))
    }
    valid = [
        set(v.keys()).issuperset({typeA, typeB}) for v in grouped.values()
    ]
    if not all(valid):
        raise UnpairedImagesError
    return {
        l: {t: load_tiled_image(f) for t, f in g.items()
           } for l, g in grouped.items()
    }


class TrainValPairs(object):

    @property
    def training(self):
        if not hasattr(self, '_train_pairs'):
            self._train_pairs = []
        return self._train_pairs

    @training.setter
    def training(self, pairs):
        if not isinstance(pairs, list):
            raise ValueError('"training" must be a list')
        self._train_pairs = pairs
        self._ncells = None

    @property
    def validation(self):
        if not hasattr(self, '_val_pairs'):
            self._val_pairs = []
        return self._val_pairs

    @validation.setter
    def validation(self, pairs):
        if not isinstance(pairs, list):
            raise ValueError('"validation" must be a list')
        self._val_pairs = pairs
        self._ncells = None

    @property
    def ncells(self):
        if not hasattr(self, '_ncells') or not self._ncells:
            ntrainval = Counter(self._metadata['train_val'])
            ncells_tuple = namedtuple('ncells', 'training, validation')
            self._ncells = ncells_tuple(**ntrainval)
        return self._ncells

    @property
    def metadata(self):
        if not hasattr(self, '_metadata'):
            trainvalpairs = {
                'training': self.training,
                'validation': self.validation
            }
            sub_metadata = []
            for k, pairs in trainvalpairs.items():
                meta = []
                for _, l in pairs:
                    info = json.loads(imread(l).meta.get('Description', '{}'))
                    meta.append({field : value for field, value in info.items()})
                    meta[-1]['cellLabels'] = aslist(meta[-1]['cellLabels'])
                    meta[-1]['filename'] = l
                    meta[-1]['train_val'] = k

                sub_metadata.append(pd.DataFrame(meta))
                self._metadata = pd.concat(sub_metadata, axis=0, ignore_index=True)
        return self._metadata

    def load(self, filename):
        with open(filename, 'rt') as f:
            trainval = json.load(f)
        if 'train_data' in trainval and 'val_data' in trainval:
            train_pairs = trainval['train_data']
            val_pairs = trainval['val_data']
        else:
            train_pairs = trainval.get('training', [])
            val_pairs = trainval.get('validation', [])
        train_pairs = [(Path(img), Path(lbl)) for img, lbl in train_pairs]
        val_pairs = [(Path(img), Path(lbl)) for img, lbl in val_pairs]
        self.training = train_pairs
        self.validation = val_pairs

    def save(self, filename):
        with open(filename, 'wt') as f:
            json.dump(
                {
                    'training': self.training,
                    'validation': self.validation
                },
                f,
                cls=PathEncoder)

    def add_from(self,
                 base_dir,
                 img_suffix='brightfield',
                 lbl_suffix='segoutlines',
                 test_size=0.25,
                 group_by=('experimentID', 'position', 'trap')):

        # Find image files, and ensure label files are paired by prefix
        # NB: the following code is written such that pairs are found in a
        # case insensitive manner
        re_img = re.compile(r'^(.*)' + img_suffix + r'$', re.IGNORECASE)
        re_lbl = re.compile(r'^(.*)' + lbl_suffix + r'$', re.IGNORECASE)
        png_files = sorted(Path(base_dir).rglob('*.png'))
        matches = [(re_img.search(f.stem), re_lbl.search(f.stem), f)
                   for f in png_files]
        matches = [('img', im, f) if im else ('lbl', lm, f)
                   for im, lm, f in matches
                   if im or lm]
        # Group by path and prefix (i.e., excluding suffix):
        prefix = lambda x: str(x[2].parent) + x[1].group(1)
        first = lambda x: x[0]
        grouped = [{
            t: list(ims)
            for t, ims in groupby(sorted(p, key=first), key=first)
        }
                   for _, p in groupby(matches, key=prefix)]
        pairs = [(p['img'][0][2], p['lbl'][0][2])
                 for p in grouped
                 if len(p.get('img', [])) == 1 and len(p.get('lbl', [])) == 1]

        # Ensure that we only add pairs that are not already present
        existing = set(self.training + self.validation)
        pairs = [p for p in pairs if p not in existing]
        if len(pairs) == 0:
            return

        # Choose a split that ensures separation by group keys and avoids,
        # e.g., splitting same cell but different time points
        info = [
            json.loads(imread(l).meta.get('Description', '{}'))
            for _, l in pairs
        ]
        pair_groups = [
            tuple(i.get(f, 'missing_' + str(e))
                  for f in group_by)
            for e, i in enumerate(info)
        ]
        unique_groups = list(set(pair_groups))
        train_groups, val_groups = train_test_split(unique_groups,
                                                    test_size=test_size)
        train_groups, val_groups = set(train_groups), set(val_groups)

        # Add new pairs to the existing train-val split
        self.training.extend(
            p for p, g in zip(pairs, pair_groups) if g in train_groups)
        self.validation.extend(
            p for p, g in zip(pairs, pair_groups) if g in val_groups)

    def __repr__(self):
        return 'TrainValPairs: {:d} training and {:d} validation pairs'.format(
            len(self.training), len(self.validation))


def aslist(val):
    '''
    Helper function useful to convert int cell_ids to lists
    '''
    # Convenience fn to convert x to a list if not one already
    if type(val) is int:
        val = [val]
    elif val is None:
        val = []

    return val
