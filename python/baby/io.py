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
import json
import re

from pathlib import Path
from typing import Union
from fnmatch import translate as glob_to_re
from os import walk
from itertools import groupby, chain, repeat
from collections import namedtuple, Counter
import numpy as np
import random
import pandas as pd
from PIL.PngImagePlugin import PngInfo
from imageio import imread, imwrite
from sklearn.model_selection import train_test_split

from .errors import LayoutError, UnpairedImagesError
from .utils import PathEncoder


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
    imwrite(filename, tImg, format='png', pnginfo=meta,
            prefer_uint8=tImg.dtype != 'uint16')


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
        self._metadata = None

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
        self._metadata = None  # reset metadata if validation data changes

    @property
    def ncells(self):
        ncells_tuple = namedtuple('ncells', 'training, validation')
        meta = self.metadata
        if len(meta) == 0:
            return ncells_tuple(0, 0)
        ntrainval = Counter(
            chain(*[
                list(repeat(t, n))
                for n, t in zip(map(len, meta.cellLabels), meta.train_val)
            ]))
        return ncells_tuple(**ntrainval)

    @property
    def metadata(self):
        if getattr(self, '_metadata', None) is None:
            if len(self.training) == 0 and len(self.validation) == 0:
                return []
            trainvalpairs = {
                'training': self.training,
                'validation': self.validation
            }
            sub_metadata = []
            for k, pairs in trainvalpairs.items():
                pair_meta = []
                for _, l in pairs:
                    info = json.loads(imread(l).meta.get(
                        'Description', '{}'))
                    pair_meta.append(
                        {field: value for field, value in info.items()})
                    pair_meta[-1]['cellLabels'] = aslist(
                        pair_meta[-1]['cellLabels'])
                    pair_meta[-1]['filename'] = l
                    pair_meta[-1]['train_val'] = k

                sub_metadata.append(pd.DataFrame(pair_meta))
                sub_metadata[-1]['list_index'] = sub_metadata[-1].index
            self._metadata = pd.concat(sub_metadata, axis=0, ignore_index=True)
            # TODO
            # The following shouldn't be there- nursery is meant to handle any
            # tile size (which is distinct from pixel size)
            # self._metadata = self.metadata.loc[
            #     np.array([x[0] for x in self.metadata['tilesize']])==81]
            self._metadata.sort_values(['experimentID', 'position', 'trap', 'tp'], inplace=True)
            self._metadata.set_index(['experimentID', 'position', 'trap'], inplace=True)
            self._metadata_tp = self._metadata.set_index('tp', append=True)
            # TODO: assert that all index has the same  trainval field
        return self._metadata

    @property
    def traps(self, chunk_size=4, min_tp=2, trap_together=True):
        ''' Group the data in chunks to use for cell tracking random forest cross-validation'''
        # df = self._metadata[self._metadata['train_val']=='training'] #TODO Reconsider this filter
        traps = pd.DataFrame(self._metadata.sort_values(['tp']).groupby(
            ['experimentID', 'position', 'trap'])['tp'].apply(list))
        # Some of the next parts are disabled while we find out why there are repeated metadatas
        # traps = traps.sample(frac=1, random_state=42) # shuffle dataframe
        traps['tp_uniq'] = traps['tp'].apply(np.unique) #TODO remove this when metadata issue is fixed
        traps['indices'] = find_continuous_tps(traps['tp_uniq'], chunk_size)
        traps['cont'] = [l[inds] for i, (l, inds) in enumerate(traps[['tp_uniq', 'indices']].values)]
        #TODO ALAN: Add split operation

        if not trap_together:  # shuffle after splitting rn chunks?
            traps = traps.sample(frac=1, random_state=24)

        # remove non-continuous values
        traps = traps.loc[traps['cont'].apply(len) > min_tp]  # clean up
        self._traps = traps
        return self._traps
        # return tp_chunks

    def load(self, filename, base_dir: Union[Path, str] = './'):
        if isinstance(base_dir, str):
            base_dir = Path(base_dir)
        with open(filename, 'rt') as f:
            trainval = json.load(f)
        if 'train_data' in trainval and 'val_data' in trainval:
            train_pairs = trainval['train_data']
            val_pairs = trainval['val_data']
        else:
            train_pairs = trainval.get('training', [])
            val_pairs = trainval.get('validation', [])
        train_pairs = [(base_dir / img, base_dir / lbl)
                       for img, lbl in train_pairs]
        val_pairs = [(base_dir / img, base_dir / lbl)
                     for img, lbl in val_pairs]
        self.training = train_pairs
        self.validation = val_pairs

    def save(self, filename, base_dir: Union[Path, str] = './'):
        if isinstance(base_dir, str):
            base_dir = Path(base_dir)
        # Create a string first to catch exceptions and avoid overwriting
        # the train_val_pairs file
        try:
            data = json.dumps(

                {
                    'training': [tuple(path.relative_to(base_dir)
                                       for path in p)
                                 for p in self.training],
                    'validation': [tuple(path.relative_to(base_dir)
                                         for path in p)
                                   for p in self.validation]
                },
                cls=PathEncoder)
            with open(filename, 'wt') as f:
                f.write(data)

        except ValueError as e:
            raise e

    def add_from(self,
                 base_dir,
                 img_suffix='brightfield',
                 lbl_suffix='segoutlines',
                 test_size=0.25,
                 group_by=('experimentID', 'position', 'trap')):
        only_outlines = False
        if img_suffix is None:
            img_suffix='segoutlines'
            only_outlines = True


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
        grouped = [{t: list(ims)
                    for t, ims in groupby(sorted(p, key=first), key=first)}
                    for _, p in groupby(matches, key=prefix)]
        if not only_outlines: # replace imgs with img if only using outlines
            pairs = [(p['img'][0][2], p['lbl'][0][2])
                        for p in grouped
                 if len(p.get('img', [])) == 1 and len(p.get('lbl', [])) == 1]
        else:
            pairs = [(p['img'][0][2], p['img'][0][2])
                 for p in grouped
                 if len(p.get('img', [])) == 1]

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
        self.training += [p for p, g in zip(pairs, pair_groups) if
                          g in train_groups]
        self.validation += [p for p, g in zip(pairs, pair_groups) if
                            g in val_groups]

    def __repr__(self):
        return 'TrainValPairs: {:d} training and {:d} validation pairs'.format(
            len(self.training), len(self.validation))


class TrainValTestPairs(object):

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
        self._metadata = None

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
        self._metadata = None  # reset metadata if validation data changes

    @property
    def testing(self):
        if not hasattr(self, '_test_pairs'):
            self._test_pairs = []
        return self._test_pairs

    @testing.setter
    def testing(self, pairs):
        if not isinstance(pairs, list):
            raise ValueError('"testing" must be a list')
        self._test_pairs = pairs
        self._metadata = None

    @property
    def ncells(self):
        ncells_tuple = namedtuple('ncells', 'training, validation, testing')
        meta = self.metadata
        if len(meta) == 0:
            return ncells_tuple(0, 0, 0)
        ntrainval = Counter(
            chain(*[
                list(repeat(t, n))
                for n, t in zip(map(len, meta.cellLabels), meta.train_val)
            ]))
        ntrainval.setdefault('training', 0)
        ntrainval.setdefault('validation', 0)
        ntrainval.setdefault('testing', 0)
        return ncells_tuple(**ntrainval)

    @property
    def metadata(self):
        if getattr(self, '_metadata', None) is None:
            if not np.any([True for x in [self.training, self.validation, self.testing] if x]):
                return []
            trainvalpairs = {
                'training': self.training,
                'validation': self.validation,
                'testing': self.testing
            }
            sub_metadata = []
            for k, pairs in trainvalpairs.items():
                pair_meta = []
                for _, l in pairs:
                    info = json.loads(imread(l).meta.get(
                        'Description', '{}'))
                    pair_meta.append(
                        {field: value for field, value in info.items()})
                    pair_meta[-1]['cellLabels'] = aslist(
                        pair_meta[-1]['cellLabels'])
                    pair_meta[-1]['filename'] = l
                    pair_meta[-1]['train_val'] = k

                sub_metadata.append(pd.DataFrame(pair_meta))
                sub_metadata[-1]['list_index'] = sub_metadata[-1].index

            self._metadata = pd.concat(sub_metadata, axis=0, ignore_index=True)

            self._metadata.sort_values(['experimentID', 'position', 'trap', 'tp'], inplace=True)
            self._metadata.set_index(['experimentID', 'position', 'trap'], inplace=True)
            self._metadata_tp = self._metadata.set_index('tp', append=True)

            # TODO: assert that all indices have the same  trainval field
            #
        return self._metadata

    @property
    def traps(self, chunk_size=4, min_tp=2, trap_together=True):
        ''' Group the data in chunks to use for cell tracking random forest cross-validation'''

        traps = pd.DataFrame(self._metadata.sort_values(['tp']).groupby(
            ['experimentID', 'position', 'trap'])['tp'].apply(list))

        traps['tp_uniq'] = traps['tp'].apply(np.unique) #TODO remove this when metadata issue is fixed

        traps['indices'] = find_continuous_tps(traps['tp_uniq'], chunk_size)

        traps['cont'] = [l[inds] for i, (l, inds) in enumerate(traps[['tp_uniq', 'indices']].values)]

        #TODO ALAN: Add split operation

        if not trap_together:  # shuffle after splitting rn chunks?
            traps = traps.sample(frac=1, random_state=24)

        # remove non-continuous values
        traps = traps.loc[traps['cont'].apply(len) > min_tp]  # clean up
        self._traps = traps
        return self._traps
        # return tp_chunks

    def load(self, filename, base_dir: Union[Path, str] = './'):
        if isinstance(base_dir, str):
            base_dir = Path(base_dir)
        with open(filename, 'rt') as f:
            trainvaltest = json.load(f)
        if 'train_data' in trainvaltest and 'val_data' in trainvaltest:
            train_pairs = trainvaltest['train_data']
            val_pairs = trainvaltest['val_data']
            test_pairs = trainvaltest['test_data']
        else:
            train_pairs = trainvaltest.get('training', [])
            val_pairs = trainvaltest.get('validation', [])
            test_pairs = trainvaltest.get('testing', [])
        train_pairs = [(base_dir / img, base_dir / lbl)
                       for img, lbl in train_pairs]
        val_pairs = [(base_dir / img, base_dir / lbl)
                     for img, lbl in val_pairs]
        test_pairs = [(base_dir / img, base_dir / lbl)
                     for img, lbl in test_pairs]
        self.training = train_pairs
        self.validation = val_pairs
        self.testing = test_pairs

    def save(self, filename, base_dir: Union[Path, str] = './'):
        if isinstance(base_dir, str):
            base_dir = Path(base_dir)
        # Create a string first to catch exceptions and avoid overwriting
        # the train_val_pairs file
        try:
            data = json.dumps(

                {
                    'training': [tuple(path.relative_to(base_dir)
                                       for path in p)
                                 for p in self.training],
                    'validation': [tuple(path.relative_to(base_dir)
                                         for path in p)
                                   for p in self.validation],
                    'testing': [tuple(path.relative_to(base_dir)
                                         for path in p)
                                   for p in self.testing]
                },
                cls=PathEncoder)
            with open(filename, 'wt') as f:
                f.write(data)

        except ValueError as e:
            raise e

    def add_from(self,
                 base_dir,
                 img_suffix='brightfield',
                 lbl_suffix='segoutlines',
                 val_size=0.2,
                 test_size=0.2,
                 group_by=('experimentID', 'position', 'trap')):
        only_outlines = False
        if img_suffix is None:
            img_suffix='segoutlines'
            only_outlines = True


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
        grouped = [{t: list(ims)
                    for t, ims in groupby(sorted(p, key=first), key=first)}
                    for _, p in groupby(matches, key=prefix)]
        if not only_outlines: # replace imgs with img if only using outlines
            pairs = [(p['img'][0][2], p['lbl'][0][2])
                        for p in grouped
                 if len(p.get('img', [])) == 1 and len(p.get('lbl', [])) == 1]
        else:
            pairs = [(p['img'][0][2], p['img'][0][2])
                 for p in grouped
                 if len(p.get('img', [])) == 1]

        # Ensure that we only add pairs that are not already present
        existing = set(self.training + self.validation + self.testing)
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
        npairs = len(unique_groups)
        # train_groups, val_groups = train_test_split(unique_groups,
        #                                             test_size=test_size)
        # Split val and test into equally-sized groups
        random.shuffle(unique_groups)
        train_size = 1 - val_size - test_size
        train_groups, val_groups, test_groups = np.split(
            unique_groups,
            [int(npairs*train_size), int(npairs * (1-test_size))])

        reformat = lambda exp, pos, trap : (exp, int(pos), int(trap))

        train_groups = set([reformat(*t) for t in train_groups])
        val_groups = set([reformat(*t) for t in val_groups])
        test_groups = set([reformat(*t) for t in test_groups])

        # Add new pairs to the existing train-val split
        self.training += [p for p, g in zip(pairs, pair_groups) if
                          g in train_groups]
        self.validation += [p for p, g in zip(pairs, pair_groups) if
                            g in val_groups]
        self.testing += [p for p, g in zip(pairs, pair_groups) if
                            g in test_groups]

    def __repr__(self):
        return 'TrainValPairs: {:d} training, {:d} validation and {:d} testing pairs'.format(
            len(self.training), len(self.validation), len(self.testing))

# ---------------- HELPER FUNCTIONS -----------------------

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

def find_continuous_tps(uniq_traps, chunk_size):
    tp_distance = uniq_traps.apply(lambda x: np.subtract(x[1:], x[:-1]))
    tp_distance.apply(lambda x: [0 if dif > 1  else dif for dif in x])
    indices_groups = tp_distance.apply(
        lambda x: [(label, sum(1 for _ in group)) for label, group in groupby(x)])
    # traps['valid_indices'] = traps['valid_chunks'].apply(lambda x: find_indices(x, chunk_size))

    indices = indices_groups.apply(lambda x: find_indices(x, chunk_size))
    return indices

def find_indices(groups, chunk_size, return_max=True):
    '''Receives the output of contiguous ones and returns a list with the
    indices of contiguous timepoints'''
    indices = []
    curind = 0
    for dif, n in groups:
        if dif == 1 and n>=chunk_size:
            indices.append([curind+i for i in range(n)])
        curind += n
    if return_max and indices:
        sizes = [len(i) for i in indices]
        maxind = sizes.index(np.max(sizes))
        indices = indices[maxind]
    return indices
