#!/usr/bin/env python

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

from pathlib import Path
from itertools import repeat, chain
import datetime
from warnings import warn
import pickle
import numpy as np
import pandas as pd
from tqdm import trange
from typing import NamedTuple

from baby.io import load_tiled_image
from baby.tracker.utils import pick_baryfun, calc_barycentre
from baby.training.utils import TrainValProperty, TrainValTestProperty
from baby.errors import BadProcess, BadParam
from .core import CellTracker

from .benchmark import CellBenchmarker

from scipy.ndimage import binary_fill_holes
from skimage.measure import regionprops_table
from skimage.transform import resize
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC#, LinearSVC
# from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import (
    make_scorer, fbeta_score, accuracy_score, balanced_accuracy_score,
    precision_score, recall_score, f1_score
)
import sklearn
if int(sklearn.__version__[0]) > 0:
    from sklearn.metrics import PrecisionRecallDisplay
    plot_precision_recall_curve = PrecisionRecallDisplay.from_estimator
else:
    from sklearn.metrics import plot_precision_recall_curve

class CellTrainer(CellTracker):
    '''
    :meta: Metadata Dataframe
    :traps: Dataframe with cleaned trap locations and their continuous tps
    '''

    def __init__(self, meta, data=None, masks=None,
                 val_masks=None, all_feats2use=None):

        if all_feats2use is None:
            feats2use = ('centroid', 'area', 'minor_axis_length',
                         'major_axis_length', 'convex_area')
            trap_feats = ('baryangle', 'barydist')
            extra_feats = ('distance',)

        else:
            feats2use, trapfeats, extra_feats = all_feats2use

        super().__init__(feats2use = feats2use, trapfeats = trapfeats,
                         extra_feats = extra_feats)
        self.px_size = None

        self.indices = ['experimentID', 'position', 'trap', 'tp']
        self.cindices =  self.indices + ['cellLabels']
        self.data = data
        self.traps = data.traps
        self.meta = data._metadata_tp
        self.process_metadata()
        if masks is None:
            self.masks= [load_tiled_image(fname)[0] for
                         fname  in self.meta['filename']]
        self.process_traps()

    def verify_mask_df_integrity(masks, df):
        nlayers=[mask.shape[2] for mask in masks]
        ncells = [len(x) for x in df['cellLabels'].values]

        for x,(i,j) in enumerate(zip(nlayers, ncells)):
            if i!=j:
                print(x)

    def process_metadata(self):
        '''
        Process all traps (run on finished experiments), combine results with location df and drop
        unused columns.
        '''

        self.meta = self.meta[~self.meta.index.duplicated(keep='first')]
        self.traps = self.traps.explode('cont')
        self.traps['tp'] = self.traps['cont']
        self.traps.set_index('tp', inplace=True, append=True)
        self.clean_indices = self.traps.index

        self.meta = self.meta.loc(axis=0)[self.clean_indices]
        self.meta['ncells'] = [len(i) for i in self.meta['cellLabels'].values]

    def gen_train(self):
        '''
        Generates the data for training using all the loaded images.

        '''

        traps = np.unique([ind[:self.indices.index('trap')+1] for ind in self.traps.index], axis=0)
        traps = [(ind[0], *map(int, ind[1:])) for ind in traps] # str->int conversion
        traps = *map(
            tuple, traps),

        train = []
        for trap in traps: # fill the training dataset with traps' tuples
            trapsets = self.gen_train_from_trap(trap)
            if trapsets:
                train.append(trapsets)
        self.train = np.concatenate(train)

    def gen_train_from_pair(self, pair_loc):
        '''
        Produces a list of training (truth, features) pairs from
        a pair of timepoints in a position

        input

        :pair_loc: (experiment, position, trap, (tp1,tp2)) tuple

        returns

        list of (bool, floats list) tuples
        
        '''
            
        tmp_subdf = self.meta[['list_index', 'cellLabels', 'pixel_size']].loc(axis=0)[pair_loc]

        # Set px_size if available, if not use default value
        px_size = np.unique(tmp_subdf['pixel_size'])
        if len(px_size) > 1:
            if all(np.isnan(x) for x in px_size):
                px_size = [0.263]
            else:
                px_size = [next(x for x in px_size if not np.isnan(x))]
        if len(px_size) == 1:
            px_size = px_size[0]
        if np.isnan(px_size):
            px_size = 0.263

        subdf = tmp_subdf[['list_index', 'cellLabels']]

        if not subdf['cellLabels'].all():
            return None

        truemat = np.equal.outer(*subdf['cellLabels'].values).reshape(-1)

        propsmat = self.df_calc_feat_matrix(pair_loc, px_size=px_size)


        propsmat = propsmat.reshape(
            -1, self.noutfeats)

        return [x for x in zip(truemat, propsmat)]

    def process_traps(self):
        '''
        Generates a region_proprieties DataFrame
        '''

        nindex = []
        props_lst = []
        # Move cell_id index to use calc_feats_from mask fn
        self.masks = [np.moveaxis(mask, 2, 0) for mask in self.masks] 

        # TODO do this more elegantly
        for ind, px_size, (index, 
                  lbl) in zip(self.meta.index, self.meta['pixel_size'].values,
                              enumerate(self.meta['cellLabels'].values)):
            try:
                #check nlabels and ncells agree
                assert (len(lbl)==self.masks[index].shape[0]) | (len(lbl)==0)
            except AssertionError:
                print('nlabels and img mismatch in row: ')
              
            if np.isnan(px_size): #Cover for cases where px_size is nan
                px_size = 0.263

            trapfeats = self.calc_feats_from_mask(self.masks[index], px_size=px_size)

            for cell, feats in zip(lbl, trapfeats):
                nindex.append(ind + (cell, ))
                props_lst.append(feats)

        nindex = pd.MultiIndex.from_tuples(nindex, names=self.cindices)

        self.rprops = pd.DataFrame(np.array(props_lst),
                                   index=nindex, columns = self.tfeats)

        self.rprop_keys = self.rprops.columns

    def gen_train_from_trap(self, trap_loc):
        subdf = self.meta[['list_index', 'cellLabels'
                             ]].loc(axis=0)[trap_loc]
        pairs = [
            trap_loc + tuple((pair, ))
            for pair in zip(subdf.index[:-1], subdf.index[1:])
        ]

        res_tuples = []
        for pair in pairs: # linearise the resulting tuples
            pair_trainset = self.gen_train_from_pair(pair)
            if pair_trainset:
                for tup in pair_trainset:
                    res_tuples.append(tup)

        return res_tuples

    def df_calc_feat_matrix(self, pair_loc, norm=True, px_size=None, df=None):
        '''
        Takes an indexer (list) with pos-trap and tuple of two tps to be
        compared and returns the feature comparison matrix of their cells
        '''

        if df is None:
            df = self.rprops

        if px_size is None:
            px_size = self.px_size

        subdf = df.loc(axis=0)[pair_loc]

        prev_feats, new_feats = subdf.groupby('tp').apply(np.array)

        array_3d = self.calc_feat_ndarray(prev_feats, new_feats)

        return array_3d

    def explore_hyperparams(self, model_type = 'rf'):
        self.model_type = model_type
        truth, data = *zip(*self.train),
        if self.model_type == 'SVC':
            model = SVC(probability = True, shrinking=False,
                        verbose=True, random_state=1)
            param_grid = {
              # 'method': ['sigmoid', 'isotonic']
              # 'class_weight':['balanced', None],
              # 'C': [0.1, 1, 10, 100, 1000],
              'C': [0.1, 10, 100],
              'gamma': [1, 0.01, 0.0001],
              'kernel': ['rbf', 'sigmoid']
            }
        elif model_type == 'rf':
            model = RandomForestClassifier(n_estimators=15,
                                    criterion='gini',
                                    max_depth=3,
                                    class_weight='balanced')

            param_grid = {
                'n_estimators': [20, 25, 30],
                'max_features': ['auto', 'sqrt', 'log2'],
                'max_depth': [None, 3, 4, 5],
                'class_weight': [None, 'balanced', 'balanced_subsample']
            }
        else:
            raise("model_type not found")

        self.model = GridSearchCV(estimator=model, param_grid=param_grid, cv=5)
        self.model.fit(data, truth)
        # Add data on the features and their order
        self.model.best_estimator_.all_ofeats = self.all_ofeats
        self.model.best_estimator_.all_ifeats = [self.feats2use, self.trapfeats, self.extra_feats]
        print(self.model.best_score_, self.model.best_params_)

    def save_model(self, filename):
        date = datetime.date.today().strftime("%Y%m%d")
        nfeats = str(self.noutfeats)
        model_type = 'svc' if isinstance(self.model.best_estimator_, SVC) else 'rf'

        f = open(filename + '_'.join(('ct', model_type, date,
                                      nfeats)) + '.pkl', 'wb')
        pickle.dump(self.model.best_estimator_, f)

    def save_self(self, filename):
        f = open(filename, 'wb')
        pickle.dump(self, f)

    @property
    def benchmarker(self):
        '''
        Create a benchmarker instance to test the new model

        requires
        :self.model:
        :self.meta:

        returns
        :self._benchmarker:
        
        '''
        if not hasattr(self, '_benchmarker'):
            val_meta = self.meta.loc[self.meta['train_val'] == 'validation']
            self._benchmarker = CellBenchmarker(val_meta, self.model.best_estimator_,
                                                self.model.best_estimator_)
        return self._benchmarker




def get_ground_truth(cell_labels, buds):
    ncells = len(cell_labels)
    truth = np.zeros((ncells, ncells),dtype=bool)
    for i, bud in enumerate(buds):
        if bud != 0:
            truth[cell_labels.index(bud), i] = True

    return truth
