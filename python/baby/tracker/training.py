#!/usr/bin/env python

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
from .core import CellTracker, BudTracker

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
    precision_score, recall_score, f1_score, plot_precision_recall_curve
)

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
        if self.model_type is 'SVC':
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
        elif model_type is 'rf':
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



class BudTrainer(BudTracker):
    '''
    :props_file: File where generated property table will be saved
    :kwargs: Additional arguments passed onto the parent Tracker; `px_size` is
        especially useful.
    '''

    def __init__(self, props_file=None, **kwargs):
        super().__init__(**kwargs)
        # NB: we inherit self.feats2use from CellTracker class
        self.props_file = props_file
        self.rf_feats = ["p_bud_mat", "size_ratio_mat", "p_budneck_mat",
                "budneck_ratio_mat", "adjacency_mat"]

    @property
    def props_file(self):
        return getattr(self, '_props_file')

    @props_file.setter
    def props_file(self, filename):
        if filename is not None:
            self._props_file = Path(filename)

    @property
    def props(self):
        if getattr(self, '_props', None) is None:
            if self.props_file and self.props_file.is_file():
                self.props = pd.read_csv(self.props_file)
            else:
                raise BadProcess(
                        'The property table has not yet been generated')
        return self._props

    @props.setter
    def props(self, props):
        props = pd.DataFrame(props)
        required_cols = self.rf_feats + ['is_mb_pair', 'validation']
        if not all(c in props for c in required_cols):
            raise BadParam(
                '"props" does not have all required columns: {}'.format(
                    ', '.join(required_cols)))
        self._props = props
        if self.props_file:
            props.to_csv(self.props_file)

    def generate_property_table(self, data, flattener, val_data=None):
        '''Generates properties table that gets used for training

        :data: List or generator of `baby.training.SegExample` tuples
        :flattener: Instance of a `baby.preprocessing.SegmentationFlattening`
            object describing the targets of the CNN in data
        '''
        tnames = flattener.names()
        i_budneck = tnames.index('bud_neck')
        bud_target = 'sml_fill' if 'sml_fill' in tnames else 'sml_inte'
        i_bud = tnames.index(bud_target)

        if val_data is not None:
            data = TrainValProperty(data, val_data)
        if isinstance(data, (TrainValProperty, TrainValTestProperty)):
            data = chain(zip(repeat(False), data.train),
                         zip(repeat(True), data.val))
        else:
            data = zip(repeat(None), data)

        p_list = []
        for is_val, seg_example in data:
            if len(seg_example.target) < 2:
                # Skip if no pairs are present
                continue
            mb_stats = self.calc_mother_bud_stats(seg_example.pred[i_budneck],
                    seg_example.pred[i_bud], seg_example.target)
            p = pd.DataFrame(mb_stats, columns=self.rf_feats)
            p['validation'] = is_val

            # "cellLabels" specifies the label for each mask
            cell_labels = seg_example.info.get('cellLabels', []) or []
            if type(cell_labels) is int:
                cell_labels = [cell_labels]
            # "buds" specifies the label of the bud for each mask
            buds = seg_example.info.get('buds', []) or []
            if type(buds) is int:
                buds = [buds]

            # Build a ground truth matrix identifying mother-bud pairs
            ncells = len(seg_example.target)
            is_mb_pair = np.zeros((ncells, ncells), dtype=bool)
            mb_inds = [
                (i, cell_labels.index(b))
                for i, b in enumerate(buds)
                if b > 0 and b in cell_labels
            ]
            if len(mb_inds) > 0:
                mother_inds, bud_inds = zip(*mb_inds)
                is_mb_pair[mother_inds, bud_inds] = True
            p['is_mb_pair'] = is_mb_pair.flatten()

            # Ignore any rows containing NaNs
            nanrows = np.isnan(mb_stats).any(axis=1)
            if (p['is_mb_pair'] & nanrows).any():
                id_keys = ('experimentID', 'position', 'trap', 'tp')
                info = seg_example.info
                img_id = ' / '.join(
                        [k + ': ' + str(info[k]) for k in id_keys if k in info])
                warn('Mother-bud pairs omitted due to feature NaNs')
                print('Mother-bud pair omitted in "{}"'.format(img_id))
            p = p.loc[~nanrows, :]
            p_list.append(p)

        props = pd.concat(p_list, ignore_index=True)
        # TODO: should search for any None values in validation column and
        # assign a train-validation split to those rows

        self.props = props # also saves

    def explore_hyperparams(self, hyper_param_target='precision'):
        # Train bud assignment model on validation data, since this more
        # closely represents real-world performance of the CNN:
        data = self.props.loc[self.props['validation'], self.rf_feats]
        truth = self.props.loc[self.props['validation'], 'is_mb_pair']

        rf = RandomForestClassifier(n_estimators=15,
                                    criterion='gini',
                                    max_depth=3,
                                    class_weight='balanced')

        param_grid = {
            'n_estimators': [6, 15, 50, 100],
            'max_features': ['auto', 'sqrt', 'log2'],
            'max_depth': [2, 3, 4],
            'class_weight': [None, 'balanced', 'balanced_subsample']
        }

        def get_balanced_best_index(cv_results_):
            '''Find a model balancing F1 score and speed'''
            df = pd.DataFrame(cv_results_)
            best_score = df.iloc[df.mean_test_f1.idxmax(), :]
            thresh = best_score.mean_test_f1 - 0.1 * best_score.std_test_f1
            return df.loc[df.mean_test_f1 > thresh, 'mean_score_time'].idxmin()

        self._rf = GridSearchCV(estimator=rf, param_grid=param_grid, cv=5,
                scoring=SCORING_METRICS, refit=hyper_param_target)
        self._rf.fit(data, truth)

        df = pd.DataFrame(self._rf.cv_results_)
        disp_cols = [c for c in df.columns if c.startswith('mean_')
                     or c.startswith('param_')]
        print(df.loc[self._rf.best_index_, disp_cols])

    def performance(self):
        if not isinstance(getattr(self, '_rf', None), GridSearchCV):
            raise BadProcess('"explore_hyperparams" has not been run')

        best_rf = self._rf.best_estimator_
        isval = self.props['validation']
        data = self.props.loc[~isval, self.rf_feats]
        truth = self.props.loc[~isval, 'is_mb_pair']
        valdata = self.props.loc[isval, self.rf_feats]
        valtruth = self.props.loc[isval, 'is_mb_pair']
        metrics = tuple(SCORING_METRICS.values())
        return TrainValProperty(
                Score(*(m(best_rf, data, truth) for m in metrics)),
                Score(*(m(best_rf, valdata, valtruth) for m in metrics)))

    def plot_PR(self):
        best_rf = self._rf.best_estimator_
        isval = self.props['validation']
        valdata = self.props.loc[isval, self.rf_feats]
        valtruth = self.props.loc[isval, 'is_mb_pair']
        plot_precision_recall_curve(best_rf, valdata, valtruth)

    def save_model(self, filename):
        f = open(filename, 'wb')
        pickle.dump(self._rf.best_estimator_, f)


SCORING_METRICS = {
    'accuracy': make_scorer(accuracy_score),
    'balanced_accuracy': make_scorer(balanced_accuracy_score),
    'precision': make_scorer(precision_score),
    'recall': make_scorer(recall_score),
    'f1': make_scorer(f1_score),
    'f0_5': make_scorer(fbeta_score, beta=0.5),
    'f2': make_scorer(fbeta_score, beta=2)
}


class Score(NamedTuple):
    accuracy: float
    balanced_accuracy: float
    precision: float
    recall: float
    F1: float
    F0_5: float
    F2: float

    def __str__(self):
        return 'Score({})'.format(', '.join([
            '{}={:.3f}'.format(k, v) for k, v in self._asdict().items()
            ]))



def get_ground_truth(cell_labels, buds):
    ncells = len(cell_labels)
    truth = np.zeros((ncells, ncells),dtype=bool)
    for i, bud in enumerate(buds):
        if bud != 0:
            truth[cell_labels.index(bud), i] = True

    return truth
