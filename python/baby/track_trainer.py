#!/usr/bin/env python
import numpy as np
import pandas as pd

from .tracker import Tracker
from .io import load_tiled_image

from scipy.ndimage import binary_fill_holes
from skimage.measure import regionprops_table
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

class TrackTrainer(Tracker):
    '''
    :meta: Metadata Dataframe
    :traps: Dataframe with cleaned trap locations and their continuous tps
    '''

    def __init__(self, meta, data=None, masks=None):
        super().__init__()
        self.indices = ['experimentID', 'position', 'trap', 'tp']
        self.cindices =  self.indices + ['cellLabels']
        self.meta = meta.set_index(self.indices)
        self.data = data
        self.traps = data.traps
        if masks is None:
            self.masks= [load_tiled_image(mask)[0] for
                         bf, mask  in self.data.training]
        self.process_traps()
        self.gen_train()

    def get_img_feats(self, img_array):
        props_df = pd.DataFrame([
            regionprops_table(img, properties=self.feats2use, cache=True)
            for img in img_array
        ]).applymap(lambda x: x[0])

        return props_df

    def gen_train(self):
        '''
        Generates the data for training using all the loaded images.
        '''

        traps = np.unique([ind[:3] for ind in self.traps.index], axis=0)
        traps = [(ind[0], *map(int, ind[1:])) for ind in traps] # str->int conversion
        traps = *map(
            tuple, traps),

        train = *map(self.gen_train_from_trap, traps),
        self.train = np.concatenate(train)

    def gen_train_from_pair(self, pair_loc):
        subdf = self.meta[['list_index', 'cellLabels']].loc(axis=0)[pair_loc]

        truemat = np.equal.outer(*subdf['cellLabels'].values).reshape(-1)
        propsmat = self.df_calc_feat_matrix(pair_loc).reshape(-1, self.nfeats)

        return [x for x in zip(truemat, propsmat)]

    def process_traps(self):
        '''
        Process all traps (run for finished experiments), combine results with location df and drop
        unused columns.

        Generates a region_proprieties DataFrame
        '''

        nindex = []
        props_list = []

        # Clean up duplicates and non-continuous timepoints
        self.meta = self.meta[~self.meta.index.duplicated(keep='first')]
        self.traps = self.traps.explode('cont')
        self.traps['tp'] = self.traps['cont']
        self.traps.set_index('tp', inplace=True, append=True)
        clean_indices = self.traps.index

        for ind, (index,
                  lbl) in zip(clean_indices,
                               self.meta.loc[clean_indices][['list_index', 'cellLabels']].values):
            try:
                assert (len(lbl)==self.masks[index].shape[2]) | (len(lbl)==0) #check nlabels and ncells agree
            except AssertionError:
                print('nlabels and img mismatch in row:')
                print(self.meta.iloc[index])

            trapfeats = [
                regionprops_table(self.masks[index][..., i].astype('int'),
                                  properties=self.feats2use)  #
                for i in range(len(lbl))
            ]
            for cell, feats in zip(lbl, trapfeats):
                nindex.append(ind + (cell, ))
                props_list.append(feats)

        out_dict = {key: [] for key in props_list[0].keys()}
        nindex = pd.MultiIndex.from_tuples(nindex, names=self.cindices)

        for cells_props in props_list:
            for key, val in cells_props.items():
                out_dict[key].append(val[0])

        self.rprops = pd.DataFrame(out_dict, index=nindex)
        self.rprop_keys = self.rprops.columns

    def gen_train_from_trap(self, trap_loc):
        subdf = self.meta[['list_index', 'cellLabels'
                             ]].loc(axis=0)[trap_loc].sort_values('tp')
        pairs = [
            trap_loc + tuple((pair, ))
            for pair in zip(subdf.index[:-1], subdf.index[1:])
        ]

        res_tuples = [
            tup for pair in pairs for tup in self.gen_train_from_pair(pair)
        ]

        return res_tuples

    def df_calc_feat_matrix(self, pair_loc, df=None):
        '''
        Takes an indexer (list) with pos-trap and tuple of two tps to be
        compared and returns the feature comparison matrix of their cells
        '''

        if df is None:
            df = self.rprops

        if pair_loc[1]==38:
            print(pair_loc)
            print(np.unique([i[2] for i in self.rprops.index]))

        subdf = df.loc(axis=0)[pair_loc]
        group_props = subdf.groupby('tp')
        group_sizes = group_props.size().to_list()

        self.out_feats = subdf.columns.to_list()
        self.nfeats = len(self.out_feats) + len(self.xtrafeats)
        # Array to pour the calculations and get cellxcell feature vectors
        n3darray = np.empty(*[group_sizes + [self.nfeats]])

        for i, feat in enumerate(self.out_feats):
            n3darray[..., i] = np.subtract.outer(
                *group_props[feat].apply(list).to_list())

        # Calculate extra features
        for i, feat in enumerate(self.xtrafeats, len(self.out_feats)):
            if feat == 'distance':
                n3darray[..., i] = np.sqrt(
                    n3darray[..., self.out_feats.index('centroid-0')]**2 +
                    n3darray[..., self.out_feats.index('centroid-1')]**2)

            # Add here any other calculation to use it as a feature
        return n3darray  # Removed globids

    def explore_hyperparams(self):

        truth, data = *zip(*self.train),
        rf = RandomForestClassifier(n_estimators=15,
                                    criterion='gini',
                                    max_depth=3,
                                    class_weight='balanced')

        param_grid = {
            'n_estimators': [4, 6, 9],
            'max_features': ['auto', 'sqrt', 'log2'],
            'max_depth': [2, 3],
            'class_weight': [None, 'balanced', 'balanced_subsample']
        }

        self.rf = GridSearchCV(estimator=rf, param_grid=param_grid, cv=5)
        self.rf.fit(data, truth)
        print(self.rf.best_score_, self.rf.best_params_)

    def train_model(self):
        truth, data = *zip(*self.train),
        self.single_rf = RandomForestClassifier(n_estimators=6,
                                                n_jobs=1,
                                                max_depth=2,
                                                class_weight='balanced')
        self.single_rf.fit(data, truth)

    def save_model(self, filename):
        f = open(filename, 'wb')
        pickle.dump(track_trainer.rf.best_estimator_)
