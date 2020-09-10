#!/usr/bin/env python
import numpy as np
import pandas as pd
import pickle

from .io import load_tiled_image
from .tracker import Tracker
from .tracker_benchmark import TrackBenchmarker

from scipy.ndimage import binary_fill_holes
from skimage.measure import regionprops_table
from skimage.transform import resize
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

class TrackTrainer(Tracker):
    '''
    :meta: Metadata Dataframe
    :traps: Dataframe with cleaned trap locations and their continuous tps
    '''

    def __init__(self, meta, data=None, masks=None, val_masks = None):
        super().__init__()
        self.indices = ['experimentID', 'position', 'trap', 'tp']
        self.cindices =  self.indices + ['cellLabels']
        self.data = data
        self.traps = data.traps
        # if val_masks is None:
        #     self.val_masks = [load_tiled_image(mask)[0] for
        #                       bf, mask  in self.data.validation]
        self.meta = data._metadata_tp
        self.process_metadata()
        if masks is None:
            self.masks= [load_tiled_image(fname)[0] for
                         fname  in self.meta['filename']]
        self.process_traps()
        self.gen_train()

    def verify_mask_df_integrity(masks, df):
        nlayers=[mask.shape[2] for mask in masks]
        ncells = [len(x) for x in df['cellLabels'].values]

        for x,(i,j) in enumerate(zip(nlayers, ncells)):
            if i!=j:
                print(x)


    def process_metadata(self):
        '''
        Process all traps (run for finished experiments), combine results with location df and drop
        unused columns.
        '''
        
        self.meta = self.meta[~self.meta.index.duplicated(keep='first')]
        self.traps = self.traps.explode('cont')
        self.traps['tp'] = self.traps['cont']
        self.traps.set_index('tp', inplace=True, append=True)
        self.clean_indices = self.traps.index

        self.meta = self.meta.loc(axis=0)[self.clean_indices]
        self.meta['ncells'] = [len(i) for i in self.meta['cellLabels'].values]

    def get_truth_matrix_from_tups(self, tup1, tup2):
        for i in range(3):
            if tup1[i]!=tup2[i]:
                raise IndexError("Indices are not from the same trap")
            
        if np.abs(tup[4] - tup2[3])>5:
            raise Warning("Indices are more than five timepoints apart")

        clabs1 = self.meta.loc(axis=0)[tup1, 'cellLabels'].tolist()
        clabs2 = self.meta.loc(axis=0)[tup2, 'cellLabels'].tolist()

        truth_mat = gen_boolmat_from_clabs(clabs1, clabs2)

        return truth_mat

    def gen_train(self):
        '''
        Generates the data for training using all the loaded images.
        '''

        traps = np.unique([ind[:self.indices.index('trap')+1] for ind in self.traps.index], axis=0)
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
        Generates a region_proprieties DataFrame
        '''

        nindex = []
        props_list = []
        i=0
        for ind, (index,
                  lbl) in zip(self.meta.index, enumerate(
                               self.meta['cellLabels'].values)):
            try:
                assert (len(lbl)==self.masks[index].shape[2]) | (len(lbl)==0) #check nlabels and ncells agree
            except AssertionError:
                print('nlabels and img mismatch in row: ')

            trapfeats = [
                regionprops_table(resize(self.masks[index][..., i],
                                         (100, 100)).astype('bool').astype('int'),
                                  properties=self.feats2use)  #
                for i in range(len(lbl))
            ]

            for cell, feats in zip(lbl, trapfeats):
                nindex.append(ind + (cell, ))
                props_list.append(feats)
            i+=1

        out_dict = {key: [] for key in props_list[0].keys()}
        nindex = pd.MultiIndex.from_tuples(nindex, names=self.cindices)

        for cells_props in props_list:
            for key, val in cells_props.items():
                out_dict[key].append(val[0])

        self.rprops = pd.DataFrame(out_dict, index=nindex)
        self.rprop_keys = self.rprops.columns

    def gen_train_from_trap(self, trap_loc):
        subdf = self.meta[['list_index', 'cellLabels'
                             ]].loc(axis=0)[trap_loc]
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

        return n3darray  

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
        pickle.dump(self.rf.best_estimator_, f)

    @property
    def benchmarker(self):
        if not hasattr(self, '_benchmarker'):
            val_meta = self.meta.loc[self.meta['train_val'] == 'validation']
            self._benchmarker = TrackBenchmarker(val_meta, self.rf)
        return self._benchmarker


class BudTrainer:
    '''
    :masks: List of ndarrays with the cell outlines
    :data:  Data obtained from the TrainPairVal class
    :rprops: List of list of regionprops for each cell
    '''

    def __init__(self, meta, data=None, masks=None, props=None ):
        super().__init__()
        self.indices = ['experimentID', 'position', 'trap', 'tp']
        self.data = data
        self.traps = data.traps
        self.feats2use = ['centroid', 'area',
                          'major_axis_length', 'minor_axis_length']
        self.outfeats = list(
            regionprops_table(np.diag((1, 0)),
                              properties=self.feats2use).keys())
        self.rf_input = ['centroid_distance', 'area_ratio', 'shared_area',
                         'overlap_rel_loc', 'overlap_major_ax', 'overlap_minor_ax']
        if masks is None:
            self.masks= [load_tiled_image(mask)[0] for
                         bf, mask  in self.data.training]
        if props is None:
            self.props = []
            for mask, lbls in zip(self.masks, self.meta['cellLabels']):
                props_mat = np.zeros((len(lbls), len(self.outfeats)), dtype=float)
                if lbls:
                    for i in range(mask.shape[2]):
                        props_mat[i] = [prop[0] for prop in regionprops_table(
                            mask[..., i].astype('int'),
                            properties=self.feats2use).values()]
                self.props.append(props_mat)

    @property
    def train(self):
        if not hasattr(self, '_train'):
            train = []
            all_lbls, all_buds = zip(*self.meta[['cellLabels', 'buds']].values)
            for mask, props, lbls, buds in zip(
                    self.masks, self.props, all_lbls, all_buds):
                if lbls and buds:
                    feat_matrix = self.calc_feat_matrix(mask, props).reshape(-1,
                                                                             len(self.rf_input))
                    truth = get_ground_truth(lbls, buds).reshape(-1)

                    for train_item in zip(truth, feat_matrix):
                        train.append(train_item)

            self._train = train
        return self._train

    def calc_feat_matrix(self, mask, props):
        ncells = mask.shape[2]
        feat_matrix = np.zeros((ncells, ncells, len(self.rf_input)),dtype=float)

        for i in range(ncells):
            for j in range(ncells):
                feat_matrix[i,j] = self.calc_comparison(mask[i], mask[j],
                                                        props[i], props[j])

        return feat_matrix


    def calc_comparison(self, outline1, outline2, props1=None, props2=None):
        if props1 is None:
            props1 = regionprops_table(outline1, coordinates='rc')[0]
        if props2 is None:
            props2 = regionprops_table(outline2, coordinates='rc')[0]

        centroid_1 = self.get_centroid(props1)
        centroid_2 = self.get_centroid(props2)
        centroid_dist = get_distance(centroid_1, centroid_2)


        area1 = self.get_area(props1)
        area2 =  self.get_area(props2)
        area_ratio = area1/area2

        # Calculate values of the small cell
        small_props = props1 if area1<area2 else props2
        small_centroid = self.get_centroid(small_props)
        small_minor_ax = self.get_minor_ax(small_props)
        small_major_ax = self.get_major_ax(small_props)

        # Calculate features for the overlap
        overlap = outline1 & outline2
        if np.sum(overlap) > 0:
            overlap_props = [feat[0] for feat in regionprops_table(overlap, properties =
                                              ['centroid', 'area', 'major_axis_length',
                                               'minor_axis_length']).values()]

            overlap_centroid = self.get_centroid(overlap_props)
            overlap_area = self.get_area(overlap_props)
            overlap_major_ax = self.get_major_ax(overlap_props)
            overlap_minor_ax = self.get_minor_ax(overlap_props)

            overlap_rel_loc = get_distance(overlap_centroid, small_centroid) / centroid_dist
        else:
            overlap_area=0
            overlap_rel_loc=400
            overlap_major_ax=0
            overlap_minor_ax=0

        output = (centroid_dist, area_ratio, overlap_area,
               overlap_rel_loc, overlap_major_ax, overlap_minor_ax)
        return output

        #TODO implement rectangle calculation

    def get_centroid(self, props):
        return (props[self.outfeats.index('centroid-0')],
                  props[self.outfeats.index('centroid-1')])
    def get_area(self, rprops):
        return rprops[self.outfeats.index('area')]

    def get_major_ax(self, rprops):
        return rprops[self.outfeats.index('major_axis_length')]

    def get_minor_ax(self, rprops):
        return rprops[self.outfeats.index('minor_axis_length')]

    def explore_hyperparams(self):

        truth, data = *zip(*self.train),
        rf = RandomForestClassifier(n_estimators=15,
                                    criterion='gini',
                                    max_depth=3,
                                    class_weight='balanced')

        param_grid = {
            'n_estimators': [6, 50, 100],
            'max_features': ['auto', 'sqrt', 'log2'],
            'max_depth': [2, 3, 4],
            'class_weight': [None, 'balanced', 'balanced_subsample']
        }

        self._rf = GridSearchCV(estimator=rf, param_grid=param_grid, cv=5)
        self._rf.fit(data, truth)
        print(self._rf.best_score_, self._rf.best_params_)
        return self._rf.best_estimator_

    def save_model(self, filename):
        f = open(filename, 'wb')
        pickle.dump(track_trainer._rf.best_estimator_, f)

def get_distance(point1, point2):
    return(np.sqrt(np.sum(np.array([point1[i]-point2[i] for i in [0,1]])**2)))

def get_ground_truth(cell_labels, buds):
    ncells = len(cell_labels)
    truth = np.zeros((ncells, ncells),dtype=bool)
    for i, bud in enumerate(buds):
        if bud != 0:
            truth[cell_labels.index(bud), i] = True

    return truth

def gen_boolmat_from_clabs(clabs1, clabs2):
    boolmat = np.zeros((len(clabs1), len(clabs2))).astype(bool)
    for i, lab1 in enumerate(clabs1):
        for j, lab2 in enumerate(clabs2):
            if lab1==lab2:
                boolmat[i, j] = True

    return boolmat
