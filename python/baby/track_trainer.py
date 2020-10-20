#!/usr/bin/env python
import numpy as np
import pandas as pd
import pickle
from pathlib import Path
from itertools import repeat, chain
from warnings import warn
from tqdm import trange

from .io import load_tiled_image
from .tracker import CellTracker, MasterTracker
from .tracker_benchmark import TrackBenchmarker
from .utils import TrainValProperty
from .errors import BadProcess, BadParam

from scipy.ndimage import binary_fill_holes
from skimage.measure import regionprops_table
from skimage.transform import resize
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn import metrics

class TrackTrainer(CellTracker):
    '''
    :meta: Metadata Dataframe
    :traps: Dataframe with cleaned trap locations and their continuous tps
    '''

    def __init__(self, meta, data=None, masks=None,
                 val_masks = None, all_feats2use=None,
                 px_size=None):

        if all_feats2use is None:
            feats2use, extra_feats = (None, None)
        else:
            feats2use, extra_feats = all_feats2use

        if px_size is None:
            px_size = 0.263
        self.px_size = px_size
            
        super().__init__(feats2use = feats2use, extra_feats = extra_feats,
                         px_size=px_size)

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
        subdf = self.meta[['list_index', 'cellLabels']].loc(axis=0)[pair_loc]

        if not subdf['cellLabels'].all():
            return None

        truemat = np.equal.outer(*subdf['cellLabels'].values).reshape(-1)
        propsmat = self.df_calc_feat_matrix(pair_loc).reshape(
            -1, self.nfeats) 

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

        res_tuples = []
        for pair in pairs: # linearise the resulting tuples
            pair_trainset = self.gen_train_from_pair(pair)
            if pair_trainset:
                for tup in pair_trainset:
                    res_tuples.append(tup)

        # res_tuples = [
        #     tup for pair in pairs for tup in self.gen_train_from_pair(pair)
        # ]

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

        if pair_loc[2]== 11 and pair_loc[3]==(1,2): #debugging
            print(subdf)

        group_props = subdf.groupby('tp')
        group_sizes = group_props.size().to_list()

        self.out_feats = subdf.columns.to_list()
        self.nfeats = len(self.out_feats) + len(self.extra_feats)
        # Array to pour the calculations and get cellxcell feature vectors
        array_3d = np.empty(*[group_sizes + [self.nfeats]])

        for i, feat in enumerate(self.out_feats):
            array_3d[..., i] = np.subtract.outer(
                *group_props[feat].apply(list).to_list())

        if norm: # normalise features
            ncells1, ncells2, noutfeats = array_3d.shape
            for i in range(ncells1):
                for j in range(ncells2):
                    array_3d[i,j,:noutfeats] = \
                    self.norm_feats(array_3d[i,j,:noutfeats], px_size=px_size)
            

        # Calculate extra features
        for i, feat in enumerate(self.extra_feats, len(self.out_feats)):
            if feat == 'distance':
                array_3d[..., i] = np.sqrt(
                    array_3d[..., self.out_feats.index('centroid-0')]**2 +
                    array_3d[..., self.out_feats.index('centroid-1')]**2)

            # Add here any other calculation to use it as a feature

        return array_3d  

    def explore_hyperparams(self):

        truth, data = *zip(*self.train),
        rf = RandomForestClassifier(n_estimators=15,
                                    criterion='gini',
                                    max_depth=3,
                                    class_weight='balanced')

        param_grid = {
            'n_estimators': [5, 15, 30, 60, 100],
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
        '''
        Create a benchmarker instance to test the newly calculated model

        requires
        :self.rf:
        :self.meta:

        returns
        :self._benchmarker:
        
        '''
        if not hasattr(self, '_benchmarker'):
            val_meta = self.meta.loc[self.meta['train_val'] == 'validation']
            self._benchmarker = TrackBenchmarker(val_meta, self.rf.best_estimator_)
        return self._benchmarker


class BudTrainer(MasterTracker):
    '''
    :props_file: File where generated property table will be saved
    :kwargs: Additional arguments passed onto the parent Tracker; `px_size` is
        especially useful.
    '''

    def __init__(self, props_file=None, **kwargs):
        super().__init__(**kwargs)
        # NB: we inherit self.ba_feat_names from Tracker class
        self.props_file = props_file

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
        required_cols = self.ba_feat_names + ('is_mb_pair', 'validation')
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
        if isinstance(data, TrainValProperty):
            data = chain(zip(repeat(False), data.train),
                         zip(repeat(True), data.val))
        else:
            data = zip(repeat(None), data)

        p_list = []
        for is_val, seg_example in data:
            if len(seg_example.target) < 2:
                continue
            mb_stats = self.calc_mother_bud_stats(seg_example.pred[i_budneck],
                    seg_example.pred[i_bud], seg_example.target)
            p = pd.DataFrame(mb_stats, columns=self.ba_feat_names)
            p['validation'] = is_val
            cell_labels = seg_example.info.get('cellLabels', []) or []
            if type(cell_labels) is int:
                cell_labels = [cell_labels]
            buds = seg_example.info.get('buds', []) or []
            if type(buds) is int:
                buds = [buds]
            p['is_mb_pair'] = get_ground_truth(cell_labels, buds).flatten()
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

    def explore_hyperparams(self):
        data = self.props.loc[~self.props['validation'], self.ba_feat_names]
        truth = self.props.loc[~self.props['validation'], 'is_mb_pair']

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
        scoring = ['accuracy', 'balanced_accuracy', 'precision',
                   'recall', 'f1'] 

        def get_balanced_best_index(cv_results_):
            '''Find a model balancing F1 score and speed'''
            df = pd.DataFrame(cv_results_)
            best_score = df.iloc[df.mean_test_f1.idxmax(), :]
            thresh = best_score.mean_test_f1 - 0.1 * best_score.std_test_f1
            return df.loc[df.mean_test_f1 > thresh, 'mean_score_time'].idxmin()

        self._rf = GridSearchCV(estimator=rf, param_grid=param_grid, cv=5,
                scoring=scoring, refit='f1')
        self._rf.fit(data, truth)

        df = pd.DataFrame(self._rf.cv_results_)
        disp_cols = [c for c in df.columns if c.startswith('mean_')
                     or c.startswith('param_')]
        print(df.loc[self._rf.best_index_, disp_cols])

        print('\nValidation performance:')
        best_rf = self._rf.best_estimator_
        valdata = self.props.loc[self.props['validation'], self.ba_feat_names]
        valtruth = self.props.loc[self.props['validation'], 'is_mb_pair']
        preds = best_rf.predict(valdata)
        print('Accuracy:', metrics.accuracy_score(valtruth, preds))
        print('Precision:', metrics.precision_score(valtruth, preds))
        print('Recall:', metrics.recall_score(valtruth, preds))
        print('F1 score:', metrics.f1_score(valtruth, preds))

        return best_rf

    def save_model(self, filename):
        f = open(filename, 'wb')
        pickle.dump(self._rf.best_estimator_, f)

def get_distance(point1, point2):
    return(np.sqrt(np.sum(np.array([point1[i]-point2[i] for i in [0,1]])**2)))

def get_ground_truth(cell_labels, buds):
    ncells = len(cell_labels)
    truth = np.zeros((ncells, ncells),dtype=bool)
    for i, bud in enumerate(buds):
        if bud != 0:
            truth[cell_labels.index(bud), i] = True

    return truth

