import pickle
from itertools import chain
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
from baby.brain import BabyBrain
from baby.io import load_tiled_image
from baby.performance import calc_IoUs, best_IoU
from baby.preprocessing import (
    robust_norm, SegmentationFlattening, flattener_norm_func, seg_norm
)
from baby.segmentation import morph_seg_grouped
from baby.utils import batch_iterator, split_batch_pred
from scipy.ndimage import (
    binary_fill_holes
)
from skimage.draw import polygon
from skimage.measure import regionprops, regionprops_table
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

if tf.__version__.startswith('1'):
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    tf.keras.backend.set_session(tf.Session(config=config))


class BudTrainer:
    '''
    :masks: List of ndarrays with the cell outlines
    :data:  Data obtained from the TrainPairVal class
    :rprops: List of list of regionprops for each cell
    '''

    def __init__(self, meta, data=None, masks=None, props=None, mb_stats=None, brain=None,
                 flattener=None, log_dir=None, train_pred=None, val_pred=None, train_data=None,
                 val_data=None, train_agg=None, val_agg=None, train=None, val=None):
        super().__init__()
        self.indices = ['experimentID', 'position', 'trap', 'tp']
        self.meta = meta.set_index(self.indices, drop=False)
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
        # if props is None:
        #     self.props = []
        #     for mask, lbls in zip(self.masks, self.meta['cellLabels']):
        #         props_mat = np.zeros((len(lbls), len(self.outfeats)), dtype=float)
        #         if lbls:
        #             for i in range(mask.shape[2]):
        #                 props_mat[i] = [prop[0] for prop in regionprops_table(
        #                     mask[..., i].astype('int'),
        #                     properties=self.feats2use).values()]
        #         self.props.append(props_mat)

        if train is None:
            if log_dir is None:
                log_dir = Path('training-logs/190905_babyseg')

            if flattener is None:
                flattener = SegmentationFlattening(log_dir / 'flattener_v2.json')
                self.flattener_norm = flattener_norm_func(flattener)

            if brain is None:
                model_dir = log_dir / 'msd_d32r2_d16_grps_tv2'
                brain = BabyBrain(morph_model_file=model_dir / 'weights.hdf5', flattener_file=log_dir / 'flattener_v2.json')

            self.flattener = flattener
            self.brain = brain

            if not (hasattr(self, '_train')) and (train_data is None):
                train_data = self.load_data_from_img_seg('training')
                self._train_data = train_data
                val_data = self.load_data_from_img_seg('validation')
            self._val_data = val_data


            if not hasattr(self, '_train_pred') and (train_pred is None):
                train_pred = self.gen_pred(train_data)
                self._train_pred = train_pred
                val_pred = self.gen_pred(val_data)
            self._val_pred = val_pred

            if not hasattr(self, '_train_agg') and train_agg is None:
                train_agg = self.aggregate_rf_inputs(train_data, train_pred)
                val_agg = self.aggregate_rf_inputs(val_data, val_pred)
            self.train_agg = train_agg
            self.val_agg = val_agg

            if not (hasattr(self, '_train') and (hasattr(self, '_val'))):
                train = *zip(train_agg.iloc[:,1].astype('bool'), train_agg.iloc[:,1:].astype(float).values),
                val = *zip(val_agg.iloc[:,1].astype('bool'), val_agg.iloc[:,1:].astype(float).values),

        self._train = train
        self._val = val
            

    def load_data_from_img_seg(self, train_val):
            bf_fname = self.data.validation if train_val=='validation' else self.data.training
            bf_data = [self.load_data(img, seg) for img, seg in bf_fname]

            return bf_data

    def gen_pred(self, data):
        img2stack = [img for img, _, _, _ in data]
        bit = batch_iterator(np.stack(img2stack))
        return (list(chain(*[split_batch_pred(self.brain.morph_predict(b)) for b in
                                bit])))

    def aggregate_rf_inputs(self, data, pred):
        return pd.concat([self.calculate_rf_input(tsegs, info, pred) \
                               for (_, tsegs, _, info), pred in zip(data, pred)],
                              ignore_index=True)

    def calculate_rf_input(self, tsegs, info, pred):
        if tsegs.any():
            tsegs = [tsegs[:,:,s]>0 for s in range(tsegs.shape[2])]

        psegs =  morph_seg_grouped(
            pred, self.flattener, interior_threshold=0.6, nclosing=2, nopening=0,
            pedge_thresh=0.001, fit_radial=True, use_group_thresh=False
        )

        p_bud_mat, size_ratio_mat, bud_neck_mat, adjacent_mat = get_mother_bud_stats(
            pred, self.flattener, psegs)
        gt_mat = get_ground_truth_baby_mat(psegs, tsegs, info)
        train = {
            'ground_truth': gt_mat.flatten(), 'p_bud': p_bud_mat.flatten(),
            'size_ratio': size_ratio_mat.flatten(), 'p_bud_neck': bud_neck_mat.flatten(),
            'adjacency': adjacent_mat.flatten()
        }

        return pd.DataFrame(train)

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

    @property
    def val(self):
        if not hasattr(self, '_val'):
            val = None
            self._val = val
        return self._val

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

    def load_data(self, img_file, seg_file):
        img, bfinfo = load_tiled_image(img_file)
        img = robust_norm(img, bfinfo)
        seg, info = load_tiled_image(seg_file)
        lbl = self.flattener_norm(seg, info)
        seg, info = seg_norm(seg, info)
        return img, seg, lbl, info

    def get_centroid(self, props):
        return (props[self.outfeats.index('centroid-0')],
                  props[self.outfeats.index('centroid-1')])
    def get_area(self, rprops):
        return rprops[self.outfeats.index('area')]

    def get_major_ax(self, rprops):
        return rprops[self.outfeats.index('major_axis_length')]

    def get_minor_ax(self, rprops):
        return rprops[self.outfeats.index('minor_axis_length')]

    def fit_simple_model(self):
        truth, data = *zip(*self.train),
        self._model = RandomForestClassifier(n_estimators=50, max_depth=3)
        self._model.fit(data, truth)
        return self._model

    def explore_hyperparams(self):

        truth, data = *zip(*self.train),
        rf = RandomForestClassifier(n_estimators=15,
                                    criterion='gini',
                                    max_depth=3,
                                    class_weight='balanced')

        param_grid = {
            'n_estimators': [6, 50],
            'max_features': ['auto', 'log2'],
            'max_depth': [2, 3],
            'class_weight': [None, 'balanced']
        }

        self._rf = GridSearchCV(estimator=rf, param_grid=param_grid, cv=5)
        self._rf.fit(data, truth)
        print(self._rf.best_score_, self._rf.best_params_)
        self._model = self._rf.best_estimator_
        return self._model

    def validate_hyperparams(self):
        pred = self._rf.best_estimator_.predict([i[1] for i in self.val])
        true = [i[0] for i in self.val]
        print('Fraction correct:', np.sum(pred==true)/len(true))
        print('True positives:', np.sum(pred & true))
        print('False positives:', np.sum(pred & ~true))
        print('False negatives:', np.sum(~pred & true))

    def save_model(self, filename):
        f = open(filename, 'wb')
        pickle.dump(track_trainer._rf.best_estimator_)


def get_mother_bud_stats(pred, flattener, segs, masks=None, rprops=None):
    tnames = flattener.names()
    p_budneck = pred[tnames.index('bud_neck')]
    p_bud = pred[tnames.index('sml_fill')]

    if masks is None:
        masks = [binary_fill_holes(s) for s in segs]
    if rprops is None:
        rprops = [regionprops(m.astype('int'), coordinates='rc')[0] for m in masks]

    ncells = len(rprops)

    p_bud_mat = np.zeros((ncells, ncells))
    p_sizratio_mat = np.zeros((ncells, ncells))
    p_bud_neck_mat = np.zeros((ncells, ncells))
    p_adjacent_mat = np.zeros((ncells, ncells))

    for m in range(ncells):
        for d in range(ncells):
            if m == d:
                continue

            p_bud_mat[m, d] = np.mean(p_bud[masks[d]])
            p_sizratio_mat[m, d] = rprops[m].area / rprops[d].area

            # Draw connecting rectangle
            m_centre = np.array(rprops[m].centroid).T
            d_centre = np.array(rprops[d].centroid).T
            r_width = np.max((2, rprops[d].minor_axis_length * 0.25))
            r_hvec = d_centre - m_centre
            r_wvec = np.matmul(np.array([[0, -1], [1, 0]]), r_hvec)
            r_wvec = r_width * r_wvec / np.linalg.norm(r_wvec)
            r_points = np.zeros((2,4))
            r_points[:,0] = m_centre - 0.5 * r_wvec
            r_points[:,1] = r_points[:,0] + r_hvec
            r_points[:,2] = r_points[:,1] + r_wvec
            r_points[:,3] = r_points[:,2] - r_hvec
            rr, cc = polygon(r_points[0,:], r_points[1,:], p_budneck.shape)
            r_im = np.zeros(p_budneck.shape, dtype='bool')
            r_im[rr, cc] = True

            # Calculate the mean of bud neck probabilities greater than some threshold
            pbn = p_budneck[r_im].flatten()
            pbn = pbn[pbn > 0.2]
            p_bud_neck_mat[m, d] = np.mean(pbn) if len(pbn) > 0 else 0

            # Adjacency is the proportion of the joining rectangle that overlaps the mother daughter union
            p_adjacent_mat[m, d] = np.sum((masks[m] | masks[d]) & r_im) / np.sum(r_im)

    return p_bud_mat, p_sizratio_mat, p_bud_neck_mat, p_adjacent_mat

def get_ground_truth_baby_mat(psegs, tsegs, info):
    _, assignments = best_IoU(calc_IoUs(psegs, tsegs))
    c_label = info.get('cellLabels', [])
    if c_label is None:
        c_label = []
    elif type(c_label) is int:
        c_label = [c_label]

    buds = info.get('buds', [])
    if buds is None:
        buds = []
    elif type(buds) is int:
        buds = [buds]

    c_label = np.array(c_label)
    buds = np.array(buds)

    gt_mat = np.zeros((len(psegs), len(psegs)))
    for p in range(len(psegs)):
        gind = assignments[p]
        if gind < 0:
            continue
        try:
            bl = buds[gind]
        except IndexError as err:
            print(gind)
            print(buds)
            raise err

        if bl not in c_label:
            continue
        gt_mat[p, np.flatnonzero(assignments==np.flatnonzero(bl==c_label))] = 1

    return gt_mat


def get_distance(point1, point2):
    return(np.sqrt(np.sum(np.array([point1[i]-point2[i] for i in [0,1]])**2)))

def get_ground_truth(cell_labels, buds):
    ncells = len(cell_labels)
    truth = np.zeros((ncells, ncells),dtype=bool)
    for i, bud in enumerate(buds):
        if bud != 0:
            truth[cell_labels.index(bud), i] = True

    return truth
