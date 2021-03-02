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
import numpy as np
import pickle
import pandas as pd
from pathlib import Path, PosixPath

from baby.io import load_tiled_image
from baby.tracker.core import CellTracker
from baby.tracker.utils import lol_to_adj, compare_pred_truth_lols

from scipy.ndimage import binary_fill_holes
from skimage.measure import regionprops_table

class CellBenchmarker: #TODO Simplify this by inheritance
    '''
    Takes a metadata dataframe and a model and estimates the prediction in a trap-wise manner.

    This class can also produce confusion matrices for a given Tracker and validation dataset.
     '''
    def __init__(self, meta, model, bak_model, nstepsback=None):
        self.indices = ['experimentID', 'position', 'trap', 'tp']
        self.cindices =  self.indices + ['cellLabels']
        self.meta = meta.copy()
        self.meta['cont_list_index'] = *range(len(self.meta)),

        self.tracker = CellTracker(model = model, bak_model= bak_model)
        if nstepsback is None:
            self.nstepsback = self.tracker.nstepsback
        self.traps_loc

    @property
    def traps_loc(self):
        '''
        Generates a list of trap locations using the metadata.
        '''
        if not hasattr(self, '_traps_loc'):

            traps = np.unique([ind[:self.indices.index('trap')+1]
                               for ind in self.meta.index], axis=0)
            # str->int conversion
            traps = [(ind[0], *map(int, ind[1:])) for ind in traps] 
            self._traps_loc = *map(
                tuple, traps),

        return self._traps_loc

    @property
    def masks(self):
        if not hasattr(self, '_masks'):
            self._masks= [load_tiled_image(fname)[0] for
                              fname in self.meta['filename']]
            for i, mask in enumerate(self._masks):
                for j in range(mask.shape[2]):
                    self._masks[i][..., j] = \
                    binary_fill_holes(self._masks[i][..., j])

            self._masks = [np.moveaxis(mask, 2, 0) for mask in self._masks]

        return self._masks

    def predict_lbls_from_tpimgs(self, tp_img_tuple):
        max_lbl = 0
        prev_feats = []
        cell_lbls = []
        for tp, masks in tp_img_tuple:
            lastn_lbls = cell_lbls[-self.nstepsback:]
            lastn_feats = prev_feats[-self.nstepsback:]
            new_lbl, feats, max_lbl = self.tracker.get_new_lbls(
                masks, lastn_lbls, lastn_feats, max_lbl)
            cell_lbls = cell_lbls + [new_lbl]
            prev_feats = prev_feats + [feats]

        return (tp, cell_lbls)

    def df_get_imglist(self, exp, pos, trap, tp=None):
        df = self.meta.loc[(exp, pos, trap), ['cont_list_index', 'cellLabels']]
        return zip(df.index, [self.masks[i] for i in df['cont_list_index']])

    def predict_set(self, exp, pos, trap, tp=None):
        '''
        Predict labels using tp1-tp2 accuracy of prediction
        '''
        # print("Processing trap {}".format(exp, pos, trap))
        tp_img_tuple = *self.df_get_imglist(exp, pos, trap),
        tp, lbl_list = self.predict_lbls_from_tpimgs(tp_img_tuple)
        # print("loc {}, {}, {}, labels: {}".format(exp, pos, trap, lbl_list))
        return lbl_list

        
    def compare_traps(self, exp, pos, trap):
        '''
        Error calculator for testing model and assignment heuristics.

        Uses the trap id to compare the amount of cells correctly predicted.
        This uses local indices, not whole timepoints. It returns the
        fraction of cells correctly predicted, and the timepoints of mistakes

        Returns:
        float: Fraction of cells correctly predicted
        list of 2-sized tuples: list of tp id of errors and the mistaken cell

        '''
        print("Processing trap {}, {}, {}".format(exp,pos,trap))
        new_cids = self.predict_set(exp, pos, trap)

        test_df = self.meta.loc(axis=0)[(exp, pos, trap)].copy()
        test_df['pred_cellLabels'] = new_cids

        orig = test_df['cellLabels'].values
        new = test_df['pred_cellLabels'].values
        local_indices = [[], []]

        # Case just defines if it is the test or new set
        # print("Making tp-wise comparison")
        for i, case in enumerate((zip(orig[:-1],
                                      orig[1:]), zip(new[:-1], new[1:]))):
            for prev_cells, pos_cells in case:
                local_assignment = [
                    prev_cells.index(cell) if cell in prev_cells else -1
                    for cell in pos_cells
                ]
                local_indices[i] += local_assignment

        # Flatten
        if len(local_indices) > 2: 
            flt_test, flt_new = [
                np.array([j for i in case for j in i]) for case in local_indices
            ]
            tp_list = np.array(
                [i for i, vals in enumerate(local_indices[0]) for j in vals])
        else:
            flt_test, flt_new = [
                np.array([i for i in case]) for case in local_indices
            ]
            # tp_list = np.array(
            #     [i for i, vals in enumerate(local_indices[0]) for j in vals])

        correct = flt_test==flt_new
        if len(local_indices) > 2:
            error_list = tp_list[~correct]
        error_cid = test_df.iloc[1:]['cellLabels'].explode().dropna()[~correct].values
        frac_correct = np.mean(correct)

        print("Fraction of correct predictions", frac_correct)
        if len(local_indices)>2:
            return (frac_correct, list(zip(error_list, error_cid)))
        else:
            # print("Warning: Single set of tps for this position")
            return (frac_correct, error_cid)

    def predict_all(self):
        '''
        Predict all datasets defined in self.traps_loc
        '''
        stepsback = [2]
        threshs = [0.9]
        self.predictions  = {}
        for nstepsback in stepsback:
            for thresh in threshs:
                self.nstepsback = nstepsback
                self.tracker.nstepsback = nstepsback
                self.low_thresh = 1-thresh
                self.high_thresh = thresh
                self.thresh = thresh*5/8
                for address in self.traps_loc:
                    self.predictions[(nstepsback, thresh, address)] = self.predict_set(*address)

    def calculate_errsum(self):
        '''
        Calculate all errors, addresses of images with errors and error fractions. 
        '''
        frac_errs = {}
        all_errs = {}
        nerrs = {}
        stepsback = list(range(1, 3))
        threshs = [0.65, 0.95]
        for nstepsback in stepsback:
            for thresh in threshs:
                self.nstepsback = nstepsback
                self.tracker.nstepsback = nstepsback
                self.low_thresh = 1-thresh
                self.high_thresh = thresh
                self.thresh = thresh*5/8
                all_errs[(thresh, nstepsback)] = {}
                frac_errs[(thresh, nstepsback)] = []
                nerrs[(thresh, nstepsback)] = []
                for address in self.traps_loc:
                    fraction, errors = self.compare_traps(*address)
                    if len(errors):
                        all_errs[(thresh, nstepsback)][address] = errors
                        frac_errs[(thresh, nstepsback)].append(fraction)
                        nerrs[(thresh, nstepsback)].append(len(errors))
                    else:
                        nerrs[(thresh, nstepsback)].append(0)
                        frac_errs[(thresh, nstepsback)].append(1.0)
                        
                    

        return (frac_errs, all_errs, nerrs)


    def get_truth_matrix_from_pair(self, pair):
        '''
        Requires self.meta

        args:
        :pair: tuple of size 4 (experimentID, position, trap (tp1, tp2))

        returns

       :truth_mat: boolean ndarray of shape (ncells(tp1) x ncells(tp2)
        links cells in tp1 to cells in tp2
        '''
        
        clabs1 = self.meta.loc[pair[:3] + (pair[3][0], ), 'cellLabels']
        clabs2 = self.meta.loc[pair[:3] + (pair[3][1], ), 'cellLabels']

        truth_mat = gen_boolmat_from_clabs(clabs1, clabs2)

        return truth_mat

    def gen_cm_stats(self, pair, thresh=0.7):
        '''
        Calculate confusion matrix for a pair of pos-timepoints
        '''

        masks = [self.masks[i] for i in self.meta.loc[pair,'cont_list_index']]
        feats = [self.tracker.calc_feats_from_mask(mask) for mask in masks]
        if len(feats) > 3:
            print('bug')
        ndarray = self.tracker.calc_feat_ndarray(*feats)
        self.tracker.low_thresh = 1-thresh
        self.tracker.high_thresh = thresh
        prob_mat = self.tracker.predict_proba_from_ndarray(ndarray)
        pred_mat = prob_mat > thresh
        
        true_mat = self.get_truth_matrix_from_pair(pair)
        if not len(true_mat) and not len(pred_mat):
            return(0,0,0,0)


        true_flat = true_mat.flatten()
        pred_flat = pred_mat.flatten()

        true_pos = np.sum(true_flat & pred_flat)
        false_pos = np.sum(true_flat & ~pred_flat)
        false_neg = np.sum(~true_flat & pred_flat)
        true_neg = np.sum(~true_flat & ~pred_flat)

        return (true_pos, false_pos, false_neg, true_neg)

    def extract_pairs_from_trap(self, trap_loc):
        subdf = self.meta[['list_index', 'cellLabels'
                             ]].loc(axis=0)[trap_loc]
        pairs = [
            trap_loc + tuple((pair, ))
            for pair in zip(subdf.index[:-1], subdf.index[1:])
        ]

        return pairs

    def gen_pairlist(self):
        self.pairs = [self.extract_pairs_from_trap(trap) for trap in self.traps_loc]

    def gen_cm_from_pairs(self, thresh=0.5):
        con_mat = {}
        con_mat['tp'] = 0
        con_mat['fp'] = 0
        con_mat['fn'] = 0
        con_mat['tn'] = 0
        for pairset in self.pairs:
            for pair in pairset:
                res = self.gen_cm_stats(pair, thresh=thresh)
                con_mat['tp'] += res[0]
                con_mat['fp'] += res[1]
                con_mat['fn'] += res[2]
                con_mat['tn'] += res[3]
        self._con_mat = con_mat
        return self._con_mat


    def gen_errorplots(self):
        '''
        Calculates the trap-wise error and averages across a position.
       '''
        self.frac_errs, self.all_errs, self.nerrs = self.calculate_errsum()

        nerrs_df = pd.DataFrame(self.nerrs).melt()
        frac_df = pd.DataFrame(self.frac_errs).melt()

        from matplotlib import pyplot as plt
        import seaborn as sns

        # ax = sns.barplot(x='variable_0', y='value', data=frac_df)
        ax = sns.barplot(x='variable_1', y='value', hue='variable_0', data=frac_df);
        ax.set(xlabel='Backtrace depth',
               ylabel='Fraction of correct assignments',
               ylim=(0.9, 1))
        plt.legend(title="Threshold")
        plt.savefig('tracker_benchmark_btdepth.png')
        plt.show()

    # def plot_pair(self, address)
    
def gen_boolmat_from_clabs(clabs1, clabs2):
    if not np.any(clabs1) and not np.any(clabs2):
        return(np.array([]))

    boolmat = np.zeros((len(clabs1), len(clabs2))).astype(bool)
    for i, lab1 in enumerate(clabs1):
        for j, lab2 in enumerate(clabs2):
            if lab1==lab2:
                boolmat[i, j] = True

    return boolmat

def gen_stats_dict(results):
    '''
    Generates a dictionary using results from different binary classification tasks,
    for example, using different thresholds

    output

    dictionary containing the name of statistic as a key and a list
    of that statistic for the data subsets.
    '''
    funs = (get_precision, get_recall, get_tnr, get_balanced_acc)
    names = ('precision', 'recall', 'TNR', 'balanced_acc')
    stats_dict = {name : [fun(res) for res in results]
                  for fun, name in zip(funs, names)}

    return stats_dict

def get_precision(res_dict):
    return (res_dict['tp']) / (res_dict['tp'] + res_dict['fp'])

def get_recall(res_dict):
    return res_dict['tp'] / (res_dict['tp'] + res_dict['fn'])

def get_tnr(res_dict):
    return res_dict['tn'] / (res_dict['tn'] + res_dict['fp'])

def get_balanced_acc(res_dict):
    return (get_recall(res_dict) + get_tnr(res_dict))/2
