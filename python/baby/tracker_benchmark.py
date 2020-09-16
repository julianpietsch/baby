import numpy as np
import pickle
import pandas as pd

from baby.tracker import Tracker
from baby.io import load_tiled_image
# from .tracker import Tracker
# from .io import load_tiled_image

from scipy.ndimage import binary_fill_holes
from skimage.measure import regionprops_table

class TrackBenchmarker:
    '''
    Takes a metadata dataframe and a model and estimates the prediction in a trap-wise manner.
    '''
    def __init__(self, meta, model):
        self.indices = ['experimentID', 'position', 'trap', 'tp']
        self.cindices =  self.indices + ['cellLabels']
        self.meta = meta
        self.meta['cont_list_index'] = [i for i in range(len(self.meta))]
        self.tracker = Tracker(ctrack_model = model)
        self.nstepsback = self.tracker.nstepsback
        self.traps_loc
        # self.test = self.predict_set(*self.traps_loc[0])
        # self.calculate_errsum()

    @property
    def traps_loc(self):
        '''
        Generates a list of trap locations using the metadata.
        '''
        if not hasattr(self, '_traps_loc'):

            traps = np.unique([ind[:self.indices.index('trap')+1] for ind in self.meta.index], axis=0)
            traps = [(ind[0], *map(int, ind[1:])) for ind in traps] # str->int conversion
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
                    self._masks[i][..., j] = binary_fill_holes(self._masks[i][..., j])

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
        print("Processing trap {}".format(exp, pos, trap))
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
        print("Testing trap {}, {}, {}".format(exp,pos,trap))
        new_cids = self.predict_set(exp, pos, trap)

        test_df = self.meta.loc(axis=0)[(exp, pos, trap)]
        test_df['pred_cellLabels'] = new_cids

        test = test_df['cellLabels'].values
        new = test_df['pred_cellLabels'].values
        local_indices = [[], []]


        #      # Case just defines if it is the test or new set
        print("Making tp-wise comparison")
        for i, case in enumerate((zip(test[:-1],
                                      test[1:]), zip(new[:-1], new[1:]))):
            for prev_cells, pos_cells in case:
                local_assignment = [
                    prev_cells.index(cell) if cell in prev_cells else -1
                    for cell in pos_cells
                ]
                local_indices[i].append(local_assignment)

        # Flatten
        flt_test, flt_new = [
            np.array([j for i in case for j in i]) for case in local_indices
        ]
        tp_list = np.array(
            [i for i, vals in enumerate(local_indices[0]) for j in vals])
        correct = flt_test==flt_new
        error_list = tp_list[~correct]
        error_cid = test_df.iloc[1:]['cellLabels'].explode()[~correct].values
        frac_correct = np.mean(correct)

        print("Fraction of correct predictions", frac_correct)
        return (frac_correct, list(zip(error_list, error_cid)))

    def calculate_errsum(self):
        frac_errs = {}
        all_errs = {}
        nerrs = {}
        stepsback = list(range(1, 6))
        threshs = [0.1, 0.5, 0.9]
        for nstepsback in stepsback:
            for thresh in threshs:
                self.nstepsback = nstepsback
                self.tracker.nstepsback = nstepsback
                # self.tracker.red_fun = rand
                self.ctrack_thresh = thresh
                all_errs[(thresh, nstepsback)] = []
                frac_errs[(thresh, nstepsback)] = []
                nerrs[(thresh, nstepsback)] = []
                for address in self.traps_loc:
                    fraction, errors = self.compare_traps(*address)
                    all_errs[(thresh, nstepsback)].append(errors)
                    frac_errs[(thresh, nstepsback)].append(fraction)
                    nerrs[(thresh, nstepsback)].append(len(errors))

        return (frac_errs, all_errs, nerrs)

    def gen_errorplots(self):
        '''
        Calculates the trap-wise error and averages across a position.
       '''
        self.frac_errs, self.all_errs, self.nerrs = self.calculate_errsum()

        nerrs_df = pd.DataFrame(self.nerrs).melt()
        frac_df = pd.DataFrame(self.frac_errs).melt()

        from matplotlib import pyplot as plt
        import seaborn as sns

        ax = sns.barplot(x='variable_0', y='value', data=frac_df)
        ax.set(xlabel='Backtrace depth',
               ylabel='Fraction of correct assignments',
               ylim=(0.9, 1))
        plt.savefig('tracker_benchmark_btdepth.png')
        plt.show()
    

    def get_truth_matrix_from_pair(self, pair):
        '''
        Requires self.meta

        args:
        :pair: tuple of size 4 (experimentID, position, trap (tp1, tp2))

        output

       :truth_mat: boolean ndarray of shape (ncells(tp1) x ncells(tp2)
        links cells in tp1 to cells in tp2
        '''
        
        clabs1 = self.meta.loc[pair[:3] + (pair[3][0], ), 'cellLabels']
        clabs2 = self.meta.loc[pair[:3] + (pair[3][1], ), 'cellLabels']

        truth_mat = gen_boolmat_from_clabs(clabs1, clabs2)

        return truth_mat

    def gen_cm_stats(self, pair, red_fun=np.nanmax, thresh=0.5):

        masks = [self.masks[i] for i in self.meta.loc[pair,'cont_list_index']]
        feats = [self.tracker.calc_feats_from_mask(mask) for mask in masks]
        ndarray = self.tracker.calc_feat_ndarray(*feats)
        prob_mat = self.tracker.predict_proba_from_ndarray(ndarray)
        pred_mat = prob_mat > thresh
        
        true_mat = self.get_truth_matrix_from_pair(pair)

        true_flat = true_mat.flatten()
        pred_flat = pred_mat.flatten()

        acc=np.sum(true_flat==pred_flat)/len(true_flat)
        print('Fraction correct: ', acc)
        true_pos = np.sum(true_flat & pred_flat)
        false_pos = np.sum(true_flat & ~pred_flat)
        false_neg = np.sum(~true_flat & pred_flat)

        return (acc, true_pos, false_pos, false_neg)

def gen_boolmat_from_clabs(clabs1, clabs2):
    boolmat = np.zeros((len(clabs1), len(clabs2))).astype(bool)
    for i, lab1 in enumerate(clabs1):
        for j, lab2 in enumerate(clabs2):
            if lab1==lab2:
                boolmat[i, j] = True

    return boolmat