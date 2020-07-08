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
        self.test = self.predict_set(*self.traps_loc[0])
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
        frac_errs, all_errs, nerrs = self.calculate_errsum()

        nerrs_df = pd.DataFrame(nerrs).melt()
        frac_df = pd.DataFrame(frac_errs).melt()

        from matplotlib import pyplot as plt
        import seaborn as sns

        ax = sns.barplot(x='variable_0', y='value', data=frac_df)
        ax.set(xlabel='Backtrack depth',
               ylabel='Fraction of correct assignments',
               ylim=(0.9, 1))
        plt.savefig('tracker_benchmark_btdepth.png')
        plt.show()

        # TODO check if it is worth incorporating absolute number of errors
        # ax = sns.barplot(x='variable_1', y='value', data=nerrs_df)
        # ax.set(xlabel='Backtrack depth', ylabel='Number of mistakes')
        # plt.savefig('tracker_benchmark_nmistakes.png')
        # plt.show()
