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
    '''
    def __init__(self, meta, model):
        self.indices = ['experimentID', 'position', 'trap', 'tp']
        self.cindices =  self.indices + ['cellLabels']
        self.meta = meta
        self.tracker = Tracker(ctrack_model = model)
        # self.tracker.get_new_lbls(None, prev_lbls, prev_feats, max_lbl, new_feats)

    def iterate_tracker(self):
        new_feats=None
        self.tracker.get_new_lbls(None, [], [], 0, new_feats=None)
         
    def filter(self, exp, pos, trap, tp=None):

        return df.loc(axis=0)[(exp,pos,trap)]

    @property
    def masks(self):
        if not hasattr(self, '_masks'):
            self._masks= [load_tiled_image(fname)[0] for
                              fname in self.meta['filename']]
            for i, mask in enumerate(self._masks):
                for j in range(mask.shape[2]):
                    self._masks[i][..., j] = binary_fill_holes(self._masks[i][..., j])

        return self._masks

    # @property
    # def props(self):
    #     if not hasattr(self, '_props'):
    #         self._props = [self.tracker.calc_feats_from_masks(
    #             [masks[..., i] for i in range(masks.shape[2])])
    #                        for masks in self.masks]

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
        df = self.meta.loc[(exp, pos, trap), ['list_index', 'cell_id']]
        return zip(df.index, self.masks[df['list_index'].values])

    def split_outlines(self, filename, cell_id):
        '''
        Process trap using row of location DataFrame. if get_props=True it
        returns the calculated properties. If it is False it returns a list
        of outlines.

        The main use of this function is to use the filename and default ids to
        produce a list of outlines or the corresponding proprieties.
        '''

        ncells = len(aslist(cell_id))
        img = imread(filename)
        cellolines = np.hsplit(binary_fill_holes(img).astype(int), ncells)
        # TODO: convert list to ndarray for BABY compatibility and remove cell_id

        return cellolines

    def predict_set(self, exp, pos, trap, tp=None):

        print("Processing trap %s" % str(exp, pos, trap))
        tp_img_tuple = *self.df_get_imglist(exp, pos, trap),
        tp, lbl_list = self.predict_lbls_from_tpimgs(tp_img_tuple)
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
        printf("Testing trap {}, {}, {}".format(exp,pos,trap))
        new_cids = self.predict_set(exp, pos, trap)

        test_df = self.meta.loc(axis=0)[(exp, pos, trap)]
        test_df.loc[:, 'pred_cellLabels'] = new_cids

        test = test_df['cellLabels'].values
        new = test_df['pred_cellLabels'].values
        local_indices = [[], []]
        #
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
        correct = flt_test == flt_new
        error_list = tp_list[~correct]
        error_cid = test_df.iloc[1:]['cellLabels'].explode()[~correct].values
        frac_correct = np.mean(correct)

        print(frac_correct)
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
                for i in range(1, 19):
                    fraction, errors = self.compare_traps((i, 4))
                    all_errs[(thresh, nstepsback)].append(errors)
                    frac_errs[(thresh, nstepsback)].append(fraction)
                    nerrs[(thresh, nstepsback)].append(len(errors))

        return (frac_errs, all_errs, nerrs)

def get_cells(filename, cell_id):
    '''
    Read a filename and cell id (only to get the number of cells)
    and return the ndarray containing the cells in the z-axis
    ---
    input
    filename: str indication the png location
    cell_id: list of ints, where the ints are cell labels

    output
    ndarray (size_x, size_y, ncells). ndarray containing the mask for cells
    in the z-axis
    '''

    ncells = len(cell_id)
    img = imread(filename)
    cell_masks = np.hsplit(binary_fill_holes(img).astype(int), ncells)

    return np.dstack(cell_masks)
