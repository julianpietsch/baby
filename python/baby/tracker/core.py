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

'''
TrackerCoordinator class to coordinate cell tracking and bud assignment.
'''
from os.path import join, dirname
from collections import Counter
import pickle
import numpy as np
from pathlib import Path, PosixPath
from skimage.measure import regionprops_table
from skimage.draw import polygon
from scipy.optimize import linear_sum_assignment
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from baby.errors import BadOutput
from baby.tracker.utils import calc_barycentre, pick_baryfun

models_path = join(dirname(__file__), '../models')

class FeatureCalculator:
    '''
    Base class for making use of regionprops-based features.
    If no features are offered it uses most of them.

    This class is not to be used directly
    '''
    def __init__(self, feats2use, trapfeats=None, aweights=None,
                 px_size=None):

        self.feats2use = feats2use

        if trapfeats is None:
            trapfeats=()
        self.trapfeats = trapfeats

        if aweights is None:
            aweights=None
        self.aweights = aweights

        if px_size is None:
            px_size = 0.263
        self.px_size = px_size

        self.outfeats = self.get_outfeats()

        self.tfeats = self.outfeats + self.trapfeats
        self.ntfeats = len(self.tfeats)

        self.set_named_ids()

    def get_outfeats(self, feats2use=None):
        if feats2use is None:
            feats2use = self.feats2use
        outfeats = tuple(
            regionprops_table(np.diag((1, 0)),
                              properties=self.feats2use).keys())
        return(outfeats)

    def set_named_ids(self):
        # Convenience function to call important feats by name
        if 'centroid-0' in self.outfeats:
            self.xind = self.outfeats.index('centroid-0')
        if 'centroid-1' in self.outfeats:
            self.yind = self.outfeats.index('centroid-1')
        if 'area' in self.outfeats:
            self.aind = self.outfeats.index('area')

    def load_model(self, path, fname):
            model_file = join(path, fname)
            with open(model_file, 'rb') as file_to_load:
                model = pickle.load(file_to_load)

            return model

    def calc_feats_from_mask(self, masks, feats2use=None, trapfeats=None,
                             norm=True, px_size=None):
        '''
        Calculate feature ndarray from ndarray of cell masks
        ---
        input

        :masks: ndarray (ncells, x_size, y_size), typically dtype bool
        :feats2use: list of strings with the feature properties to extract.
            If it is None it uses the ones set in self.feats2use.
        :trapfeats: List of str with additional features to use
            calculated immediately after basic features.
        :norm: bool, if True normalises mask to a defined px_size.
        :px_size: float, used to normalise the images.

        returns

        (ncells, nfeats) ndarray of features for input masks
        '''
        if px_size is None:
            px_size = self.px_size

        if feats2use is None:
            feats2use = self.feats2use

        if trapfeats is None:
            trapfeats = self.trapfeats

        feats = np.empty((masks.shape[0], self.ntfeats)) # ncells * nfeats
        if masks.sum():
            for i, mask in enumerate(masks):
                cell_feats = []
                for feat in regionprops_table(mask.astype(int),
                        properties=feats2use).values():
                    cell_feats.append(feat[0])

                if norm:
                    cell_feats = self.norm_feats(cell_feats, px_size)

                feats[i, :len(self.outfeats)] = cell_feats

            if trapfeats:
            
                tfeats = self.calc_trapfeats(feats)
                feats[:, len(self.outfeats):self.ntfeats] = tfeats
        else:

            feats = np.zeros((0, self.ntfeats))


        return feats

    def calc_trapfeats(self, basefeats):
        '''
        Calculate trap-based features
        using basic ones.
        :basefeats: (n basic outfeats) 1-D array with features outputed by
            skimage.measure.regionprops_table


        requires
            self.aind
            self.aweights
            self.xind
            self.yind
            self.trapfeats

        returns
        (ntrapfeats) 1-D array with 
        '''
        if self.aweights is not None:
            weights = basefeats[:, self.aind]
        else:
            weights = None

        barycentre = calc_barycentre(basefeats[:, [self.xind, self.yind]],
                                     weights = weights)

        trapfeat_nd = np.empty((basefeats.shape[0], len(self.trapfeats)))
        for i, trapfeat in enumerate(self.trapfeats):
            trapfeat_nd[:,i] = pick_baryfun(trapfeat)(
                basefeats[:, [self.xind, self.yind]], barycentre)

        return(trapfeat_nd)

    def norm_feats(self, feats, px_size):
        '''
        input

        :feats: list of float, extracted features obtained from regionprops_table
        :px_size: Value used to normalise the images.

        returns
        normalised list of feature values
        '''
        area = px_size**2

        degrees = {'linear': px_size, 'square':area}
        degrees_feats = {'linear':['minor_axis_length',  'major_axis_length',
                        'perimeter', 'perimeter_crofton',
                                   'equivalent_diameter'],
                       'square': ['area', 'convex_area']}

        if 'centroid' in self.feats2use:
            feats[self.outfeats.index('centroid-0')] /= px_size
            feats[self.outfeats.index('centroid-1')] /= px_size


        for deg, feat_names in degrees_feats.items():
            for feat_name in feat_names:
                if feat_name in self.feats2use:
                    feats[self.outfeats.index(feat_name)] /= degrees[deg]

        return feats

class CellTracker(FeatureCalculator):
    '''
    Class used to manage cell tracking. You can call it using an existing model or
    use the inherited CellTrainer to get a new one.

    Initialization parameters:

    :model: sklearn.ensemble.RandomForestClassifier object
    :trapfeats: Features to manually calculate within a trap
    :extra_feats: Additional features to calculate
    :model: Model to use, if provided ignores all other args but threshs
    :bak_model: Backup mode to use when prediction is unsure
    :nstepsback: int Number of timepoints to go back
    :thresh: float Cut-off value to assume a cell is not new
    :low_thresh: Lower thresh for model switching
    :high_thresh: Higher thresh for model switching.
        Probabilities between these two thresholds summon the
        backup model.

    :aweights: area weight for barycentre calculations


    # Feature order in array features
    1. basic features
    2. trap features (within a trap)
    3. extra features (process between imgs)

    '''
    def __init__(self,
                 feats2use=None,
                 trapfeats=None,
                 extra_feats=None,
                 model=None,
                 bak_model=None,
                 thresh=None,
                 low_thresh=None,
                 high_thresh=None,
                 nstepsback=None,
                 aweights=None,
                 red_fun=None,
                 **kwargs):

        if trapfeats is None:
            trapfeats = ()

        if extra_feats is None:
            extra_feats = ()

        if type(model) is str or type(model) is PosixPath: 
            with open(Path(model), 'rb') as f:
                model = pickle.load(f)

        if type(bak_model) is str or type(bak_model) is PosixPath: 
            with open(Path(bak_model), 'rb') as f:
                bak_model = pickle.load(f)

        if aweights is None:
            self.aweights = None

        if feats2use is None: # Ignore this block when training
            if model is None:
                model = self.load_model( models_path,
                                         'ct_rf_20210201_12.pkl')
            if bak_model is None:
                bak_model = self.load_model( models_path,
                                         'ct_rf_20210125_9.pkl')
            self.model = model
            self.bak_model = bak_model

            main_feats = model.all_ifeats
            bak_feats = bak_model.all_ifeats
            feats2use, trapfeats, extra_feats = [
                tuple(set(main).union(bak)) for main,bak
                                                 in zip(main_feats, bak_feats)]

            
        # Training AND non-training part
        super().__init__(feats2use, trapfeats=trapfeats, **kwargs)

        self.extra_feats = extra_feats

        self.all_ofeats = self.outfeats + trapfeats + extra_feats

        self.noutfeats = len(self.all_ofeats)

        
        if hasattr(self, 'bak_model'): # Back to non-training only

            self.mainof_ids = [self.all_ofeats.index(f) for f in self.model.all_ofeats]

            self.bakof_ids = [self.all_ofeats.index(f) for f in self.bak_model.all_ofeats]



        if nstepsback is None:
            nstepsback = 3
        self.nstepsback = nstepsback

        if thresh is None:
            thresh = 0.5
        self.thresh = thresh
        if low_thresh is None:
            low_thresh = 0.1
        if high_thresh is None:
            high_thresh = 0.9
        self.low_thresh, self.high_thresh = low_thresh, high_thresh

        if red_fun is None:
            red_fun = np.nanmax
        self.red_fun = red_fun

    def get_feats2use(self):
        '''
        Return feats to be used from a loaded random forest model model
        '''
        nfeats = get_nfeats_from_model(self.model)
        nfeats_bak = get_nfeats_from_model(self.bak_model)
        # max_nfeats = max((nfeats, nfeats_bak))

        return(switch_case_nfeats(nfeats), switch_case_nfeats(nfeats_bak))

    def calc_feat_ndarray(self, prev_feats, new_feats):
        '''
        Calculate feature ndarray using two ndarrays of features.
        ---

        input

        :prev_feats: ndarray (ncells, nfeats) of timepoint 1
        :new_feats: ndarray (ncells, nfeats) of timepoint 2

        returns

        :n3darray: ndarray (ncells_prev, ncells_new, nfeats) containing a
            cell-wise substraction of the features in the input ndarrays.
        '''
        if not (new_feats.any() and prev_feats.any()):
            return np.array([])

        n3darray = np.empty((len(prev_feats), len(new_feats),
                             self.noutfeats ))

        # print('self: ', self, ' self.ntfeats: ', self.ntfeats, ' featsshape: ', new_feats.shape)
        for i in range(self.ntfeats):
            n3darray[..., i] = np.subtract.outer(prev_feats[:, i],
                                                 new_feats[:, i])

        n3darray = self.calc_dtfeats(n3darray)

        return n3darray
            
    def calc_dtfeats(self, n3darray):
        '''
        Calculates features obtained between timepoints, such as distance
        for every pair of cells from t1 to t2.
        ---

        input

        :n3darray: ndarray (ncells_prev, ncells_new, nfeats) containing a
            cell-wise substraction of the features in the input ndarrays.

        returns

        :n3darray: updated 3-D array with dtfeats added
        '''
        for i, feat in enumerate(self.all_ofeats):
            if feat == 'distance':
                n3darray[..., i] = np.sqrt(
                    n3darray[..., self.xind]**2 +
                    n3darray[..., self.yind]**2) # TODO clean this expression
                # n3darray[..., i] = np.sqrt(np.sum(n3darray[...,
                # [self.xind, self.yind]]**2))


        return(n3darray)

    def assign_lbls(self, pred_3darray, prev_lbls, red_fun=None):
        '''Assign labels using a prediction matrix of nxmxl where n is the number
        of cells in the previous image, m the number of steps back considered
        and l in the new image. It assigns the
        number zero if it doesn't find the cell.
        ---
        input

        :pred_3darray: Probability n x m x l array obtained as an output of rforest
        :prev_labels: List of cell labels for previous timepoint to be compared.
        :red_fun: Function used to collapse the previous timepoints into one.
            If none provided it uses maximum and ignores np.nans.

        returns

        :new_lbls: ndarray of newly assigned labels obtained, new cells as
        zero.
        '''
        if red_fun is None:
            red_fun = self.red_fun

        new_lbls = np.zeros(pred_3darray.shape[2], dtype=int)
        pred_matrix = np.apply_along_axis(red_fun, 1, pred_3darray)

        if pred_matrix.any():
            # assign available hits
            row_ids, col_ids = linear_sum_assignment(-pred_matrix)
            for i,j in zip(row_ids, col_ids):
                if  pred_matrix[i, j] > self.thresh:
                    new_lbls[j] = prev_lbls[i]

        return new_lbls

    def predict_proba_from_ndarray(self, array_3d, boolean=False):
        '''

        input

        :array_3d: (ncells_tp1, ncells_tp2, out_feats) ndarray
        :boolean: bool, if False returns probability, if True returns prediction

        requires
        :self.model:
        :self.mainof_ids: list of indices corresponding to the main model's features
        :self.bakof_ids: list of indices corresponding to the backup model's features

        returns

        (ncells_tp1, ncells_tp2) ndarray with probabilities or prediction
            of cell identities depending on "boolean" arg.

        '''

        if array_3d.size == 0:
            return np.array([])

        predict_fun = self.model.predict if boolean else \
            self.model.predict_proba
        bak_pred_fun = self.bak_model.predict if boolean else \
            self.bak_model.predict_proba

        orig_shape = array_3d.shape[:2]

        # Flatten for predictions and then reshape back into matrix
        pred_list = []
        for vec in array_3d.reshape(-1, array_3d.shape[2]):
            prob = predict_fun(
                vec[self.mainof_ids].reshape(1,-1))[0][1]
            if self.low_thresh < prob < self.high_thresh:
                bak_prob = bak_pred_fun(vec[self.bakof_ids].reshape(
                1,-1))[0][1] 
                prob = max(prob, bak_prob)
            pred_list.append(prob)

        pred_matrix = np.array(pred_list).reshape(orig_shape)

        return pred_matrix

    def get_new_lbls(self,
                     new_img,
                     prev_lbls,
                     prev_feats,
                     max_lbl,
                     new_feats=None,
                     px_size=None):
        '''
        Core function to calculate the new cell labels.

        ----

        input

        :new_img: ndarray (len, width, ncells) containing the cell outlines
        :max_lbl: int indicating the last assigned cell label
        :prev_feats: list of ndarrays of size (ncells x nfeatures)
        containing the features of previous timepoints
        :prev_lbls: list of list of ints corresponding to the cell labels in
            the previous timepoints
        :new_feats: (optional) Directly give a feature ndarray. It ignores
            new_img if given.

        returns

        :new_lbls: list of labels assigned to new timepoint
        :new_feats: list of ndarrays containing the updated features
        :new_max: updated max cell label assigned

        '''

        if new_feats is None:
            new_feats = self.calc_feats_from_mask(new_img)

        if new_feats.any():
            if np.any([len(prev_feat) for prev_feat in prev_feats]):
                counts = Counter([lbl for lbl_set in prev_lbls for lbl in lbl_set])
                lbls_order = list(counts.keys())
                probs = np.full(
                    (len(lbls_order), self.nstepsback, len(new_feats)), np.nan)

                for i, (lblset, prev_feat) in enumerate(zip(prev_lbls, prev_feats)):
                    if len(prev_feat):
                        feats_3darray = self.calc_feat_ndarray(
                            prev_feat, new_feats)

                        pred_matrix = self.predict_proba_from_ndarray(feats_3darray)

                        for j,lbl in enumerate(lblset):
                            probs[lbls_order.index(lbl), i, :] = pred_matrix[j,:]

                new_lbls = self.assign_lbls(probs, lbls_order)
                new_cells_pos = new_lbls==0
                new_max = max_lbl + sum(new_cells_pos)
                new_lbls[new_cells_pos] = [*range(max_lbl+1, new_max+1)]

                # ensure that label output is consistently a list
                new_lbls = new_lbls.tolist()

            else: 
                started = False if max_lbl else True

                new_lbls = [*range(max_lbl + started, max_lbl +
                                   len(new_feats) + started)]

                new_max = max_lbl + len(new_feats) 

        else:
            return ([], [], max_lbl)
        return (new_lbls, new_feats, new_max)

class BudTracker(FeatureCalculator):
    def __init__(self,
                 model=None,
                 feats2use=None,
                 **kwargs):

        if model is None:
            model_file = join(models_path,
                                      'mb_model_20201022.pkl')
            with open(model_file, 'rb') as file_to_load:
                model = pickle.load(file_to_load)
        self.model = model

        if feats2use is None:
            feats2use = ['centroid', 'area', 'minor_axis_length']
        super().__init__(feats2use, **kwargs)

        self.a_ind = self.outfeats.index('area')
        self.ma_ind = self.outfeats.index('minor_axis_length')



    ### Assign mother-
    def calc_mother_bud_stats(self, p_budneck, p_bud, masks, feats=None):
        '''
        ---

        input

        :p_budneck: 2d ndarray (size_x, size_y) giving the probability that a
            pixel corresponds to a bud neck
        :p_bud: 2d ndarray (size_x, size_y) giving the probability that a pixel
            corresponds to a bud
        :masks: 3d ndarray (ncells, size_x, size_y)
        :feats: ndarray (ncells, nfeats)

        NB: ASSUMES FEATS HAVE ALREADY BEEN NORMALISED!

        returns

        :n2darray: 2d ndarray (ncells x ncells, n_feats) specifying,
            for each pair of cells in the masks array, the features used for
            mother-bud pair prediction (as per 'feats2use')
        '''

        if feats is None:
            feats = self.calc_feats_from_mask(masks)
        elif len(feats) != len(masks):
            raise Exception('number of features must match number of masks')

        ncells = len(masks)

        # Entries will be NaN unless validly specified below
        p_bud_mat = np.nan * np.ones((ncells, ncells))
        p_budneck_mat = np.nan * np.ones((ncells, ncells))
        budneck_ratio_mat = np.nan * np.ones((ncells, ncells))
        size_ratio_mat = np.nan * np.ones((ncells, ncells))
        adjacency_mat = np.nan * np.ones((ncells, ncells))

        for m in range(ncells):
            for d in range(ncells):
                if m == d:
                    # Mother-bud pairs can only be between different cells
                    continue

                p_bud_mat[m, d] = np.mean(p_bud[masks[d].astype('bool')])

                a_i = self.a_ind
                size_ratio_mat[m, d] = feats[m, a_i] / feats[d, a_i]

                # Draw rectangle
                r_points = self.get_rpoints(feats, d, m)
                if r_points is None:
                    continue
                rr, cc = polygon(r_points[0, :], r_points[1, :],
                                 p_budneck.shape)
                if len(rr) == 0:
                    # Rectangles with zero size are not informative
                    continue

                r_im = np.zeros(p_budneck.shape, dtype='bool')
                r_im[rr, cc] = True

                # Calculate the mean of bud neck probabilities greater than some threshold
                pbn = p_budneck[r_im].flatten()

                pbn = pbn[pbn > 0.2]
                p_budneck_mat[m, d] = np.mean(pbn) if len(pbn) > 0 else 0

                # Normalise number of bud-neck positive pixels by the scale of
                # the bud (a value proportional to circumference):
                raw_circumf_est = np.sqrt(feats[d, a_i]) * self.px_size
                budneck_ratio_mat[m, d] = pbn.sum() / raw_circumf_est

                # Adjacency is the proportion of the joining rectangle that overlaps the mother daughter union
                md_union = masks[m] | masks[d]
                adjacency_mat[m, d] = np.sum(md_union & r_im) / np.sum(r_im)

        return np.hstack([
            s.flatten()[:, np.newaxis]
            for s in (p_bud_mat, size_ratio_mat, p_budneck_mat,
                budneck_ratio_mat, adjacency_mat)
        ])

    def predict_mother_bud(self, p_budneck, p_bud, masks, feats=None):
        '''
        ---

        input

        :p_budneck: 2d ndarray (size_x, size_y) giving the probability that a
            pixel corresponds to a bud neck
        :p_bud: 2d ndarray (size_x, size_y) giving the probability that a pixel
            corresponds to a bud
        :masks: 3d ndarray (ncells, size_x, size_y)
        :feats: ndarray (ncells, nfeats)

        returns

        :n2darray: 2d ndarray (ncells, ncells) giving probability that a cell
            (row) is a mother to another cell (column)
        '''

        ncells = len(masks)

        mb_stats = self.calc_mother_bud_stats(
                p_budneck, p_bud, masks, feats)

        good_stats = ~np.isnan(mb_stats).any(axis=1)
        # Assume probability of bud assignment for any rows that are NaN will
        # be zero
        ba_probs = np.zeros(ncells**2)
        if good_stats.any():
            ba_probs[good_stats] = self.model.predict_proba(
                mb_stats[good_stats, :])[:, 1]
        ba_probs = ba_probs.reshape((ncells, ) * 2)

        return ba_probs

    def get_rpoints(self, feats, d, m):
        '''
        Draw a rectangle in the budneck of cells
        ---

        NB: ASSUMES FEATS HAVE ALREADY BEEN NORMALISED!

        input

        feats: 2d ndarray (ncells, nfeats)

        returns

        r_points: 2d ndarray (2,4) with the coordinates of the rectangle corner

        '''

        # Get un-normalised features for m-d pair
        m_centre = feats[m, :2] * self.px_size
        d_centre = feats[d, :2] * self.px_size
        r_width = np.max((2, feats[d, self.ma_ind] * self.px_size * 0.25))

        # Draw connecting rectangle
        r_hvec = d_centre - m_centre
        r_wvec = np.matmul(np.array([[0, -1], [1, 0]]), r_hvec)
        r_wvec_len = np.linalg.norm(r_wvec)
        if r_wvec_len == 0:
            return None
        r_wvec = r_width * r_wvec / r_wvec_len
        r_points = np.zeros((2, 4))
        r_points[:, 0] = m_centre - 0.5 * r_wvec
        r_points[:, 1] = r_points[:, 0] + r_hvec
        r_points[:, 2] = r_points[:, 1] + r_wvec
        r_points[:, 3] = r_points[:, 2] - r_hvec

        return r_points

class MasterTracker(FeatureCalculator):
    '''
    Coordinates the data transmission from CellTracker to BudTracker to
    reduce number of calls to regionprops function.

    input
    :ctrack_args: dict with arguments to pass on to CellTracker constructor
        if None it passes all the features to use
    :btrack_args: dict with arguments to pass on to BudTracker constructor
        if None it passes all the features to use
    :**kwargs: additional arguments passed to FeatureCalculator constructor
    '''
    def __init__(self,
                 ctrack_args=None,
                 btrack_args=None,
                 min_bud_tps=3,
                 isbud_thresh=0.5,
                 **kwargs):
        if ctrack_args is None:
            ctrack_args = {}
        if 'px_size' not in ctrack_args and 'px_size' in kwargs:
            ctrack_args['px_size'] = kwargs['px_size']
        self.cell_tracker = CellTracker(**ctrack_args)

        if btrack_args is None:
            btrack_args = {}
        if 'px_size' not in btrack_args and 'px_size' in kwargs:
            btrack_args['px_size'] = kwargs['px_size']
        self.bud_tracker = BudTracker(**btrack_args)

        feats2use = tuple(set(self.cell_tracker.feats2use).union(set(
                self.bud_tracker.feats2use)))

        trapfeats = tuple(set(self.cell_tracker.trapfeats).union(set(
                self.bud_tracker.trapfeats)))

        super().__init__(feats2use, trapfeats=trapfeats, **kwargs)

        # Extract indices of the relevant features
        self.ct_idx = [self.tfeats.index(f) for f in self.cell_tracker.tfeats]
        # DONE Tests passing, TODO check if the change budtracker.outfeats to tfeats broke anything
        self.bt_idx = [self.tfeats.index(f) for f in self.bud_tracker.tfeats] 

        # Save bud assignment parameters
        self.min_bud_tps = min_bud_tps
        self.isbud_thresh = isbud_thresh

    def step_trackers(self,
                      masks,
                      p_budneck,
                      p_bud,
                      state=None,
                      assign_mothers=False,
                      return_baprobs=False,
                      keep_full_state=False):
        '''
        Calculate features and track cells and budassignments

        input

        :masks: 3d ndarray (ncells, size_x, size_y) containing cell masks
        :p_budneck: 2d ndarray (size_x, size_y) giving the probability that a
            pixel corresponds to a bud neck
        :p_bud: 2d ndarray (size_x, size_y) giving the probability that a pixel
            corresponds to a bud
        :state: running state for the tracker, or None for initialisation
        :assign_mothers: whether to include mother assignments in the returned
            returns
        :return_baprobs: whether to include bud assignment probability matrix
            in the returned output

        returns a dict consisting of

        :cell_label: list of int, the tracked global ID for each cell mask
        :state: the updated state to be used in a subsequent step
        :mother_assign: (optional) list of int, specifying the assigned mother
            for each cell
        :p_bud_assign: (optional) matrix (list of lists of floats), bud assignment
            probability matrix from `predict_mother_bud`
        '''

        if state is None:
            state = {}

        max_lbl = state.get('max_lbl', 0)
        cell_lbls = state.get('cell_lbls', [])
        prev_feats = state.get('prev_feats', [])

        # Get features for cells at this time point
        feats = self.calc_feats_from_mask(masks)
        # print(self.ct_idx)

        nstepsback = self.cell_tracker.nstepsback
        lastn_lbls = cell_lbls[-nstepsback:]
        lastn_feats = [fset[:,self.ct_idx] for fset in prev_feats[-nstepsback:]]

        new_lbls, _, max_lbl = self.cell_tracker.get_new_lbls(
            masks, lastn_lbls, lastn_feats, max_lbl, feats[:,self.ct_idx])

        # if necessary, allocate more memory for state vectors/matrices
        init = {
            'lifetime': np.zeros(0),  # vector (>=max_lbl)
            'p_is_mother': np.zeros(0),  # vector (>=max_lbl)
            'p_was_bud': np.zeros(0),  # vector (>=max_lbl)
            'ba_cum': np.zeros((0, 0))  # matrix (>=max_lbl, >=max_lbl)
        }

        for k, v in init.items():
            v = state.get(k, v)
            l = len(v)
            if max_lbl > l:
                state[k] = np.pad(v, (0, max_lbl + 32 - l), 'constant')

        lifetime = state.get('lifetime', init['lifetime'])
        p_is_mother = state.get('p_is_mother', init['p_is_mother'])
        p_was_bud = state.get('p_was_bud', init['p_was_bud'])
        ba_cum = state.get('ba_cum', init['ba_cum'])

        # Update lineage state variables
        if len(masks) > 0 and len(new_lbls) > 0:
            ba_probs = self.bud_tracker.predict_mother_bud(
                p_budneck, p_bud, masks, feats[:, self.bt_idx])
            lblinds = np.array(new_lbls) - 1  # new_lbls are indexed from 1
            lifetime[lblinds] += 1
            p_is_mother[lblinds] = np.maximum(p_is_mother[lblinds],
                                              ba_probs.sum(1))
            p_was_bud[lblinds] = np.maximum(p_was_bud[lblinds],
                                            ba_probs.max(0))
            ba_cum[np.ix_(
                lblinds,
                lblinds)] += ba_probs * (1 - p_is_mother[lblinds][None, ])
        else:
            ba_probs = np.zeros((0, 0))

        if not keep_full_state:
            cell_lbls = lastn_lbls
            prev_feats = prev_feats[-nstepsback:]

        # Finally update the state
        state = {
            'max_lbl': max_lbl,
            'cell_lbls': cell_lbls + [new_lbls],
            'prev_feats': prev_feats + [feats],
            'lifetime': lifetime,
            'p_is_mother': p_is_mother,
            'p_was_bud': p_was_bud,
            'ba_cum': ba_cum
        }

        output = {
            'cell_label': new_lbls,
            'state': state
        }

        if assign_mothers:
            if max_lbl > 0:
                # Calculate mother assignments for this time point
                ma = ba_cum[0:max_lbl, 0:max_lbl].argmax(0) + 1
                # Cell must have been a bud and been present for at least
                # min_bud_tps
                isbud = (p_was_bud[0:max_lbl] > self.isbud_thresh) & (
                        lifetime[0:max_lbl] >= self.min_bud_tps)
                ma[~isbud] = 0  # 0 indicates no assignment (lbls indexed from 1)
            else:
                ma = np.zeros(0)

            if np.any(ma == np.arange(1, len(ma) + 1)):
                raise BadOutput('Bud has been assigned as mother to itself')

            output['mother_assign'] = ma.tolist()

        if return_baprobs:
            output['p_bud_assign'] = ba_probs.tolist()

        return output


# Helper functions

def switch_case_nfeats(nfeats):
    '''
    Convenience TEMPORAL function to determine whether to use distance/location
    as a feature for tracking or not (nfeats=5 for no distance, 7 for distance)
    input
    number of feats

    returns
    list of main and extra feats based on the number of feats
    '''
    main_feats = {
        4 : [(
            'area', 'minor_axis_length', 'major_axis_length', 
            'bbox_area'), (), ()],
            # Including centroid
        7 : [('centroid', 'area', 'minor_axis_length', 'major_axis_length',
              'bbox_area', 'perimeter'), () , ()],
            # Including centroid and distance
        8 : [(
            'centroid', 'area', 'minor_axis_length', 'major_axis_length', 
            'bbox_area', 'perimeter'), (),
                                                    ('distance',)],
        9 : [(
            'centroid', 'area', 'minor_axis_length', 'major_axis_length', 
            'bbox_area', 'perimeter'), ('baryangle', 'barydist'),
                                                    ()],
        10 : [(
            'centroid', 'area', 'minor_axis_length', 'major_axis_length', 
            'bbox_area', 'perimeter'), ('baryangle', 'barydist'),
                                                    ('distance',)],
        12 : [(
            'centroid', 'area', 'minor_axis_length', 'major_axis_length', 
            'bbox_area', 'eccentricity', 'equivalent_diameter', 'solidity',
            'extent',
            'orientation', 'perimeter'), (), ()],
        14 : [(
            'centroid', 'area', 'minor_axis_length', 'major_axis_length', 
            'bbox_area', 'eccentricity', 'equivalent_diameter', 'solidity',
            'extent',
            'orientation', 'perimeter'), ('baryangle', 'barydist'), ()],
        15 : [(
            'centroid', 'area', 'minor_axis_length', 'major_axis_length', 
            'bbox_area', 'eccentricity', 'equivalent_diameter', 'solidity',
            'extent', 'orientation', 'perimeter'), ('baryangle', 'barydist'),
                                                    ('distance',)]
    }

    return(main_feats.get(nfeats, []))

def get_nfeats_from_model(model):
    if isinstance(model, SVC):
        nfeats = model.support_vectors_.shape[-1]
    elif isinstance(model, RandomForestClassifier):
        nfeats = model.n_features_

    return nfeats
