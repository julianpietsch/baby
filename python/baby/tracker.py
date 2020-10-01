''' Tracker class to perform cell tracking inside baby.
'''
from collections import Counter
import pickle
import numpy as np
from skimage.measure import regionprops_table 
from skimage.transform import resize
from skimage.draw import polygon
import os

models_path = os.path.join(os.path.dirname(__file__), '..', '..', 'models')

class Tracker:
    '''
    Class used to manage cell tracking.

    Initialization parameters:

    :ctrack_model: sklearn.ensemble.RandomForestClassifier object
    :ba_model: sklearn.ensemble.RandomForestClassifier object.
    :nstepsback: int Number of timepoints to go back
    :ctrac_thresh: float Cut-off value to assume a cell is not new
    '''

    def __init__(self,
                 ctrack_model=None,
                 ba_model=None,
                 feats2use=None,
                 ba_feats=None,
                 ctrack_thresh=None,
                 nstepsback=None,
                 red_fun=None):

        if ba_model is None:
            ba_model_file = os.path.join(models_path, 'baby_randomforest_20190906.pkl')
            with open(ba_model_file, 'rb') as file_to_load:
                ba_model = pickle.load(file_to_load)
        self.ba_model = ba_model

        if ctrack_model is None:
            ctrack_model_file = os.path.join(models_path,
                                      'ctrack_randomforest_20200903.pkl')
            with open(ctrack_model_file, 'rb') as file_to_load:
                ctrack_model = pickle.load(file_to_load)
        self.ctrack_model = ctrack_model

        if feats2use is None:
            feats2use = ('centroid', 'area', 'minor_axis_length',
                          'major_axis_length', 'convex_area')
        self.feats2use = feats2use

        if ba_feats is None:
            ba_feats = ('centroid', 'area', 'minor_axis_length')
        self.ba_feats = ba_feats

        self.outfeats = list(
            regionprops_table(np.diag((1, 0)),
                              properties=self.feats2use).keys())
        self.a_ind = self.outfeats.index('area')
        self.ma_ind = self.outfeats.index('minor_axis_length')

        self.xtrafeats = ('distance', )

        if nstepsback is None:
            self.nstepsback = 2
        if ctrack_thresh is None:
            self.ctrack_thresh = 0.75
        if red_fun is None:
            self.red_fun = np.nanmax

    def calc_feats_from_mask(self, masks, feats2use=None):
        '''
        Calculate feature ndarray from ndarray of cell masks
        ---
        input
        :masks: ndarray (x_size, y_size, ncells), typically dtype bool

        output
        :n2darray: ndarray (ncells, nfeats)
        '''
        feats=[]
        if masks.sum():
            for i in range(masks.shape[2]):
                # Double conversion to prevent values from being floored to zero
                resized_mask = resize(masks[..., i], (100, 100)).astype(bool).astype(int)
                cell_feats = []
                for feat in regionprops_table(resized_mask,
                        properties=feats2use or self.feats2use).values():
                    cell_feats.append(feat[0])
                feats.append(cell_feats)

        return np.array(feats)

    def calc_feat_ndarray(self, prev_feats, new_feats):
        '''
        Calculate feature ndarray using two ndarrays of features.
        ---
        input
        :prev_feats: ndarray (ncells, nfeats) of timepoint 1
        :new_feats: ndarray (ncells, nfeats) of timepoint 2

        output
        :n3darray: ndarray (ncells_prev, ncells_new, nfeats) containing a
        cell-wise substraction of the features in the input ndarrays.
        '''
        if not (new_feats.any() and prev_feats.any()):
            return np.array([])

        nnew = len(new_feats)
        noutfeats = len(self.outfeats)

        n3darray = np.empty((len(prev_feats), nnew,
                             noutfeats + len(self.xtrafeats)))

        for i in range(noutfeats):
            n3darray[..., i] = np.subtract.outer(prev_feats[:, i],
                                                 new_feats[:, i])

        # Calculate extra features
        for i, feat in enumerate(self.xtrafeats, noutfeats):
            if feat == 'distance':
                # Assume that centroid-0 and centroid-1 are in the first two cols
                n3darray[..., i] = np.sqrt(
                    n3darray[..., 0]**2 + n3darray[..., 1]**2)

        return n3darray

    def predict_from_imgpair(self, img1, img2):
        ''' Generate predictions for two images. Useful to produce statistics.
        '''
        feats1 = self.calc_feats_from_mask(img1)
        feats2 = self.calc_feats_from_mask(img2)

        feats_3darray = self.calc_feat_ndarray(feats1, feats2)

        pred_matrix_bool = self.predict_proba_from_ndarray(feats_3darray, boolean=True)

        return pred_matrix_bool

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

        
    def predict_proba_from_ndarray(self, array_3d, boolean=False):

        if not array_3d.any():
            return np.array([])

        predict_fun = self.ctrack_model.predict if boolean else self.ctrack_model.predict_proba

        orig_shape = array_3d.shape[:2]

        # Flatten for predictions and then reshape back into matrix
        pred_list = np.array([
            val[1] for val in predict_fun(array_3d.reshape(
                -1, array_3d.shape[2]))
        ])
        pred_matrix = pred_list.reshape(orig_shape)

        return pred_matrix

    def get_new_lbls(self,
                     new_img,
                     prev_lbls,
                     prev_feats,
                     max_lbl,
                     new_feats=None):
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

        output
        :new_lbls: list of labels assigned to new timepoint
        :new_feats: list of ndarrays containing the updated features
        :new_max: updated max cell label assigned

        '''
        if new_feats is None:
            new_feats = self.calc_feats_from_mask(new_img)

        if new_feats.any():
            if prev_feats:
                counts = Counter([lbl for lbl_set in prev_lbls for lbl in lbl_set])
                lbls_order = list(counts.keys())
                probs = np.full(
                    (len(lbls_order), self.nstepsback, len(new_feats)), np.nan)
      
                for i, (lblset, prev_feat) in enumerate(zip(prev_lbls, prev_feats)):
                    if prev_feat.any():
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
                new_max = len(new_feats)
                new_lbls = [*range(1, new_max + 1)]

        else:
            return ([], [], max_lbl)
        return (new_lbls, new_feats, new_max)

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

        output

        :new_lbls: ndarray of newly assigned labels obtained, new cells as
        zero.
        '''
        if red_fun is None:
            red_fun = self.red_fun

        new_lbls = np.zeros(pred_3darray.shape[2], dtype=int)  

        pred_matrix = np.apply_along_axis(red_fun, 1, pred_3darray)


        # We remove any possible conflict by taking the maximum vals
        if pred_matrix.any():
            clean_mat = np.zeros(pred_matrix.shape)
            for j, i in enumerate(pred_matrix.argmax(0)):
                clean_mat[i, j] = pred_matrix[i, j]

            # assign available hits
            for i, j in enumerate(clean_mat.argmax(1)):
                if pred_matrix[i, j] > self.ctrack_thresh:
                    new_lbls[j] = prev_lbls[i]

        return new_lbls

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

        output
        :n2darray: 2d ndarray (ncells, ncells) giving probability that a cell
            (row) is a mother to another cell (column)
        '''

        if feats is None:
            feats = self.calc_feats_from_mask(masks, feats2use=self.ba_feats)
        elif len(feats) != len(masks):
            raise Exception('number of features must match number of masks')

        ncells = len(masks)

        p_bud_mat = np.zeros((ncells, ncells))
        p_budneck_mat = np.zeros((ncells, ncells))
        size_ratio_mat = np.zeros((ncells, ncells))
        adjacency_mat = np.zeros((ncells, ncells))


        for m in range(ncells):
            for d in range(ncells):
                if m == d:
                    continue

                p_bud_mat[m, d] = np.mean(p_bud[masks[d].astype('bool')])

                size_ratio_mat[m, d] = feats[m, self.a_ind] / feats[d,
                                                                    self.a_ind]

                # Draw rectangle
                r_points = self.get_rpoints(feats, d, m)
                rr, cc = polygon(r_points[0, :], r_points[1, :],
                                 p_budneck.shape)
                r_im = np.zeros(p_budneck.shape, dtype='bool')
                r_im[rr, cc] = True

                # Calculate the mean of bud neck probabilities greater than some threshold
                pbn = p_budneck[r_im].flatten()

                pbn = pbn[pbn > 0.2]
                p_budneck_mat[m, d] = np.mean(pbn) if len(pbn) > 0 else 0

                # Adjacency is the proportion of the joining rectangle that overlaps the mother daughter union
                adjacency_mat[m, d] = np.sum((masks[m]
                                              | masks[d]) & r_im) / np.sum(
                                                  r_im)

        mb_stats = np.hstack([
            s.flatten()[:, np.newaxis]
            for s in (p_bud_mat, size_ratio_mat, p_budneck_mat, adjacency_mat)
        ])

        good_stats = ~np.isnan(mb_stats).any(axis=1)
        ba_probs = np.nan * np.ones(ncells**2)
        if good_stats.any():
            ba_probs[good_stats] = self.ba_model.predict_proba(
                mb_stats[good_stats, :])[:, 1]
        ba_probs = ba_probs.reshape((ncells, ) * 2)

        return ba_probs

    def get_rpoints(self, feats, d, m):
        '''
        Draw a rectangle in the budneck of cells
        ---
        input
        feats: 2d ndarray (ncells, nfeats)

        output:
        r_points: 2d ndarray (2,4) with the coordinates of the rectangle corner

        '''
        # Draw connecting rectangle
        m_centre = feats[m, :2]
        d_centre = feats[d, :2]
        r_width = np.max((2, feats[d, self.ma_ind] * 0.25))
        r_hvec = d_centre - m_centre
        r_wvec = np.matmul(np.array([[0, -1], [1, 0]]), r_hvec)
        r_wvec = r_width * r_wvec / np.linalg.norm(r_wvec)
        r_points = np.zeros((2, 4))
        r_points[:, 0] = m_centre - 0.5 * r_wvec
        r_points[:, 1] = r_points[:, 0] + r_hvec
        r_points[:, 2] = r_points[:, 1] + r_wvec
        r_points[:, 3] = r_points[:, 2] - r_hvec

        return r_points

    def step_trackers(self,
                      masks,
                      p_budneck,
                      p_bud,
                      state=None,
                      assign_mothers=False,
                      return_baprobs=False):
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
            output
        :return_baprobs: whether to include bud assignment probability matrix
            in the returned output

        returns a dict consisting of

        :cell_label: list of int, the tracked global ID for each cell mask
        :state: the updated state to be used in a subsequent step
        :mother_assign: (optional) list of int, specifying the assigned mother
            for each cell
        :p_bud_assign: (optional) matrix (list of lists of floats), bud assignment
            probability matrix from `calc_mother_bud_stats`
        '''

        if state is None:
            state = {}

        max_lbl = state.get('max_lbl', 0)
        cell_lbls = state.get('cell_lbls', [])
        prev_feats = state.get('prev_feats', [])

        # Get features for cells at this time point
        feats = self.calc_feats_from_mask(masks)

        lastn_lbls = cell_lbls[-self.nstepsback:]
        lastn_feats = prev_feats[-self.nstepsback:]

        new_lbls, _, max_lbl = self.get_new_lbls(masks, lastn_lbls,
                                                 lastn_feats, max_lbl, feats)

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
                state[k] = np.pad(v, (0, l - max_lbl + 32), 'constant')

        lifetime = state.get('lifetime', init['lifetime'])
        p_is_mother = state.get('p_is_mother', init['p_is_mother'])
        p_was_bud = state.get('p_was_bud', init['p_was_bud'])
        ba_cum = state.get('ba_cum', init['ba_cum'])

        # Update lineage state variables
        if len(masks) > 0 and new_lbls:
            ba_probs = self.calc_mother_bud_stats(p_budneck, p_bud, masks,
                                                  feats)
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
                # Cell must have been a bud and been present for at least 2 tps
                isbud = (p_was_bud[0:max_lbl] > 0.5) & (lifetime[0:max_lbl] >
                                                        2)
                ma[~isbud] = 0  # 0 indicates no assignment (lbls indexed from 1)
            else:
                ma = np.zeros(0)

            output['mother_assign'] = ma.tolist()

        if return_baprobs:
            output['p_bud_assign'] = ba_probs.tolist()

        return output

# def decay(array, c=0.5):
#     '''Calculates the average using a decay function p/(a*t) where
#     'p' is the probability of two cells being the same, 't' the timestep
#     and 'a' a chosen coefficient
#     :array: List of probabilities
#     :c: Scaling coefficient
#     '''
#     result = 0
#     for t, p in enumerate(array):
#         if not np.isnan(p):
#             result += p / (t * c)
#     return result

