''' Tracker class to perform cell tracking inside baby.
'''
from collections import Counter
import pickle
import numpy as np
from skimage.measure import regionprops_table 
from skimage.draw import polygon
import os

models_path = os.path.join(os.path.dirname(__file__), 'models')

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
                 xtrafeats=None,
                 ctrack_thresh=None,
                 px_size=None,
                 nstepsback=None,
                 red_fun=None):

        if ba_model is None:
            ba_model_file = os.path.join(models_path, 'baby_randomforest_20190906.pkl')
            with open(ba_model_file, 'rb') as file_to_load:
                ba_model = pickle.load(file_to_load)
        self.ba_model = ba_model

        if ctrack_model is None:
            ctrack_model_file = os.path.join(models_path,
                                      'ctrack_randomforest_20201012.pkl')
            with open(ctrack_model_file, 'rb') as file_to_load:
                ctrack_model = pickle.load(file_to_load)
        self.ctrack_model = ctrack_model

        if feats2use is None:
            feats2use, xtrafeats = self.get_feats2use()
        self.feats2use = feats2use
        self.xtrafeats = xtrafeats

        self.ba_feat_names = ('p_bud', 'size_ratio', 'p_budneck',
                'budneck_ratio', 'adjacency')

        self.outfeats = list(
            regionprops_table(np.diag((1, 0)),
                              properties=self.feats2use).keys())
        if 'area' in self.feats2use:
            self.a_ind = self.outfeats.index('area')
        if 'minor_axis_length' in self.feats2use:
            self.ma_ind = self.outfeats.index('minor_axis_length')

        if px_size is None:
            px_size = 0.263
        self.px_size = px_size

        if nstepsback is None:
            nstepsback = 5
        self.nstepsback = nstepsback

        if ctrack_thresh is None:
            ctrack_thresh = 0.7
        self.ctrack_thresh = ctrack_thresh

        if red_fun is None:
            red_fun = np.nanmax
        self.red_fun = red_fun

    def calc_feats_from_mask(self, masks, feats2use=None, norm=True,
                             px_size=None):
        '''
        Calculate feature ndarray from ndarray of cell masks
        ---
        input

        :masks: ndarray (ncells, x_size, y_size), typically dtype bool
        :feats2use: list of strings with the feature properties to extract.
            If it is None it uses the ones set in self.feats2use.
        :norm: bool, if True normalises mask to a defined px_size.
        :px_size: float, used to normalise the images.

        returns

        (ncells, nfeats) ndarray of features for input masks
        '''
        if px_size is None:
            px_size = self.px_size

        feats=[]
        if masks.sum():
            for mask in masks:
                cell_feats = []
                for feat in regionprops_table(mask.astype(int),
                        properties=feats2use or self.feats2use).values():
                    cell_feats.append(feat[0])

                if norm:
                    cell_feats = self.norm_feats(cell_feats, px_size)# or self.px_size)

                feats.append(cell_feats)

        return np.array(feats)

    def norm_feats(self, feats, px_size):
        '''
        input

        :feats: list of float, extracted features obtained from regionprops_table
        :px_size: Value used to normalise the images.

        returns
        normalised list of feature values
        '''
        area = px_size**2

        if 'centroid' in self.feats2use:
            feats[self.outfeats.index('centroid-0')] /= px_size
            feats[self.outfeats.index('centroid-1')] /= px_size
        
        if 'area' in self.feats2use:
            feats[self.outfeats.index('area')] /= area
            
        if 'convex_area' in self.feats2use:
            feats[self.outfeats.index('convex_area')] /= area
            
        if 'bbox_area' in self.feats2use:
            feats[self.outfeats.index('bbox_area')] /= area

        if 'minor_axis_length' in self.feats2use:
            feats[self.outfeats.index('minor_axis_length')] /= px_size

        if 'major_axis_length' in self.feats2use:
            feats[self.outfeats.index('major_axis_length')] /= px_size

        return feats
         
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

        input

        :img1: (nxm) ndarray containing a single cell
        :img2: (nxm) ndarray containing a single cell

        returns

        tracking prediction matrix resultant of input images' comparison
        '''
        feats1 = self.calc_feats_from_mask(img1)
        feats2 = self.calc_feats_from_mask(img2)

        feats_3darray = self.calc_feat_ndarray(feats1, feats2)

        pred_matrix_bool = self.predict_proba_from_ndarray(feats_3darray, boolean=True)

        return pred_matrix_bool

    def get_truth_matrix_from_pair(self, pair):
        '''
        Requires self.meta

        input
        :pair: tuple of size 4 (experimentID, position, trap (tp1, tp2))

        returns

       :truth_mat: boolean ndarray of shape (ncells(tp1) x ncells(tp2)
            links cells in tp1 to cells in tp2
        '''
        
        clabs1 = self.meta.loc[pair[:3] + (pair[3][0], ), 'cellLabels']
        clabs2 = self.meta.loc[pair[:3] + (pair[3][1], ), 'cellLabels']

        truth_mat = gen_boolmat_from_clabs(clabs1, clabs2)

        return truth_mat

        
    def predict_proba_from_ndarray(self, array_3d, boolean=False):
        '''

        input

        :array_3d: (ncells_tp1, ncells_tp2, out_feats) ndarray
        :boolean: bool, if False returns probability, if True returns prediction

        returns

        (ncells_tp1, ncells_tp2) ndarray with probabilities or prediction
            of cell identities depending on "boolean" arg.

        '''

        if array_3d.size == 0:
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

        returns

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

        NB: ASSUMES FEATS HAVE ALREADY BEEN NORMALISED!

        returns

        :n2darray: 2d ndarray (ncells x ncells, n_ba_feat_names) specifying,
            for each pair of cells in the masks array, the features used for
            mother-bud pair prediction (as per 'ba_feat_names')
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
                p_budneck, p_bud, masks, feats=None)

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
            ba_probs = self.predict_mother_bud(p_budneck, p_bud, masks,
                                                  feats)
            lblinds = np.array(new_lbls) - 1  # new_lbls are indexed from 1
            lifetime[lblinds] += 1
            p_is_mother[lblinds] = np.maximum(p_is_mother[lblinds],
                                              np.nansum(ba_probs, 1))
            p_was_bud[lblinds] = np.maximum(p_was_bud[lblinds],
                                            np.nanmax(ba_probs, 0))
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

    def get_feats2use(self):
        '''
        Return feats to be used from loaded ctrack_model
        '''

        model_nfeats = len(self.ctrack_model.feature_importances_)
        return(switch_case_nfeats(model_nfeats))

        
def switch_case_nfeats(nfeats):
    '''
    Convenience TEMPORAL function to determine whether to use distance/location
    as a feature for tracking or not (nfeats=5 for no distance, 7 for distance)
    input
    number of feats

    returns
    list of main and extra feats based on the number of feats
    '''
    main_feats = {5 : [(
            'area', 'minor_axis_length', 'major_axis_length', 'convex_area',
        'bbox_area'), ()],
                  
            # Including centroid
            7 : [('centroid', 'area', 'minor_axis_length',
                          'major_axis_length', 'convex_area'), ('distance',)]}

    return(main_feats.get(nfeats, []))

