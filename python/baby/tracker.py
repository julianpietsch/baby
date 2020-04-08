'''
Tracker class to perform cell tracking inside baby.

How to run:
TODO Add this

'''
from collections import Counter
import pickle
import numpy as np
from imageio import imread
from skimage.measure import regionprops_table
from skimage.draw import polygon
from scipy.ndimage.morphology import binary_fill_holes


class Tracker:
    '''
    Class used to manage cell tracking.

    Initialization parameters:

    :model: sklearn.ensemble.RandomForestClassifier object
    :nstepsback: int Number of timepoints to go back
    :ctrac_thresh: float Cut-off value to assume a cell is not new
    '''
    def __init__(self,
                 ctrack_model=None,
                 ba_model=None,
                 nstepsback=None,
                 ctrack_thresh=None,
                 ba_thresh=None):
        '''
        '''
        self.feats2use = ('centroid', 'area', 'minor_axis_length',
                          'major_axis_length', 'convex_area')
        # self.mb_feats = ('centroid', 'area', 'minor_axis_length')

        self.outfeats = list(
            regionprops_table(np.diag((1, 0)),
                              properties=self.feats2use).keys())
        self.a_ind = self.outfeats.index('area')
        self.ma_ind = self.outfeats.index('minor_axis_length')

        self.xtrafeats = ('distance', )

        if ba_model is None:
            with open('models/test.pkl', 'rb') as file_to_load:
                self.ba_model = pickle.load(file_to_load)
        if ctrack_model is None:
            with open('models/test.pkl', 'rb') as file_to_load:
                self.ctrack_model = pickle.load(file_to_load)
        if nstepsback is None:
            self.nstepsback = 2
        if ctrack_thresh is None:
            self.ctrack_thresh = 0.5
        if ba_thresh is None:
            self.ba_thresh = 0.5

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
        nnew = len(new_feats)

        n3darray = np.empty(
            (len(prev_feats), nnew, self.outfeats + len(self.xtrafeats)))

        for i in range(self.outfeats):
            n3darray[..., i] = np.subtract.outer(prev_feats[:, i],
                                                 new_feats[:, i])

        # Calculate extra features
        for i, feat in enumerate(self.xtrafeats, self.outfeats):
            if feat == 'distance':
                # Assume that centroid-0 and centroid-1 are in the first two cols
                n3darray[..., i] = np.sqrt(n3darray[..., 0]**2 +
                                           n3darray[..., 1]**2)

        return n3darray

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
        :new_max: updated max cell albel assigned
        :new_lbl: list of labels assigned to new timepoint
        :new_feats: list of ndarrays containing the updated features

        '''
        if new_feats is None:
            new_feats = np.array([[
                feat[0] for feat in regionprops_table(
                    new_img[..., i], properties=self.feats2use).values()
            ] for i in range(new_img.shape[2])])

        if len(prev_feats):
            pred_lbls = np.zeros((len(prev_feats), len(new_feats)), dtype=int)

            for i, prev_feat in enumerate(prev_feats):
                feats_3darray = self.calc_feat_ndarray(prev_feat, new_feats)
                orig_shape = feats_3darray.shape[:2]

                # Flatten for predictions and then reshape back into matrix
                pred_list = np.array([
                    val[1] for val in self.ctrack_model.predict_proba(
                        feats_3darray.reshape(-1, feats_3darray.shape[2]))
                ])
                pred_matrix = pred_list.reshape(orig_shape)
                pred_lbls[i, :] = self.assign_lbls(pred_matrix, prev_lbls[i],
                                                   self.ctrack_thresh)

            new_lbls, new_max = self.count_votes(pred_lbls, max_lbl)

        else:
            new_max = len(new_feats)
            new_lbls = [*range(1, new_max + 1)]

        return (new_lbls, new_feats, new_max)

    def assign_lbls(self, pred_matrix, prev_lbls, thresh, norm=False):
        '''Assign labels using a prediction matrix of nxm where n is the number
        of cells in the previous image and m in the new image. It assigns the
        number zero if it doesn't find the cell.
        :pred_matrix: Probability n x m matrix obtained as an output of rforest
        :prev_labels: List of cell labels for previous timepoint to be compared.
        ---
        input

        :pred_matrix: Matrix with probabilities of the corresponding two cells
        being the same.
        :prev_lbls: List of ints representing the cell labels in the previous tp.

        output

        :new_lbls: Newly assigned labels obtained, new cells as zero.
        '''

        # We remove any possible conflict by taking the maximum vals
        clean_mat = np.zeros(pred_matrix.shape)
        for j, i in enumerate(pred_matrix.argmax(0)):
            clean_mat[i, j] = pred_matrix[i, j]

        # Normalise (useful for aggregating probabilities)
        if norm:
            clean_mat = clean_mat / (clean_mat + 1).max(1)[:, np.newaxis]

            # assign available hits
            new_lbls = np.repeat(0, pred_matrix.shape[1])
            for i, j in enumerate(clean_mat.argmax(1)):
                if pred_matrix[i, j] > thresh:
                    new_lbls[j] = prev_lbls[i]

        return new_lbls

    def count_votes(self, pred_lbls, max_lbl):
        '''
        Counts the individual cell assignment in the new image.
        If there is only one, assigns the most common value.
        ----
        input
        :pred_lbls: ndarray(nvals*nprevtps). Cell assignment prediction for
            each previous timepoint.
        :max_lbl: int to indicate the current max label

        output
        :decision: list of labels assigned for new cells after comparing multiple tps.
        :new_max: updated maximum cell label
        '''
        counts = np.apply_along_axis(Counter, 0, pred_lbls)
        decision = np.zeros(pred_lbls.shape[1], dtype=int)
        # Count the amount of times a cell label was selected

        for i, count in enumerate(counts):
            common = np.array(count.most_common())
            if len(common) > 1:
                common = common[common.all(axis=1)]
                common = common[common[:, 1] == common[0, 1]]

            # TODO FUTURE: Use probability or timepoint to pick cell when tied
            decision[i] = np.min(common[:, 0])

        # Give new numbers to unassigned cells
        new_cells = decision == 0
        new_max = max_lbl + np.sum(new_cells)
        decision[new_cells] = [*range(max_lbl + 1, new_max + 1)]

        return (decision.tolist(), new_max)

    ## Handler function

    def get_tseries_lbls(self, img_list):
        '''
        TODO BABY: modify this function to add new images as they appear
        Handler function to calculate the timepoints of a list of images
        ---
        input
        :img_list: list of ndarrays, one for each timepoint

        output
        :cell_lbls: list of lists of ints the corresponding cell labels
        '''
        cell_lbls = []
        feats = []
        max_lbls = 0

        for img in img_list:
            new_lbls, new_feats, max_lbls = self.get_new_lbls(
                img, cell_lbls[-self.nstepsback:], feats[-self.nstepsback:],
                max_lbls)

            cell_lbls.append(new_lbls)
            feats.append(new_feats)

            # Where to put this line? Or also yield new_feats?
            self.calc_mother_bud_stats(pred,
                                       flattener,
                                       img_list,
                                       feats=new_feats)

            yield cell_lbls

    def get_cells(self, filename, cell_id):
        '''
        Read a filename and cell id (only to get the number of cells)
        and return the ndarray containing the cells in the z-axis
        ---
        input
        :filename: str indication the png location
        :cell_id: list of ints, where the ints are cell labels

        output
        ndarray (size_x, size_y, ncells). ndarray containing the mask for cells
        in the z-axis
        '''

        ncells = len(aslist(cell_id))
        img = imread(filename)
        cell_masks = np.hsplit(binary_fill_holes(img).astype(int), ncells)

        return np.dstack(cell_masks)

    ### Assign mother-
    def calc_mother_bud_stats(self, pred, flattener, masks, feats=None):
        '''
        ---
        input
        :pred: output from cnn
        :flattener: TODO Ask Julian for help here
        :mask: 3d ndarray (size_x, size_y, ncells)
        :feats: ndarray (ncells, nfeats)
        '''

        if feats is None:
            feats = [
                regionprops_table(m.astype('int'), properties=self.mb.feats)[0]
                for m in masks
            ]

        ncells = len(masks)

        # Assign mothers to daughters (masks and rprops will be updated)
        # mb_stats = get_mother_bud_stats(pred,
        #                                 self.flattener,
        #                                 masks=masks)

        tnames = flattener.names()
        p_budneck = pred[tnames.index('bud_neck')]
        p_bud = pred[tnames.index('sml_fill')]

        ncells = len(feats)

        p_bud_mat, p_budneck_mat, size_ratio_mat, adjacency_mat = np.repeat(
            np.zeros((ncells, ncells)), 4)

        for m in range(ncells):
            for d in range(ncells):
                if m == d:
                    continue

                p_bud_mat[m, d] = np.mean(p_bud[masks[d]])
                size_ratio_mat[m,
                               d] = feats[m, self.a_ind] / feats[d, self.a_ind]

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
                adjacency_mat[m,
                              d] = np.sum((masks[m]
                                           | masks[d]) & r_im) / np.sum(r_im)

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

    # Utility functions

    def coord_trackers(self, masks, pred, flattener, ba_cum, max_lbls,
                       cell_lbls, prev_feats):
        '''
        Calculate features and track cells and budassignments
        input
        :masks: 3d int ndarray (size_x, size_y, ncells) containing cell masks
        pred
        flattener
        :max_lbl: int indicating the last assigned cell label
        :prev_feats: list of ndarrays of size (ncells x nfeatures)
        containing the features of previous timepoints
        :cell_lbls: list of list of ints corresponding to the cell labels in
            the previous timepoints

        returns
        :feats: 2d float ndarray (ncells, nfeatures) containing the feature
        values for each cell
        :new_lbls: list of int, contains information on the global id of the cell
        :ba_props: 2d ndarray (ncells, ncells) with probability of cells being
        daughter of one another
        '''

        # Get features
        feats = np.array([[
            feat[0]
            for feat in regionprops_table(masks[..., i],
                                          properties=self.feats2use).values()
        ] for i in range(masks.shape[2])])

        new_lbls, _, max_lbls = self.get_new_lbls(
            masks, cell_lbls[-self.nstepsback:], prev_feats[-self.nstepsback:],
            max_lbls)

        ba_pred_matrix = self.calc_mother_bud_stats(pred, flattener, masks,
                                                    feats)

        # if necessary, increase the matrix size
        if max_lbls > len(ba_cum):
            new_size = (np.floor(max_lbls / 32).astype(int) + 1) * 32
            ba_cum = np.pad(ba_cum, (0, new_size), 'constant')

        # Subindex matrix with cells present in this image
        nlbls = len(new_lbls)
        new_lbls_ind = (np.repeat(new_lbls, nlbls).reshape(-1, nlbls),
                        np.tile(new_lbls, nlbls).reshape(-1, nlbls))
        ba_cum[new_lbls_ind] += ba_pred_matrix
        newmothers = self.assign_lbls(ba_pred_matrix[new_lbls_ind],
                                      self.ctrack_thresh)

        return (feats, new_lbls, max_lbls, newmothers, ba_cum)
