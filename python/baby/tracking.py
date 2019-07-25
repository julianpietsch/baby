import numpy as np
from scipy.ndimage import binary_fill_holes
from skimage.measure import regionprops
from skimage.draw import polygon


def get_mother_bud_stats(cnn_output, segs, masks=None, rprops=None):
    _, _, _, _, p_budneck, p_bud = cnn_output

    if masks is None:
        masks = [binary_fill_holes(s) for s in segs]
    if rprops is None:
        rprops = [regionprops(m.astype('int'), coordinates='rc')[0] for m in masks]

    ncells = len(rprops)

    p_bud_mat = np.zeros((ncells, ncells))
    p_budneck_mat = np.zeros((ncells, ncells))
    size_ratio_mat = np.zeros((ncells, ncells))
    adjacency_mat = np.zeros((ncells, ncells))

    for m in range(ncells):
        for d in range(ncells):
            if m == d:
                continue

            p_bud_mat[m, d] = np.mean(p_bud[masks[d]])
            size_ratio_mat[m, d] = rprops[m].area / rprops[d].area

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
            p_budneck_mat[m, d] = np.mean(pbn) if len(pbn) > 0 else 0

            # Adjacency is the proportion of the joining rectangle that overlaps the mother daughter union
            adjacency_mat[m, d] = np.sum((masks[m] | masks[d]) & r_im) / np.sum(r_im)

    return p_bud_mat, size_ratio_mat, p_budneck_mat, adjacency_mat

