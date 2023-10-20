# If you publish results that make use of this software or the Birth Annotator
# for Budding Yeast algorithm, please cite:
# Pietsch, J.M.J., Muñoz, A.F., Adjavon, D.-Y.A., Farquhar, I., Clark, I.B.N.,
# and Swain, P.S. (2023). Determining growth rates from bright-field images of
# budding cells through identifying overlaps. eLife. 12:e79812.
# https://doi.org/10.7554/eLife.79812
# 
# 
# The MIT License (MIT)
# 
# Copyright (c) Julian Pietsch, Alán Muñoz and Diane Adjavon 2023
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
from itertools import chain
from typing import NamedTuple, Tuple
import inspect
from scipy import interpolate
from scipy.optimize import least_squares
from scipy.ndimage import (minimum_filter, binary_dilation, binary_erosion,
                           binary_closing, binary_opening, binary_fill_holes)
from skimage import filters
from skimage.measure import label, regionprops
from skimage.morphology import diamond, erosion, dilation

from .errors import BadParam


############################
###  UTILITY FUNCTIONS   ###
############################


squareconn = diamond(1)  # 3x3 filter for 1-connected patches
fullconn = np.ones((3, 3), dtype='uint8')


def binary_edge(imfill, footprint=fullconn):
    """Get square-connected edges from filled image:"""
    return minimum_filter(imfill, footprint=footprint) != imfill


def iterative_erosion(img, iterations=1, **kwargs):
    if iterations < 1:
        return img

    for i in range(iterations):
        img = erosion(img, **kwargs)
    return img


def iterative_dilation(img, iterations=1, **kwargs):
    if iterations is None:
        return img
    for _ in range(iterations):
        img = dilation(img, **kwargs)
    return img


def single_region_prop(mask):
    return regionprops(mask.astype(np.uint8))[0]


def get_edge_scores(outlines, p_edge):
    """Return probability scores for a list of outlines

    NB: BREAKING CHANGE. From July 2022 this has been corrected to take the
    mean only over edge pixels rather than the entire image. Segmentation
    parameters will need to be reoptimised.

    :param outlines: list of outline images (2D bool)
    :param p_edge: edge probability image (2D float with values in [0, 1])

    :return: list of edge probability scores (each in [0, 1])
    """
    return [
        p_edge[binary_dilation(o, iterations=2)].mean() for o in outlines
    ]


def mask_iou(a, b):
    """Intersection over union (IoU) between boolean masks"""
    return np.sum(a & b) / np.sum(a | b)


def mask_containment(a, b):
    """Max of intersection over a or over b"""
    return np.max(np.sum(a & b) / np.array([np.sum(a), np.sum(b)]))


def bbox_overlaps(bboxA, bboxB):
    """Returns True if the bounding boxes are overlapping.

    Args:
        bboxA (tuple): A bounding box ``(min_row, min_col, max_row, max_col)``
            as defined by :py:func:`skimage.measure.regionprops`.
        bboxB (tuple): A bounding box ``(min_row, min_col, max_row, max_col)``
            as defined by :py:func:`skimage.measure.regionprops`.
    """
    lrA, lcA, urA, ucA = regA.bbox
    lrB, lcB, urB, ucB = regB.bbox
    rA = np.array([lrA, urA])
    cA = np.array([lcA, ucA])
    return ((not ((rA > urB).all() or (rA < lrB).all())) and
            (not ((cA > ucB).all() or (cA < lcB).all())))


def region_iou(regA, regB):
    """Efficiently computes the IoU between two RegionProperties."""
    if bbox_overlaps(regA, regB):
        bb_lr, bb_lc, _, _ = np.stack((regA.bbox, regB.bbox)).min(axis=0)
        _, _, bb_ur, bb_uc = np.stack((regA.bbox, regB.bbox)).max(axis=0)
        bboxA = np.zeros((bb_ur - bb_lr, bb_uc - bb_lc), dtype='bool')
        bboxB = bboxA.copy()
        bb = regA.bbox
        bboxA[bb[0] - bb_lr:bb[2] - bb_lr,
              bb[1] - bb_lc:bb[3] - bb_lc] = regA.image
        bb = regB.bbox
        bboxB[bb[0] - bb_lr:bb[2] - bb_lr,
              bb[1] - bb_lc:bb[3] - bb_lc] = regB.image
        return np.sum(bboxA & bboxB) / np.sum(bboxA | bboxB)
    else:
        return 0.


def limit_to_bbox(imglist, bbox, padding=10):
    imH, imW = imglist[0].shape[:2]
    assert all([i.shape[:2] == (imH, imW) for i in imglist])
    rmin, cmin, rmax, cmax = bbox
    rmin = np.maximum(rmin - padding, 0)
    cmin = np.maximum(cmin - padding, 0)
    rmax = np.minimum(rmax + padding, imH)
    cmax = np.minimum(cmax + padding, imW)
    return (i[rmin:rmax, cmin:cmax] for i in imglist), (rmin, cmin, imH, imW)


def restore_from_bbox(imglist, bbunmap):
    bbH, bbW = imglist[0].shape[:2]
    assert all([i.shape[:2] == (bbH, bbW) for i in imglist])
    rmin, cmin, imH, imW = bbunmap
    for img in imglist:
        restored = np.zeros((imH, imW) + img.shape[2:], dtype=img.dtype)
        restored[rmin:rmin + bbH, cmin:cmin + bbW] = img
        yield restored


def threshold_segmentation(p_int,
                           interior_threshold=0.5,
                           connectivity=None,
                           nclosing=0,
                           nopening=0,
                           ndilate=0,
                           return_area=False):
    """Generic threshold-based segmentation routine for eroded interiors.

    Finds cell mask instances from a thresholded probability image of cell
    interiors (possibly trained on eroded targets). Each candidate mask is
    then independently cleaned by binary closing and then binary opening.
    Finally, masks are dilated to restore to an original size (i.e., to undo
    the level of erosion applied to training target images).

    Args:
        p_int: An ndarray specifying cell interior probability for each pixel.
        interior_threshold: Threshold to apply to the probability image.
        connectivity: Connectivity as defined by `skimage.measure.label`.
        nclosing: Number of iterations in closing operation; if 0 then binary
            closing is not applied.
        nopening: Number of iterations in opening operation; if 0 then binary
            opening is not applied.
        ndilate: Number of iterations of binary dilation to apply at end.
        return_area: If True, then function yields (mask, area) tuples.

    Yields:
        Arrays with the same shape as `p_int` for each identified cell mask. If
        `return_area=True`, then function yields (mask, area) tuples where the
        area is the number of True pixels in the mask.
    """

    lbl, nmasks = label(p_int > interior_threshold,
                        return_num=True,
                        connectivity=connectivity)
    for l in range(nmasks):
        mask = lbl == l + 1
        if nclosing > 0:
            mask = binary_closing(mask, iterations=nclosing)
        if nopening > 0:
            mask = binary_opening(mask, iterations=nopening)
        if ndilate > 0:
            mask = binary_dilation(mask, iterations=ndilate)

        if return_area:
            yield mask, mask.sum()
        else:
            yield mask


def ordered_edge_points(mask, edge=None, border_rect=None):
    """Returns edge coordinates ordered around the mask perimeter.

    Uses the tangent to the filled mask image to step pixel-wise around an
    edge image and terminate when within a pixel length of the first pixel, or
    when steps would be larger than 2 pixels to continue.  Note that this
    function does not return all edge points, just those pixels closest to a 1
    pixel step along the direction of the tangent.

    Args:
        mask (ndarray): A 2D bitmask for a single cell.
        edge (None or ndarray): To save recomputation, optionally provide the
            edge image obtained from :py:func:`binary_edge`.
        border_rect (None or ndarray): To save recomputation, optionally
            provide an image of same shape as the mask with ``False`` values
            except for a 1-pixel border of ``True`` values.

    Returns:
        An ndarray of shape ``(N_edge_pixels, 2)`` giving row/column
        coordinates for the ordered edge pixels.
    """
    if edge is None:
        edge = binary_edge(mask)

    if border_rect is None:
        border_rect = np.pad(
            np.zeros(tuple(x - 2 for x in mask.shape), dtype='bool'),
            pad_width=1, mode='constant', constant_values=True)
    
    # Need to ensure an edge if mask is adjacent to the border
    edge = binary_edge(mask) | (border_rect & mask)

    X, Y = np.nonzero(edge)
    edgepts = np.c_[X,Y]

    # Use Sobel filter on Gaussian-blurred mask to estimate tangent along edge
    mask_blur = filters.gaussian(mask, 1, mode='constant')
    hgrad = filters.sobel_h(mask_blur)
    vgrad = filters.sobel_v(mask_blur)
    edge_hgrad = hgrad[edge]
    edge_vgrad = vgrad[edge]
    gradvec = np.c_[edge_hgrad, edge_vgrad]
    normgradvec = gradvec / np.sqrt(np.sum(np.square(gradvec), axis=1)[:, None])
    rot90 = np.array([[0, 1], [-1, 0]])
    tngvec = np.matmul(normgradvec, rot90)

    # Loop over edge points in direction of tangent to get ordered list
    unvisited = np.ones(edgepts.shape[0], dtype='bool')
    i = 0
    start = edgepts[i]
    ptorder = []
    for o in range(unvisited.size):
        unvisited[i] = False
        nextpt = edgepts[[i]] + tngvec[[i]]
        if np.sum(np.square(nextpt - start)) <= 2 and i > 0:
            # We have returned to the first pixel
            break
        nextpt_sqdist = np.sum(np.square(edgepts[unvisited]-nextpt), axis=1)
        minInd = np.argmin(nextpt_sqdist)
        if nextpt_sqdist[minInd] > 4:
            # We are jumping further than just neighbouring pixels
            break
        i = np.flatnonzero(unvisited)[minInd]
        ptorder.append(i)
        
    return edgepts[ptorder]


############################
### RADIAL KNOT SPLINES  ###
############################


def rc_to_radial(rr_cc, centre):
    """Converts row-column coordinates to radial coordinates.

    Can be used to directly convert (row, column) output from the
    `numpy.nonzero` function applied to a binary edge image
    ``rc_to_radial(np.nonzero(edge_image), (0, 0))``

    Args:
        rr_cc: A tuple ``(rr, cc)`` of ndarrays specifying row and column
            indices to be converted. ``rr`` and ``cc`` should have the same
            shape.
        centre: A tuple (rr, cc) specifying the origin of the radial
            coordinate system.

    Returns:
        A tuple ``(rho, phi)`` of ndarrays of same shape as ``rr`` and ``cc``
        giving the radii ``rho`` and angles ``phi`` in the radial coordinate
        system.
    """
    rr, cc = rr_cc
    rloc, cloc = centre
    rr = rr - rloc
    cc = cc - cloc
    return np.sqrt(np.square(rr) + np.square(cc)), np.arctan2(cc, rr)


def eval_radial_spline(x, rho, phi):
    """Evaluates a radial spline with knots defined in radial coordinates.

    The spline is periodic across the boundary.

    Args:
        x: An ndarray of angles at which to evaluate the spline.
        rho: An ndarray of radii in[0, Inf) for each knot of the spline.
        phi: An ndarray of angles in [-pi, pi) for each knot of the spline.

    Returns:
        An ndarray with same shape as x giving corresponding interpolated
        radii.
    """

    # Angles need to be in increasing order for expected behaviour of phi as a
    # parametric variable
    order = np.argsort(phi)
    rho = rho[order]
    phi = phi[order]
    offset = phi[0]

    # Make the boundaries periodic
    rho = np.r_[rho, rho[0]]
    phi = np.r_[phi - offset, 2 * np.pi]

    tck = interpolate.splrep(phi, rho, per=True)
    return interpolate.splev(np.mod(x - offset, 2 * np.pi), tck)


def eval_cartesian_spline(x, rho, phi):
    """Evaluates a cartesian spline with knots defined in radial coordinates.

    The spline interpolates between knots in order of appearance in the array
    and is periodic over the end-points. Notably, the knots do not necessarily
    need to be in order of increasing phi (though this will likely be the most
    common case).

    Args:
        x: An ndarray of 'angles' at which to evaluate the spline. These are
            not true radial angles as used for `phi` when defining the knots,
            but are rather parametric locations defined over [0, 2*pi).
        rho: An ndarray of radii in[0, Inf) for each knot of the spline.
        phi: An ndarray of angles in [-pi, pi) for each knot of the spline.

    Returns:
        A tuple (Sx, Sy) of ndarrays with same shape as x, giving the
        interpolated x and y coordinates corresponding to each parametric
        location in x.
    """

    # Make the boundaries periodic
    rho = np.r_[rho, rho[0]]
    phi = np.r_[phi, phi[0]]

    # TODO check whether the spline behaves better if the lengths between each
    # knot are used to scale relative distance in the parametric variable

    # Define splines according to a parametric variable over [0,2*pi)
    t = np.linspace(0, 2*np.pi, rho.size)
    Xtck = interpolate.splrep(t, rho*np.cos(phi), per=True)
    Ytck = interpolate.splrep(t, rho*np.sin(phi), per=True)

    # Evaluate over a modded x
    Sx = interpolate.splev(np.mod(x, 2 * np.pi), Xtck)
    Sy = interpolate.splev(np.mod(x, 2 * np.pi), Ytck)
    return Sx, Sy


def draw_radial(rho, phi, centre, shape, cartesian_spline=False):
    """Renders a spline defined in radial coordinates as an image.

    By default, interpolates spline in radial space.

    Args:
        rho: An ndarray of radii defining knot locations.
        phi: An ndarray of angles defining knot locations.
        centre: A sequence of length 2 specifying the row and column defining
            the origin of the radial coordinate system.
        shape: A sequence of length 2 specifying the height and width of the
            image to be rendered to.
        cartesian_spline: If True, then interpolate spline in cartesian space.

    Returns:
        An ndarray with the specified shape and of dtype bool containing the
        rendered spline.
    """

    mr, mc = shape
    im = np.zeros(shape, dtype='bool')

    # Estimate required sampling density from lengths of piecewise linear segments
    rho_loop = np.r_[rho, rho[0]]
    phi_loop = np.r_[phi, phi[0]]
    xy_loop = np.c_[rho_loop*np.cos(phi_loop), rho_loop*np.sin(phi_loop)]
    linperim = np.sum(np.sqrt(np.sum(np.square(np.diff(xy_loop, axis=0)), axis=1)))
    neval = np.round(2.5 * linperim).astype(int)

    if neval > 1:
        x = np.linspace(0, 2 * np.pi, neval)
        if cartesian_spline:
            Sx, Sy = eval_cartesian_spline(x, rho, phi)
            rr = np.round(centre[0] + Sx).astype(int)
            cc = np.round(centre[1] + Sy).astype(int)
        else:
            R = eval_radial_spline(x, rho, phi)
            rr = np.round(centre[0] + R * np.cos(x)).astype(int)
            cc = np.round(centre[1] + R * np.sin(x)).astype(int)
        rr[rr < 0] = 0
        rr[rr >= mr] = mr - 1
        cc[cc < 0] = 0
        cc[cc >= mc] = mc - 1
    else:
        rr = np.round(centre[0]).astype(int)
        cc = np.round(centre[1]).astype(int)
    im[rr, cc] = True
    return im


def _radii_from_outline(outline, centre, ray_angles, max_ray_len):
    # Improve accuracy of edge position by smoothing the outline image and using
    # weighted averaging of pixel positions below:
    outline = filters.gaussian(outline, 0.5)

    ray_tmplt = 0.5 * np.arange(np.round(2 * max_ray_len))[:, None]
    rr_max, cc_max = outline.shape

    radii = []
    for angle in ray_angles:
        ray = np.matmul(ray_tmplt, np.array((np.cos(angle), np.sin(angle)))[None, :])
        ray = np.round(centre + ray).astype('int')
        rr, cc = ray[:, 0], ray[:, 1]
        ray = ray[(rr >= 0) & (rr < rr_max) & (cc >= 0) & (cc < cc_max), :]

        edge_pix = np.flatnonzero(
            np.squeeze(outline[ray[:, 0], ray[:, 1]]) > 0.01)

        if len(edge_pix) == 0:
            radii.append(np.NaN)
            continue

        ray = ray[edge_pix, :]
        edge_pix = np.average(ray,
                              weights=outline[ray[:, 0], ray[:, 1]],
                              axis=0)
        radii.append(np.sqrt(np.sum((edge_pix - centre)**2)))

    return np.array(radii)
    

def guess_radial_edge(edge, mask=None, rprops=None,
        nrays_thresh_map=[(5., 4), (20., 6)], nrays_max=8):
    """Guesses knot placement for a radial spline by casting rays.

    Given an edge image, this function casts out rays from the (filled)
    mask centroid and finds where they intersect with the edge.

    Args:
        edge (ndarray): A 2D bitmask of the edge for a single cell.
        mask (None or ndarray): To save recomputation, optionally provide a
            mask image that is the filled version of the edge image. Not
            required if `rprops` is specified.
        rprops (None or RegionProps): To save recomputation, optionally
            provide :py:func:`skimage.measure.regionprops` that have been
            calculated on the mask image.        
        nrays_thresh_map (List[Tuple[float, int]]): An ordered list of tuples
            ``(upper_threshold, n_rays)`` that give an upper threshold on
            major axis length for which the specified number of rays
            ``n_rays`` will be used in stead of ``nrays_max``. The first
            satisfied threshold in the list will be used to select the number
            of rays.
        nrays_max (int): The number of rays that will be used if the major
            axis length is larger than all ``upper_threshold`` values
            specified in ``nrays_thresh_map``.

    Returns:
        A tuple ``(rho, phi)`` of ndarrays giving knot locations in radial
        coordinates (radii ``rho`` and angles ``phi``) with origin at mask
        centroid as determined by :py:func:`skimage.measure.regionprops`.
    """
    if mask is None and rprops is None:
        mask = binary_fill_holes(edge)

    if rprops is None:
        rprops = regionprops(mask.astype('int'))[0]

    r_maj = rprops.major_axis_length
    nrays = nrays_max
    for upper_thresh, n in nrays_thresh_map:
        if r_maj < upper_thresh:
            nrays = n
            break

    RL, CL, RU, CU = rprops.bbox
    bbH, bbW = RU - RL, CU - CL
    bbdiag = np.sqrt(bbH * bbH + bbW * bbW)

    astep = 2 * np.pi / nrays
    angles = (np.mod(rprops.orientation + np.pi, astep) +
              np.arange(nrays)*astep - np.pi)
    centre = np.array(rprops.centroid)

    radii = _radii_from_outline(edge, centre, angles, bbdiag)

    # Use linear interpolation for any missing radii (e.g., if region intersects
    # with image boundary):
    nanradii = np.isnan(radii)
    if nanradii.all():
        radii = 0.1 * np.ones(angles.shape)
    elif nanradii.any():
        radii = np.interp(angles,
                          angles[~nanradii],
                          radii[~nanradii],
                          period=2 * np.pi)

    return radii, angles


def guess_cartesian_edge(mask, p_edge, n_trials=10, pxperknot=5.,
                         squishiness=1., alignedness=1., return_probs=False,
                         edge=None, rprops=None, border_rect=None):
    """Guesses knot placement for a cartesian spline by ordering edge pixels.

    Uses :py:func:`ordered_edge_points` to obtain an ordered sequence of edge
    pixels, then selects random subsets of these with roughly even
    perimeter-wise spacing using a Dirichlet distribution. The random trial
    with the highest probability as measured by ``p_edge`` is returned. 

    The number of knots is always even and never fewer than four. The first
    knot is biased to align with the major axis, but a normal distribution
    allows for variation about this position (see ``alignedness`` argument).
    The default parameters provide even sampling of knot positions over all
    possible values. A small ``squishiness`` favours more regular spacing
    between knots. A large ``alignedness`` favours alignment with the major
    axis.

    Args:
        mask (ndarray): A 2D bitmask of a single cell (dtype bool).
        p_edge (ndarray): A 2D image of edge probabilities.
        n_trials (int): Number of random subsets to test.
        pxperknot (float): Intended spacing (in pixels) between knots.
        squishiness (float): Scaling factor for variance of the Dirichlet
            distribution.
        alignedness (float): Scaling factor for standard deviation of position
            of the first knot.
        return_probs (bool): Specify ``True`` to include probabilities of the
            best trial, the edge image (determined by binary_edge) and the
            image of all ordered edge pixels in the return value.
        edge (None or ndarray): If a binary edge image has already been
            calculated from the mask it can optionally be provided here to
            save recomputation.
        rprops (None or RegionProps): If
            :py:func:`skimage.measure.regionprops` have already been
            calculated for the mask, they can optionally be provided here to
            save recomputation.
        border_rect (None or ndarray): To save recomputation, optionally
            provide an image of same shape as the mask with ``False`` values
            except for a 1-pixel border of ``True`` values.

    Returns:
        A tuple ``(centre, rho, phi)`` of ndarrays specifying knot placement in
        radial coordinates for the best trial. The radial coordinate system is
        centred at ``centre`` (row/column format) and the knots have radii
        given by ``rho`` and angles given by ``phi``. If
        ``return_probs=True``, then a tuple ``(centre, rho, phi, trial_prob,
        edge_prob, ordered_edge_prob)`` is returned to additionally give the
        probability of the best trial, the edge image (as determined by
        :py:func:`binary_edge`), and the image of ordered edge pixels (as
        determined by :py:func:`ordered_edge_points`).
    """
    if edge is None:
        edge = binary_edge(mask)
        
    # Define knots in radial coordinates at mask centroid
    if rprops is None:
        rprops = regionprops(mask.astype(np.uint8))[0]
    ctr = np.r_[rprops.centroid]
    
    # Get ordered list of edge points in radial coordinates
    edgepts = ordered_edge_points(mask, edge=edge, border_rect=border_rect)
    edgepts_xy = edgepts - ctr[None, :]
    edgepts_rho = np.sqrt(np.sum(np.square(edgepts_xy), axis=1))
    # Need to flip X and Y here to match row/column format
    edgepts_phi = np.arctan2(edgepts_xy[:, 1], edgepts_xy[:, 0])

    # Candidate ellipse origins should differ from ellipsoid orientation by
    # at most 1 arc pixel at half maximum radius
    # <arc length> = <angle in radians> * <radius>
    phi_tol = 2. / np.max(edgepts_rho)  # allowed tolerance
    ori_dist = np.abs(edgepts_phi - rprops.orientation)
    cand_ori = np.flatnonzero(ori_dist < phi_tol)
    if len(cand_ori) > 0:
        # Candidates are then filtered to those within one pixel of maximum
        # radius from initial set of candidates
        cand_ori = cand_ori[np.max(edgepts_rho[cand_ori]) - edgepts_rho[cand_ori] < 1]
        ellipse_ori = cand_ori[np.argmin(ori_dist[cand_ori])]
    else:
        # Just pick closest to ellipse orientation
        ellipse_ori = np.argmin(ori_dist)
    
    # Randomly select edge points as knots over multiple trials
    # Keep the trial with the highest edge probability
    N_pts = edgepts_rho.size
    frac_ori = ellipse_ori / (N_pts - 1.)
    # Choose an even number of knots, no less than 4
    N_knots = np.maximum(2 * np.round(N_pts / 2 / pxperknot).astype('int'), 4)
    cand_rho_phi = []
    cand_probs = []
    for _ in range(n_trials):
        # Split snake into roughly equal segments using Dirichlet distribution
        # We want the variance to scale as 1 / N_knots
        # Variance is further scaled by squishiness
        # Alignment with major ellipse axis is controlled by variance in
        # additive normal noise
        # defaults of squishiness=1 and alignedness=1 produce even sampling of
        # knots over [0, N_pts)
        alpha = N_knots * np.ones(N_knots) / squishiness
        sec_len = np.random.dirichlet(alpha)
        frac_ind = np.cumsum(sec_len)
        frac_ind += np.random.normal(loc=frac_ori, scale=1 / N_knots / alignedness)
        frac_ind = np.sort(np.mod(frac_ind, 1.))
        inds = np.floor(N_pts * frac_ind).astype('int')

        # Get the knot positions in radial coordinates
        knots_rho = edgepts_rho[inds]
        knots_phi = edgepts_phi[inds]

        splIm = draw_radial(knots_rho, knots_phi, ctr, p_edge.shape,
                            cartesian_spline=True)
        probs = p_edge[splIm].mean()
        cand_probs.append(probs)
        cand_rho_phi.append((ctr, knots_rho, knots_phi))

    indMax = np.argmax(cand_probs)
    if return_probs:
        raw_edge_prob = p_edge[edge].mean()
        ord_edge_prob = p_edge[edgepts[:, 0], edgepts[:, 1]].mean()
        return cand_rho_phi[indMax] + (cand_probs[indMax], raw_edge_prob, ord_edge_prob)
    else:
        return cand_rho_phi[indMax]


def inds_max_curvature(R, N_pergrp, N_elim):
    """Helper function for curvy_knots_from_outline.

    Args:
        R: row vector of radii or matrix with rows respectively giving x and y
            coordinates.
        N_pergrp: group knots into groups of this size.
        N_elim: eliminate this many knots from each group.

    Returns:
        Indices for the knots that had the highest curvature in each group.
    """
    curvature = np.diff(np.diff(R[:, np.r_[-1, 0:R.shape[1], 0]], axis=1), axis=1)
    curvature = np.sqrt(np.sum(curvature**2, axis=0))
    knorder = np.argsort(np.reshape(curvature, (-1, N_pergrp)), axis=1)
    rowinds = np.reshape(np.arange(knorder.size), (-1, N_pergrp))
    rowinds = rowinds[np.tile(np.arange(knorder.shape[0])[:, None], (1, N_pergrp)), knorder]
    return np.sort(rowinds[:, N_elim:].flatten())


def curvy_knots_from_outline(outline, rprops, cartesian_spline=False,
                             n_knots_fraction=0.5):
    """Places knots on an outline image at points of high curvature.

    The method essentially progressively eliminates knots with low curvature
    from an initial dense equi-angular array of knots located on the outline.
    Curvature is calculated as the difference of the difference between
    consecutive locations (radii for a radial spline; the norm of x and y for
    a cartesian spline). 

    Args:
        outline (ndarray): A 2D bitmask for a single cell edge.
        rprops (RegionProperties): The output of
            :py:func:`skimage.measure.regionprops` for the filled outline.
        cartesian_spline (bool): Specify ``True`` to model curvature in
            cartesian rather than radial coordinates.
        n_knots_fraction (float): A value in ``[0, 1]`` that determines the
            number of knots as a fraction of the number of edge pixels.

    Returns:
        A (radii, angles) tuple specifying the locations of the curvy knots in
        radial coordinates from the `rprops` centroid
    """
    Nedgepx = outline.sum()
    # Initial dense number of rays is one ray for every four edge pixels,
    # rounded to the nearest power of 2:
    # Nrays_dense = 2**(np.round(np.log(max(Nedgepx, 16)) / np.log(2)).astype(int) - 2)
    # Initial dense number of rays is one ray for every eight edge pixels,
    # rounded to the nearest power of 2:
    Nrays_dense = 2**(np.round(np.log(max(Nedgepx, 32)) / np.log(2)).astype(int) - 3)

    # The target number of rays rounded to the nearest four:
    Nrays_final = 4 * max(1, np.round(0.25 * Nedgepx * n_knots_fraction).astype(int))

    # Rays can be no larger than the diagonal of the bounding box
    RL, CL, RU, CU = rprops.bbox
    bbdiag = np.sqrt((RU - RL)**2 + (CU - CL)**2)

    # Determine radii for initial dense array
    astep = 2 * np.pi / Nrays_dense
    angles = np.mod(rprops.orientation + np.pi, astep) + \
        np.arange(Nrays_dense)*astep - np.pi
    # Roughly compensate for elliptical squeezing by converting parameterised
    # ellipse parameter to true angle. See:
    # https://math.stackexchange.com/a/436125
    axmaj, axmin = rprops.major_axis_length, rprops.minor_axis_length
    angles -= np.arctan((axmaj - axmin) * np.tan(angles) / (axmaj + axmin * np.tan(angles)**2))

    centre = np.array(rprops.centroid)
    radii = _radii_from_outline(outline, centre, angles, bbdiag)

    # Linearly interpolate any missing radii (e.g., if region intersects with
    # image boundary):
    nanradii = np.isnan(radii)
    if nanradii.all():
        radii = 0.1 * np.ones(angles.shape)
    elif nanradii.any():
        radii = np.interp(angles,
                          angles[~nanradii],
                          radii[~nanradii],
                          period=2 * np.pi)

    if cartesian_spline:
        R = radii * np.vstack((np.cos(angles), np.sin(angles)))
    else:
        R = radii[None, :]

    # Progressively eliminate knots four at a time until the desired number of
    # knots is reached
    inds = np.arange(R.shape[1])
    for N_pergrp in np.r_[Nrays_dense // 4 : Nrays_final // 4 : -1]:
        inds = inds[inds_max_curvature(R[:, inds], N_pergrp, 1)]
        # Rotate indices over periodic boundary to mix groups
        inds = inds[np.r_[-1,0:len(inds)-1]]
    inds = np.sort(inds)

    return radii[inds], angles[inds]


class FakeRegionProperties(NamedTuple):
    centroid: Tuple[int, int]
    orientation: float
    bbox: Tuple[int, int, int, int]
    major_axis_length: float
    minor_axis_length: float


GUESS_RADIAL_EDGE_PARAMS = {
    p.name for p in
    inspect.signature(guess_radial_edge).parameters.values()
    if p.kind == p.POSITIONAL_OR_KEYWORD and p.default != p.empty
}


GUESS_CARTESIAN_EDGE_PARAMS = {
    p.name for p in
    inspect.signature(guess_cartesian_edge).parameters.values()
    if p.kind == p.POSITIONAL_OR_KEYWORD and p.default != p.empty
}


CURVY_KNOTS_FROM_OUTLINE_PARAMS = {
    p.name for p in
    inspect.signature(curvy_knots_from_outline).parameters.values()
    if p.kind == p.POSITIONAL_OR_KEYWORD and p.default != p.empty
}


def mask_to_knots(mask,
                  p_edge=None,
                  return_outline=True,
                  cartesian_spline=False,
                  curvy_knots=False,
                  bbox_padding=10,
                  **kwargs):
    """Guess knot positions from a cell bitmask.

    Essentially a unified interface to the different methods for initial knot
    placement before refinement. By default, it finds knots using
    :py:func:`guess_radial_edge`. If ``cartesian_spline=True`` and
    ``curvy_knots=False``, then knots are placed according to
    :py:func:`guess_cartesian_edge`. If ``curvy_knots=True``, then knots are
    found using :py:func:`curvy_knots_from_outline`.

    For increased performance, processing is limited to a padded bounding box
    of the mask image. Any return values are, however, restored to the input
    size/coordinates.

    Args:
        mask (ndarray): A 2D bitmask for a single cell.
        p_edge(ndarray): A 2D edge probability image with same shape as mask.
            Currently only required when ``cartesian_spline=True`` and
            ``curvy_knots=False``.
        return_outline (bool): Specify ``False`` if you only want knot
            coordinates returned.
        cartesian_spline (bool): Specify ``True`` to determine knots for
            splines interpolated in cartesian rather than radial coordinates.
        curvy_knots (bool): Specify ``True`` to place knots at points of high
            curvature according to :py:func:`curvy_knots_from_outline`.
        bbox_padding (int): The number of pixels of padding that should be
            included when limiting processing to the mask bounding box.
        **kwargs: Additional arguments to be passed to any of the downstream
            functions (:py:func:`guess_radial_edge`,
            :py:func:`guess_cartesian_edge`, or
            :py:func:`curvy_knots_from_outline`).

    Returns:
        A tuple ``(knot_coordinates, edge_image)`` of knot coordinates
        ``(centre, knot_rho, knot_phi)`` defined in a radial coordinate system
        with origin ``centre`` (a size two ndarray giving row/column
        location), and ndarrays ``knot_rho`` and ``knot_phi`` giving the
        radius and polar angle respectively for each knot. If
        ``return_outline=False``, then only ``knot_coordinates`` are returned.
    """

    rprops = single_region_prop(mask)
    if cartesian_spline and not curvy_knots:
        (mask_bb, p_edge_bb), bbunmap = limit_to_bbox(
            (mask, p_edge), rprops.bbox)
    else:
        (mask_bb,), bbunmap = limit_to_bbox((mask,), rprops.bbox)
    edge_bb = binary_edge(mask_bb)

    ctr = np.r_[rprops.centroid]
    bboffset = np.r_[bbunmap[:2]]
    limited_bb = np.r_[rprops.bbox] - np.tile(bboffset, 2)
    rprops_bb = FakeRegionProperties(
        centroid=ctr - bboffset,
        orientation=rprops.orientation,
        bbox=tuple(limited_bb.tolist()),
        major_axis_length=rprops.major_axis_length,
        minor_axis_length=rprops.minor_axis_length)

    if curvy_knots:
        params = {
            k: v for k, v in kwargs.items()
            if k in CURVY_KNOTS_FROM_OUTLINE_PARAMS
        }
        k_rho, k_phi = curvy_knots_from_outline(
            edge_bb, rprops_bb, cartesian_spline=cartesian_spline, **params)
    elif cartesian_spline:
        params = {
            k: v for k, v in kwargs.items()
            if k in GUESS_CARTESIAN_EDGE_PARAMS
        }
        _, k_rho, k_phi = guess_cartesian_edge(
            mask_bb, p_edge_bb, edge=edge_bb, rprops=rprops_bb,
            return_probs=False, **params)
    else:
        params = {
            k: v for k, v in kwargs.items() if k in GUESS_RADIAL_EDGE_PARAMS
        }
        k_rho, k_phi = guess_radial_edge(edge_bb, rprops=rprops_bb, **params)

    if return_outline:
        edge = draw_radial(k_rho, k_phi, ctr, mask.shape,
                           cartesian_spline=cartesian_spline)
        return (ctr, k_rho, k_phi), edge
    else:
        return ctr, k_rho, k_phi


############################
###  OUTLINE REFINEMENT  ###
############################


def prior_resid_weight(resid, gauss_scale=5, exp_scale=1):
    """Weights radial residuals to bias towards the initial guess

    Weight decays as a gaussian for positive residuals and exponentially for
    negative residuals. So assuming `resid = rho_guess - rho_initial`, then
    larger radii are favoured.
    """
    W = np.zeros(resid.shape)
    W[resid >= 0] = np.exp(-resid[resid >= 0]**2 / gauss_scale)
    W[resid < 0] = np.exp(resid[resid < 0] / exp_scale)
    return W


def adj_rspline_coords(adj, ref_radii, ref_angles):
    """Map optimisation-space radial spline params to standard values

    Params in optimisation space are specified relative to reference radii and
    reference angles. If constrained to [-1, 1], optimisation parameters will
    allow a 30% change in radius, or change in angle up to 1/4 the distance
    between consecutive angles.
    """
    npoints = len(ref_radii)
    return (
        # allow up to 30% change in radius
        ref_radii * (1 + 0.3 * adj[:npoints]),
        # allow changes in angle up to 1/4 the distance between points
        ref_angles + adj[npoints:] * np.pi / (2 * npoints))


def adj_rspline_resid(adj, rho, phi, probs, ref_radii, ref_angles):
    """Weighted residual for radial spline optimisation

    Optimisation params (`adj`) are mapped according to `adj_rspline_coords`.
    Target points are given in radial coordinates `rho` and `phi` with weights
    `probs`. Optimisation is defined relative to radial spline params
    `ref_radii` and `ref_angles`.
    """
    radii, angles = adj_rspline_coords(adj, ref_radii, ref_angles)
    return probs * (rho - eval_radial_spline(phi, radii, angles))


def adj_cart_spline_resid(adj, rho, phi, probs, ref_radii, ref_angles):
    """Weighted residual for cartesian spline optimisation in radial
    coordinate system

    Optimisation params (`adj`) are mapped according to `adj_cart_spline_coords`.
    Target points are given in radial coordinates `rho` and `phi` with weights
    `probs`. Optimisation is defined relative to radial coordinates
    `ref_radii` and `ref_angles`.
    """
    radii, angles = adj_rspline_coords(adj, ref_radii, ref_angles)
    return probs * (rho - eval_cartesian_spline(phi, radii, angles))


def refine_radial_grouped(grouped_coords, grouped_p_edges, cartesian_spline=False):
    """Refine initial radial spline by optimising to predicted edge

    Neighbouring groups are used to re-weight predicted edges belonging to
    other cells using the initial guess
    """

    if cartesian_spline:
        eval_spline = eval_cartesian_spline
        adj_resid = adj_cart_spline_resid
    else:
        eval_spline = eval_radial_spline
        adj_resid = adj_rspline_resid

    # Determine edge pixel locations and probabilities from NN prediction
    p_edge_locs = [np.nonzero(p_edge > 0.2) for p_edge in grouped_p_edges]
    p_edge_probs = [
        p_edge[rr, cc]
        for p_edge, (rr, cc) in zip(grouped_p_edges, p_edge_locs)
    ]

    p_edge_count = [len(rr) for rr, _ in p_edge_locs]

    opt_coords = []
    ngroups = len(grouped_coords)
    for g, g_coords in enumerate(grouped_coords):
        # If this group has no predicted edges, keep initial and skip
        if p_edge_count[g] == 0:
            opt_coords.append(g_coords)
            continue

        # Compile a list of all cells in this and neighbouring groups
        nbhd = list(
            chain.from_iterable([
                [((gi, ci), coords)
                 for ci, coords in enumerate(grouped_coords[gi])]
                for gi in range(max(g - 1, 0), min(g + 2, ngroups))
                if
                p_edge_count[gi] > 0  # only keep if there are predicted edges
            ]))
        if len(nbhd) > 0:
            nbhd_ids, nbhd_coords = zip(*nbhd)
        else:
            nbhd_ids, nbhd_coords = 2 * [[]]

        # Calculate edge pixels in radial coords for all cells in this and
        # neighbouring groups:
        radial_edges = [
            rc_to_radial(p_edge_locs[g], centre)
            for centre, _, _ in nbhd_coords
        ]

        # Calculate initial residuals and prior weights
        resids = [
            rho - eval_spline(phi, radii, angles)
            for (rho, phi), (_, radii,
                             angles) in zip(radial_edges, nbhd_coords)
        ]
        indep_weights = [prior_resid_weight(r) for r in resids]

        probs = p_edge_probs[g]

        g_opt_coords = []
        for c, (centre, radii, angles) in enumerate(g_coords):
            ind = nbhd_ids.index((g, c))
            rho, phi = radial_edges[ind]
            p_weighted = probs * indep_weights[ind]
            other_weights = indep_weights[:ind] + indep_weights[ind + 1:]
            if len(other_weights) > 0:
                p_weighted *= (1 - np.mean(other_weights, axis=0))

            # Remove insignificant fit data
            signif = p_weighted > 0.1
            if signif.sum() < 10:
                # With insufficient data, skip optimisation
                g_opt_coords.append((centre, radii, angles))
                continue
            p_weighted = p_weighted[signif]
            phi = phi[signif]
            rho = rho[signif]

            nparams = len(radii) + len(angles)
            opt = least_squares(adj_resid,
                                np.zeros(nparams),
                                bounds=(-np.ones(nparams), np.ones(nparams)),
                                args=(rho, phi, p_weighted, radii, angles),
                                ftol=5e-2)

            g_opt_coords.append((centre,) +
                                adj_rspline_coords(opt.x, radii, angles))

        opt_coords.append(g_opt_coords)

    return opt_coords


############################
### DEPRECATED FUNCTIONS ###
############################


def get_regions(p_img, threshold):
    """Find regions in a probability image sorted by likelihood"""
    p_thresh = p_img > threshold
    p_label = label(p_thresh, background=0)
    rprops = regionprops(p_label, p_img)
    rprops = [
        r for r in rprops
        if r.major_axis_length > 0 and r.minor_axis_length > 0
    ]
    rprops.sort(key=lambda x: x.mean_intensity, reverse=True)
    return rprops


def morph_thresh_masks(p_interior,
                       p_overlap=None,
                       dilate=True,
                       interior_threshold=0.9,
                       overlap_threshold=0.9):

    ilabels = label(p_interior > interior_threshold)

    if p_overlap is not None:
        othresh = p_overlap > overlap_threshold

    masks = []

    for l in range(np.max(ilabels)):
        cr = ilabels == l + 1

        if dilate:
            cr = binary_dilation(cr)

        if p_overlap is not None:
            olabels = label(binary_closing(cr | othresh))
            # Occasionally the closing operation can make all cells disappear
            nlbls = np.max(olabels)
            if nlbls > 0:
                omasks = [olabels == i + 1 for i in range(nlbls)]
                omask_iou = [mask_iou(cr, r) for r in omasks]
                cr = omasks[np.argmax(omask_iou)]

        masks.append(cr)

    return masks


def unique_masks(masks, ref_masks, threshold=0.5, iou_func=mask_iou):
    """Returns a subset of masks excluding any overlaps with ref_masks"""

    rmasks = ref_masks.copy()  # rmasks will be edited to remove used refs
    umasks = []
    for m in masks:
        if len(rmasks) == 0:
            umasks.append(m)
            continue

        ref_iou = [iou_func(m, r) for r in rmasks]
        indmax = np.argmax(ref_iou)
        if ref_iou[indmax] > threshold:
            # This reference mask is now accounted for
            rmasks.pop(indmax)
        else:
            # No reference masks match this one, so keep it
            umasks.append(m)

    return umasks


def get_edge_force(rprop, shape):
    r_major = rprop.major_axis_length / 2
    r_minor = rprop.minor_axis_length / 2
    angle = -rprop.orientation
    nr = shape[0]
    nc = shape[1]
    xmat = np.matmul(np.arange(0, nr)[:, None], np.ones((1, nc)))
    ymat = np.matmul(np.ones((nr, 1)), np.arange(0, nc)[None, :])
    xy = np.vstack([np.reshape(xmat, (1, -1)), np.reshape(ymat, (1, -1))])
    rotmat = np.array([[np.cos(angle), -np.sin(angle)],
                       [np.sin(angle), np.cos(angle)]])
    radial_index = np.matmul(
        rotmat, (xy - np.array(rprop.centroid)[:, None])) / np.array(
            [r_major, r_minor])[:, None]
    return np.reshape(1 - np.exp(-np.sum((radial_index)**2, 0)), (nr, nc))


def morph_ac_seg(cnn_outputs,
                 interior_threshold=0.9,
                 overlap_threshold=0.9,
                 bud_threshold=0.9,
                 bud_dilate=False,
                 bud_overlap=False,
                 isbud_threshold=0.5,
                 use_edge_force=True,
                 ac_edge_threshold=0.4,
                 ac_overlap_threshold=0.4,
                 iterations=5,
                 balloon_force=0.5,
                 smoothing=2):

    p_edge, _, p_interior, p_overlap, _, p_bud = cnn_outputs

    p_over = p_overlap  # guarantee existence for AC methods

    if overlap_threshold is None:
        p_overlap = None

    masks = morph_thresh_masks(p_interior,
                               interior_threshold=interior_threshold,
                               p_overlap=p_overlap,
                               overlap_threshold=overlap_threshold)

    outmasks = []
    if bud_threshold is not None:
        if not bud_overlap:
            p_overlap = None

        budmasks = morph_thresh_masks(p_bud,
                                      interior_threshold=bud_threshold,
                                      dilate=False,
                                      p_overlap=p_overlap,
                                      overlap_threshold=overlap_threshold)

        # Omit interior masks if they overlap with bud masks
        masks = unique_masks(masks, budmasks, threshold=isbud_threshold)

        outmasks += budmasks

    rprops = [regionprops(m.astype('int'))[0] for m in masks]
    rprops = [
        r for r in rprops
        if r.major_axis_length > 0 and r.minor_axis_length > 0
    ]

    e_forces = [get_edge_force(r, p_edge.shape) for r in rprops]
    edge_thresh = p_edge > ac_edge_threshold
    over_thresh = p_over > ac_overlap_threshold

    from skimage.segmentation import morphological_geodesic_active_contour
    for j, (force, mask) in enumerate(zip(e_forces, masks)):
        ij_edge_im = p_edge.copy()

        if use_edge_force:
            other_forces = e_forces.copy()
            other_forces.pop(j)
            other_forces = np.sum(np.array(other_forces), 0)
            ef = (force + other_forces) / len(e_forces)
            ij_edge_im[edge_thresh] = ef[edge_thresh]
            ij_edge_im[over_thresh] = force[over_thresh]

        final_mask = morphological_geodesic_active_contour(
            1 - ij_edge_im,
            iterations,
            mask,
            balloon=balloon_force,
            smoothing=smoothing)

        outmasks.append(final_mask)

    # Return only the mask outlines
    outlines = [
        minimum_filter(m, footprint=squareconn) != m for m in outmasks
    ]

    return outlines


def morph_ellipse_seg(cnn_outputs,
                      interior_threshold=0.9,
                      overlap_threshold=0.9,
                      bud_threshold=0.9,
                      bud_dilate=False,
                      bud_overlap=False,
                      isbud_threshold=0.5,
                      scaling=1.0,
                      offset=0):
    """Segment cell outlines from morphology output of CNN as region ellipses

    Specify `overlap_threshold` or `bud_threshold` as `None` to ignore.
    """

    _, _, p_interior, p_overlap, _, p_bud = cnn_outputs

    if overlap_threshold is None:
        p_overlap = None

    masks = morph_thresh_masks(p_interior,
                               interior_threshold=interior_threshold,
                               p_overlap=p_overlap,
                               overlap_threshold=overlap_threshold)

    if bud_threshold is not None:
        if not bud_overlap:
            p_overlap = None

        budmasks = morph_thresh_masks(p_bud,
                                      interior_threshold=bud_threshold,
                                      dilate=False,
                                      p_overlap=p_overlap,
                                      overlap_threshold=overlap_threshold)

        # Omit interior masks if they overlap with bud masks
        masks = budmasks + unique_masks(
            masks, budmasks, threshold=isbud_threshold)

    rprops = [regionprops(m.astype('int'))[0] for m in masks]
    rprops = [
        r for r in rprops
        if r.major_axis_length > 0 and r.minor_axis_length > 0
    ]

    outlines = []

    from skimage.draw import ellipse_perimeter
    for region in rprops:
        r, c = np.round(region.centroid).astype('int')
        r_major = np.round(scaling * region.major_axis_length / 2 +
                           offset).astype('int')
        r_minor = np.round(scaling * region.minor_axis_length / 2 +
                           offset).astype('int')
        orientation = -region.orientation
        rr, cc = ellipse_perimeter(r,
                                   c,
                                   r_major,
                                   r_minor,
                                   orientation=orientation,
                                   shape=p_interior.shape)
        outline = np.zeros(p_interior.shape, dtype='bool')
        outline[rr, cc] = True
        outlines.append(outline)

    return outlines


def morph_thresh_seg(cnn_outputs,
                     interior_threshold=0.9,
                     overlap_threshold=0.9,
                     bud_threshold=0.9,
                     bud_dilate=False,
                     bud_overlap=False,
                     isbud_threshold=0.5):
    """Segment cell outlines from morphology output of CNN by thresholding

    Specify `overlap_threshold` or `bud_threshold` as `None` to ignore.
    """

    _, _, p_interior, p_overlap, _, p_bud = cnn_outputs

    if overlap_threshold is None:
        p_overlap = None

    if isbud_threshold is None and bud_threshold is not None:
        p_interior = p_interior * (1 - p_bud)

    masks = morph_thresh_masks(p_interior,
                               interior_threshold=interior_threshold,
                               p_overlap=p_overlap,
                               overlap_threshold=overlap_threshold)

    if bud_threshold is not None:
        if not bud_overlap:
            p_overlap = None

        budmasks = morph_thresh_masks(p_bud,
                                      interior_threshold=bud_threshold,
                                      dilate=False,
                                      p_overlap=p_overlap,
                                      overlap_threshold=overlap_threshold)

        if isbud_threshold is not None:
            # Omit interior masks if they overlap with bud masks
            masks = unique_masks(masks,
                                 budmasks,
                                 iou_func=mask_containment,
                                 threshold=isbud_threshold) + budmasks

    # Return only the mask outlines
    outlines = [minimum_filter(m, footprint=squareconn) != m for m in masks]

    return outlines


def morph_radial_thresh_seg(cnn_outputs,
                            interior_threshold=0.9,
                            overlap_threshold=0.9,
                            bud_threshold=0.9,
                            bud_dilate=False,
                            bud_overlap=False,
                            isbud_threshold=0.5,
                            edge_dampening=False):
    """Segment cell outlines from morphology output of CNN by fitting radial
    spline to threshold output

    Specify `overlap_threshold` or `bud_threshold` as `None` to ignore.
    """

    p_edge, _, p_interior, p_overlap, _, p_bud = cnn_outputs

    shape = p_interior.shape

    if overlap_threshold is None:
        p_overlap = None

    if isbud_threshold is None and bud_threshold is not None:
        p_interior *= 1 - p_bud

    if edge_dampening:
        p_interior *= 1 - p_edge

    masks = morph_thresh_masks(p_interior,
                               interior_threshold=interior_threshold,
                               p_overlap=p_overlap,
                               overlap_threshold=overlap_threshold)

    if bud_threshold is not None:
        if not bud_overlap:
            p_overlap = None

        budmasks = morph_thresh_masks(p_bud,
                                      interior_threshold=bud_threshold,
                                      dilate=False,
                                      p_overlap=p_overlap,
                                      overlap_threshold=overlap_threshold)

        if isbud_threshold is not None:
            # Omit interior masks if they overlap with bud masks
            masks = budmasks + unique_masks(
                masks, budmasks, threshold=isbud_threshold)
        else:
            masks = masks + budmasks

    # Need mask outlines and region properties
    mseg = [minimum_filter(m, footprint=squareconn) != m for m in masks]
    rprops = [regionprops(m.astype('int'))[0] for m in masks]

    outlines = []
    for mask, outline, rp in zip(masks, mseg, rprops):
        radii, angles = guess_radial_edge(outline, mask, rp)
        if np.any(np.isnan(radii)):
            return mask, outline, rp
        outlines.append(draw_radial(radii, angles, rp.centroid, shape))

    return outlines


def morph_seg_grouped(pred,
                      flattener,
                      cellgroups=['large', 'medium', 'small'],
                      interior_threshold=0.5,
                      nclosing=0,
                      nopening=0,
                      connectivity=2,
                      min_area=10,
                      pedge_thresh=None,
                      fit_radial=False,
                      use_group_thresh=False,
                      group_thresh_expansion=0.,
                      ingroup_edge_segment=False,
                      containment_thresh=0.8,
                      containment_func=mask_containment,
                      refine_outlines=False,
                      return_masks=False,
                      return_coords=False):
    """Morphological segmentation for model predictions of flattener targets

    DEPRECATED. Use morph_thres_seg.MorphSegGrouped class.

    :param pred: list of prediction images (ndarray with shape (x, y))
        matching `flattener.names()`
    :param flattener: an instance of `SegmentationFlattening` defining the
        prediction targets `pred`
    :param cellgroups: list of `flattener` group names to be used for the
        segmentation task. Tuples of group names can also be included in the
        list, in which case, those groups will be merged, generating a new
        group.

    :returns: a list of boolean edge images (ndarray shape (x, y)), one for
        each cell identified. If `return_masks` and/or `return_coords` are
        true, the output will be a tuple of edge images, filled masks, and/or
        radial coordinates.
    """

    if len(pred) != len(flattener.names()):
        raise BadParam(
            '"pred" arg does not match number of flattener targets')

    shape = np.squeeze(pred[0]).shape

    ngroups = len(cellgroups)

    def broadcast_arg(arg, argname, t=int):
        if type(arg) == t or arg is None:
            arg = (arg,) * ngroups

        arg = list(arg)
        assert len(arg) == ngroups, \
            '"{}" is of incorrect length'.format(argname)
        return arg

    interior_threshold = broadcast_arg(interior_threshold,
                                       'interior_threshold', float)
    nclosing = broadcast_arg(nclosing, 'nclosing')
    nopening = broadcast_arg(nopening, 'nopening')
    min_area = broadcast_arg(min_area, 'min_area')
    connectivity = broadcast_arg(connectivity, 'connectivity')
    if pedge_thresh is not None:
        pedge_thresh = broadcast_arg(pedge_thresh, 'pedge_thresh', float)
    group_thresh_expansion = broadcast_arg(group_thresh_expansion,
                                           'group_thresh_expansion', float)

    tnames = flattener.names()

    if not fit_radial and len(pred) > 0:
        border_rect = np.zeros(shape, dtype='bool')
        border_rect[0, :] = True
        border_rect[:, 0] = True
        border_rect[-1, :] = True
        border_rect[:, -1] = True

    group_segs = []
    p_edges = []
    groupargs = zip(cellgroups, interior_threshold, nclosing, nopening,
                    min_area, connectivity, group_thresh_expansion)
    for group, thresh, nc, no, ma, conn, gte in groupargs:

        if type(group) == str:
            group = (group,)

        p_int, p_edge, tdefs = [], [], []
        for g in group:
            t_int, t_fill, t_edge = flattener.getGroupTargets(
                g, ['interior', 'filled', 'edge'])
            t_int = t_int or t_fill
            assert t_int is not None, \
                '"{}" has no "interior" or "filled" target'.format(g)
            tdefs.append(flattener.getTargetDef(t_int))
            p_int.append(pred[tnames.index(t_int)])
            p_edge.append(pred[tnames.index(t_edge)])

        if use_group_thresh:
            lower = tdefs[0].get('lower', 1)
            upper = tdefs[-1].get('upper', np.Inf)
            if upper == np.Inf:
                expansion = gte * lower
            else:
                expansion = gte * (upper - lower)
            lower -= expansion
            lower = max(lower, ma)
            upper += expansion
        else:
            lower, upper = ma, np.Inf

        nerode = [d.get('nerode', 0) for d in tdefs]
        max_ne = np.max(nerode)

        if len(p_int) == 1:
            p_int = p_int[0]
            p_edge = p_edge[0]
        else:
            # Perform morphological erosion on any group members that do not
            # match the maximum (NB: erosion here is non-binary):
            p_int = [
                iterative_erosion(p, max_ne - ne)
                for p, ne in zip(p_int, nerode)
            ]
            p_int = np.dstack(p_int).max(axis=2)
            p_edge = np.dstack(p_edge).max(axis=2)

        # Save the edge predictions for refinement later
        p_edges.append(p_edge)

        if ingroup_edge_segment:
            p_int = p_int * (1 - p_edge)
            if max_ne == 0:
                max_ne = 1

        masks_areas = [
            (m, a)
            for m, a in threshold_segmentation(
                p_int, interior_threshold=thresh or 0.5,
                nclosing=nc or 0, nopening=no or 0,
                ndilate=max_ne, return_area=True, connectivity=conn)
            if a >= lower and a < upper
        ]

        if len(masks_areas) > 0:
            masks, areas = zip(*masks_areas)
        else:
            masks, areas = [], []

        if fit_radial:
            coords, edges = list(
                zip(*[
                    mask_to_knots(mask)
                    for mask in masks
                ])) or ([], [])
            masks = [binary_fill_holes(o) for o in edges]
        else:
            edges = [binary_edge(m) for m in masks]
            edges = [e | (border_rect & m) for e, m in zip(edges, masks)]
            coords = [tuple()] * len(masks)

        if pedge_thresh is not None:
            assert t_edge is not None, '"{}" has no "edge" target'.format(g)
            p_edge = pred[tnames.index(t_edge)]
            edge_scores = get_edge_scores(edges, p_edge)
        else:
            edge_scores = np.ones(len(masks))

        group_segs.append(
            list(
                zip(masks, areas,
                    len(masks) * (lower, upper), edge_scores, edges, coords)))

    # Only keep cells whose outlines overlap well with the predicted edge
    if pedge_thresh is not None:
        group_segs = [[val
                       for val in group
                       if val[3] > t]
                      for t, group in zip(pedge_thresh, group_segs)]

    # Resolve any cells duplicated across adjacent groups:
    for lgi, ugi in zip(range(0, ngroups - 1), range(1, ngroups)):
        lg = group_segs[lgi]
        ug = group_segs[ugi]

        pairs = np.array([
            (li, ui) for ui in range(len(ug)) for li in range(len(lg))
        ])
        containment = np.array(
            [containment_func(lg[li][0], ug[ui][0]) for li, ui in pairs])

        pairs = pairs[containment > containment_thresh]

        if pedge_thresh is not None:
            lg_discard = [li for li, ui in pairs if lg[li][3] < ug[ui][3]]
            ug_discard = [ui for li, ui in pairs if lg[li][3] >= ug[ui][3]]
        else:
            lg_discard = [li for li, ui in pairs if lg[li][1] < ug[ui][1]]
            ug_discard = [ui for li, ui in pairs if lg[li][1] >= ug[ui][1]]

        group_segs[lgi] = [
            val for li, val in enumerate(lg) if li not in lg_discard
        ]
        group_segs[ugi] = [
            val for ui, val in enumerate(ug) if ui not in ug_discard
        ]

    if refine_outlines:
        # Refine outlines using edge predictions
        grouped_coords = [[cell[-1] for cell in group] for group in group_segs
                         ]
        coords = list(
            chain.from_iterable(refine_radial_grouped(grouped_coords,
                                                      p_edges)))
        edges = [
            draw_radial(radii, angles, centre, shape)
            for centre, radii, angles in coords
        ]
        if return_masks:
            masks = [binary_fill_holes(e) for e in edges]
    else:
        # Extract edges, masks and AC coordinates from initial segmentation
        outputs = list(
            chain.from_iterable([[(e, m, c)
                                  for m, _, _, _, e, c in group]
                                 for group in group_segs]))
        if len(outputs) > 0:
            edges, masks, coords = zip(*outputs)
        else:
            edges, masks, coords = 3 * [[]]

    if return_coords or return_masks:
        output = (edges,)
        if return_masks:
            output += (masks,)
        if return_coords:
            output += (coords,)
        return output
    else:
        return edges


def morph_radial_edge_seg(cnn_outputs):
    RL, CL, RU, CU = rp.bbox
    Rext = np.ceil(0.25 * (RU - RL)).astype('int')
    Cext = np.ceil(0.25 * (CU - CL)).astype('int')
    RL = np.max([RL - Rext, 0])
    RU = np.min([RU + Rext, pedge.shape[0]])
    CL = np.max([CL - Cext, 0])
    CU = np.min([CU + Cext, pedge.shape[1]])

    cell_p_edge = p_edge[RL:RU, CL:CU]
    rr, cc = np.where(cell_p_edge > 0.3)
    prob = cell_p_edge[rr, cc]
    rr = rr - (rp.centroid[0] - RL)
    cc = cc - (rp.centroid[1] - CL)
    phi = np.arctan2(cc, rr) - rp.orientation
    rho = np.sqrt(rr**2 + cc**2)

    ac_seg = np.zeros((RU - RL, CU - CL), dtype='bool')
    ac_phi = np.linspace(0, 2 * np.pi,
                         np.round(4 * np.pi * np.max(radii)).astype('int'))
    ac_rho = eval_radial_spline(ac_phi, radii, angles)
    ac_rr = np.round(rp.centroid[0] - RL +
                     ac_rho * np.cos(ac_phi)).astype('int')
    ac_cc = np.round(rp.centroid[1] - CL +
                     ac_rho * np.sin(ac_phi)).astype('int')
    ac_seg[ac_rr, ac_cc] = True
