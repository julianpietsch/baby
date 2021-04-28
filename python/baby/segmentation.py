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
from collections import Iterable
from itertools import chain, compress, repeat
import numpy as np
from numpy import newaxis as nax
from scipy.ndimage import (minimum_filter, binary_dilation, binary_erosion,
                           binary_closing, binary_opening, binary_fill_holes)
from scipy import interpolate
from scipy.optimize import least_squares
from skimage.measure import label, regionprops
from skimage.segmentation import morphological_geodesic_active_contour
from skimage.morphology import diamond, erosion, dilation
from skimage.draw import ellipse_perimeter
from skimage import filters

from .errors import BadParam

squareconn = diamond(1)  # 3x3 filter for 1-connected patches
fullconn = np.ones((3, 3), dtype='uint8')


def binary_edge(imfill, footprint=fullconn):
    """Get square-connected edges from filled image:"""
    return minimum_filter(imfill, footprint=footprint) != imfill


def mask_iou(a, b):
    """Intersection over union (IoU) between boolean masks"""
    return np.sum(a & b) / np.sum(a | b)


def mask_containment(a, b):
    """Max of intersection over a or over b"""
    return np.max(np.sum(a & b) / np.array([np.sum(a), np.sum(b)]))


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


def bbox_overlaps(regA, regB):
    """Returns True if the regions have overlapping bounding boxes"""
    lrA, lcA, urA, ucA = regA.bbox
    lrB, lcB, urB, ucB = regB.bbox
    rA = np.array([lrA, urA])
    cA = np.array([lcA, ucA])
    return ((not ((rA > urB).all() or (rA < lrB).all())) and
            (not ((cA > ucB).all() or (cA < lcB).all())))


def region_iou(regA, regB):
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
        return 0.0


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


def get_edge_force(rprop, shape):
    r_major = rprop.major_axis_length / 2
    r_minor = rprop.minor_axis_length / 2
    angle = -rprop.orientation
    nr = shape[0]
    nc = shape[1]
    xmat = np.matmul(np.arange(0, nr)[:, nax], np.ones((1, nc)))
    ymat = np.matmul(np.ones((nr, 1)), np.arange(0, nc)[nax, :])
    xy = np.vstack([np.reshape(xmat, (1, -1)), np.reshape(ymat, (1, -1))])
    rotmat = np.array([[np.cos(angle), -np.sin(angle)],
                       [np.sin(angle), np.cos(angle)]])
    radial_index = np.matmul(
        rotmat, (xy - np.array(rprop.centroid)[:, nax])) / np.array(
            [r_major, r_minor])[:, nax]
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


def eval_radial_spline(x, rho, phi):
    """Evaluate a radial spline defined by radii and angles
    rho: vector of radii for each point defining the spline
    phi: angles in [-pi,pi) defining points of the spline
    The spline is periodic across the boundary
    """

    # Angles need to be in increasing order to correctly loop over boundary
    order = np.argsort(phi)
    rho = rho[order]
    phi = phi[order]
    offset = phi[0]

    # Make the boundaries periodic
    rho = np.concatenate((rho, rho[0, nax]))
    phi = np.concatenate((phi - offset, (2 * np.pi,)))

    tck = interpolate.splrep(phi, rho, per=True)
    try:
        return interpolate.splev(np.mod(x - offset, 2 * np.pi), tck)
    except ValueError as err:
        print('x:')
        print(x)
        print('rho:')
        print(rho)
        print('phi:')
        print(phi)
        raise err


def morph_radial_thresh_fit(outline, mask=None, rprops=None):
    if mask is None and rprops is None:
        mask = binary_fill_holes(outline)

    if rprops is None:
        rprops = regionprops(mask.astype('int'))[0]

    r_maj = rprops.major_axis_length
    nrays = 4 if r_maj < 5 else 6 if r_maj < 20 else 8

    RL, CL, RU, CU = rprops.bbox
    bbdiag = np.sqrt((RU - RL)**2 + (CU - CL)**2)
    rr_max, cc_max = outline.shape

    astep = 2 * np.pi / nrays
    angles = np.mod(rprops.orientation + np.pi, astep) + \
        np.arange(nrays)*astep - np.pi
    centre = np.array(rprops.centroid)

    # Improve accuracy of edge position by smoothing the outline image and using
    # weighted averaging of pixel positions below:
    outline = filters.gaussian(outline, 0.5)

    radii = []
    for angle in angles:
        ray = np.matmul(0.5 * np.arange(np.round(2 * bbdiag))[:, nax],
                        np.array((np.cos(angle), np.sin(angle)))[nax, :])
        ray = np.round(centre + ray).astype('int')
        rr, cc = (ray[:, 0], ray[:, 1])
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

    radii = np.array(radii)

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


def draw_radial(radii, angles, centre, shape):
    mr, mc = shape
    im = np.zeros(shape, dtype='bool')
    neval = np.round(4 * np.pi * np.max(radii)).astype('int')
    if neval > 1:
        phi = np.linspace(0, 2 * np.pi, neval)
        rho = eval_radial_spline(phi, radii, angles)
    else:
        phi = 0
        rho = 0
    rr = np.round(centre[0] + rho * np.cos(phi)).astype('int')
    cc = np.round(centre[1] + rho * np.sin(phi)).astype('int')
    rr[rr < 0] = 0
    rr[rr >= mr] = mr - 1
    cc[cc < 0] = 0
    cc[cc >= mc] = mc - 1
    im[rr, cc] = True
    # valid = (rr >= 0) & (cc >= 0) & (rr < mr) & (cc < mc)
    # im[rr[valid], cc[valid]] = True
    return im


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
        radii, angles = morph_radial_thresh_fit(outline, mask, rp)
        if np.any(np.isnan(radii)):
            return mask, outline, rp
        outlines.append(draw_radial(radii, angles, rp.centroid, shape))

    return outlines


def thresh_seg(p_int,
               interior_threshold=0.5,
               connectivity=None,
               nclosing=0,
               nopening=0,
               ndilate=0,
               return_area=False):
    """Segment cell outlines from morphology output of CNN by fitting radial
    spline to threshold output
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


def single_region_prop(mask):
    return regionprops(mask.astype('int'))[0]


def outline_to_radial(outline, rprop, return_outline=False):
    coords = (rprop.centroid,) + morph_radial_thresh_fit(outline, None, rprop)
    if return_outline:
        centroid, radii, angles = coords
        outlines = draw_radial(radii, angles, centroid, outline.shape)
        return coords, outlines
    else:
        return coords


def get_edge_scores(outlines, p_edge):
    return [
        (p_edge * binary_dilation(o, iterations=2)).mean() for o in outlines
    ]


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
            for m, a in thresh_seg(p_int,
                                   interior_threshold=thresh or 0.5,
                                   nclosing=nc or 0,
                                   nopening=no or 0,
                                   ndilate=max_ne,
                                   return_area=True,
                                   connectivity=conn)
            if a >= lower and a < upper
        ]

        if len(masks_areas) > 0:
            masks, areas = zip(*masks_areas)
        else:
            masks, areas = [], []

        edges = [binary_edge(m) for m in masks]

        if fit_radial:
            rprops = [single_region_prop(m) for m in masks]
            coords, edges = list(
                zip(*[
                    outline_to_radial(edge, rprop, return_outline=True)
                    for edge, rprop in zip(edges, rprops)
                ])) or ([], [])
            masks = [binary_fill_holes(o) for o in edges]
        else:
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


def rc_to_radial(rr_cc, centre):
    """Helper function to convert row-column coords to radial coords
    """
    rr, cc = rr_cc
    rloc, cloc = centre
    rr = rr - rloc
    cc = cc - cloc
    return np.sqrt(rr**2 + cc**2), np.arctan2(cc, rr)


def prior_resid_weight(resid, gauss_scale=5, exp_scale=1):
    """Weights radial residuals to bias towards the initial guess

    Weight decays as a gaussian for positive residuals and exponentially for
    negative residuals. So assuming `resid = rho_guess - rho_initial`, then
    larger radii are favoured.
    """
    return (resid >= 0) * np.exp(-resid**2 / gauss_scale) \
            + (resid < 0) * np.exp(resid / exp_scale)


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


def refine_radial_grouped(grouped_coords, grouped_p_edges):
    """Refine initial radial spline by optimising to predicted edge

    Neighbouring groups are used to re-weight predicted edges belonging to
    other cells using the initial guess
    """

    # Determine edge pixel locations and probabilities from NN prediction
    p_edge_locs = [np.where(p_edge > 0.2) for p_edge in grouped_p_edges]
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
            rho - eval_radial_spline(phi, radii, angles)
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
            opt = least_squares(adj_rspline_resid,
                                np.zeros(nparams),
                                bounds=(-np.ones(nparams), np.ones(nparams)),
                                args=(rho, phi, p_weighted, radii, angles),
                                ftol=5e-2)

            g_opt_coords.append((centre,) +
                                adj_rspline_coords(opt.x, radii, angles))

        opt_coords.append(g_opt_coords)

    return opt_coords


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
