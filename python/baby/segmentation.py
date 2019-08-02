import numpy as np
from numpy import newaxis as nax
from scipy.ndimage import (
    minimum_filter, binary_dilation, binary_closing, binary_fill_holes
)
from scipy import interpolate
from skimage.measure import label, regionprops
from skimage.segmentation import morphological_geodesic_active_contour
from skimage.morphology import diamond
from skimage.draw import ellipse_perimeter
from skimage import filters


connect_filt = diamond(1) # 3x3 filter for 1-connected patches


def mask_iou(a, b):
    """Intersection over union (IoU) between boolean masks"""
    return np.sum(a & b) / np.sum(a | b)


def mask_containment(a, b):
    """Max of intersection over a or over b"""
    return np.max(np.sum(a & b) / np.array(np.sum(a), np.sum(b)))


def morph_thresh_masks(p_interior, p_overlap=None, dilate=True,
                       interior_threshold=0.9, overlap_threshold=0.9):

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
            if nlbls>0:
                omasks = [olabels == i + 1 for i in range(nlbls)]
                omask_iou = [mask_iou(cr, r) for r in omasks]
                cr = omasks[np.argmax(omask_iou)]

        masks.append(cr)

    return masks


def unique_masks(masks, ref_masks, threshold=0.5, iou_func=mask_iou):
    """Returns a subset of masks excluding any overlaps with ref_masks"""

    rmasks = ref_masks.copy() # rmasks will be edited to remove used refs
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


def morph_thresh_seg(cnn_outputs, interior_threshold=0.9,
                     overlap_threshold=0.9, bud_threshold=0.9,
                     bud_dilate=False, bud_overlap=False, isbud_threshold=0.5):
    """Segment cell outlines from morphology output of CNN by thresholding

    Specify `overlap_threshold` or `bud_threshold` as `None` to ignore.
    """

    _, _, p_interior, p_overlap, _, p_bud = cnn_outputs

    if overlap_threshold is None:
        p_overlap = None

    if isbud_threshold is None and bud_threshold is not None:
        p_interior = p_interior * (1 - p_bud)

    masks = morph_thresh_masks(
        p_interior, interior_threshold=interior_threshold,
        p_overlap=p_overlap, overlap_threshold=overlap_threshold)

    if bud_threshold is not None:
        if not bud_overlap:
            p_overlap = None

        budmasks = morph_thresh_masks(
            p_bud, interior_threshold=bud_threshold, dilate=False,
            p_overlap=p_overlap, overlap_threshold=overlap_threshold)

        if isbud_threshold is not None:
            # Omit interior masks if they overlap with bud masks
            masks = unique_masks(masks, budmasks, iou_func=mask_containment,
                                threshold=isbud_threshold) + budmasks

    # Return only the mask outlines
    outlines = [minimum_filter(m, footprint=connect_filt) != m for m in masks]

    return outlines


def get_regions(p_img, threshold):
    """Find regions in a probability image sorted by likelihood"""
    p_thresh = p_img > interior_threshold
    p_label = label(p_thresh, background=0)
    rprops = regionprops(p_label, p_img, coordinates='rc')
    rprops = [r for r in rprops if
              r.major_axis_length > 0 and r.minor_axis_length > 0]
    rprops.sort(key=lambda x: x.mean_intensity, reverse=True)
    return rprops


def bbox_overlaps(regA, regB):
    """Returns True if the regions have overlapping bounding boxes"""
    lrA, lcA, urA, ucA = regA.bbox
    lrB, lcB, urB, ucB = regB.bbox
    rA = np.array([lrA, urA])
    cA = np.array([lcA, ucA])
    return ((not ((rA>urB).all() or (rA<lrB).all())) and
            (not ((cA>ucB).all() or (cA<lcB).all())))


def region_iou(regA, regB):
    if bbox_overlaps(regA, regB):
        bb_lr, bb_lc, _, _ = np.stack((regA.bbox, regB.bbox)).min(axis=0)
        _, _, bb_ur, bb_uc = np.stack((regA.bbox, regB.bbox)).max(axis=0)
        bboxA = np.zeros((bb_ur-bb_lr, bb_uc-bb_lc), dtype='bool')
        bboxB = bboxA.copy()
        bb = regA.bbox
        bboxA[bb[0]-bb_lr:bb[2]-bb_lr,bb[1]-bb_lc:bb[3]-bb_lc] = regA.image
        bb = regB.bbox
        bboxB[bb[0]-bb_lr:bb[2]-bb_lr,bb[1]-bb_lc:bb[3]-bb_lc] = regB.image
        return np.sum(bboxA & bboxB) / np.sum(bboxA | bboxB)
    else:
        return 0.0


def morph_ellipse_seg(cnn_outputs, interior_threshold=0.9,
                      overlap_threshold=0.9, bud_threshold=0.9,
                      bud_dilate=False, bud_overlap=False, isbud_threshold=0.5,
                      scaling=1.0, offset=0):
    """Segment cell outlines from morphology output of CNN as region ellipses

    Specify `overlap_threshold` or `bud_threshold` as `None` to ignore.
    """

    _, _, p_interior, p_overlap, _, p_bud = cnn_outputs

    if overlap_threshold is None:
        p_overlap = None

    masks = morph_thresh_masks(
        p_interior, interior_threshold=interior_threshold,
        p_overlap=p_overlap, overlap_threshold=overlap_threshold)

    if bud_threshold is not None:
        if not bud_overlap:
            p_overlap = None

        budmasks = morph_thresh_masks(
            p_bud, interior_threshold=bud_threshold, dilate=False,
            p_overlap=p_overlap, overlap_threshold=overlap_threshold)

        # Omit interior masks if they overlap with bud masks
        masks = budmasks + unique_masks(masks, budmasks, threshold=isbud_threshold)

    rprops = [regionprops(m.astype('int'), coordinates='rc')[0] for m in masks]
    rprops = [r for r in rprops if
              r.major_axis_length > 0 and r.minor_axis_length > 0]

    outlines = []
    for region in rprops:
        r, c = np.round(region.centroid).astype('int')
        r_major = np.round(scaling*region.major_axis_length/2 + offset).astype('int')
        r_minor = np.round(scaling*region.minor_axis_length/2 + offset).astype('int')
        orientation = -region.orientation
        rr, cc = ellipse_perimeter(r, c, r_major, r_minor,
                                   orientation=orientation, shape=p_interior.shape)
        outline = np.zeros(p_interior.shape, dtype='bool')
        outline[rr, cc] = True
        outlines.append(outline)

    return outlines


def get_edge_force(rprop, shape):
    r_major = rprop.major_axis_length/2
    r_minor = rprop.minor_axis_length/2
    angle = -rprop.orientation
    nr = shape[0]
    nc = shape[1]
    xmat = np.matmul(np.arange(0,nr)[:,nax], np.ones((1,nc)))
    ymat = np.matmul(np.ones((nr,1)), np.arange(0,nc)[nax,:])
    xy = np.vstack([np.reshape(xmat, (1,-1)), np.reshape(ymat, (1,-1))])
    rotmat = np.array([[np.cos(angle), -np.sin(angle)],[np.sin(angle), np.cos(angle)]])
    radial_index = np.matmul(rotmat, (xy - np.array(rprop.centroid)[:,nax])) / np.array([r_major, r_minor])[:,nax]
    return np.reshape(1-np.exp(-np.sum((radial_index)**2, 0)), (nr,nc))


def morph_ac_seg(cnn_outputs, interior_threshold=0.9,
                 overlap_threshold=0.9, bud_threshold=0.9,
                 bud_dilate=False, bud_overlap=False, isbud_threshold=0.5,
                 use_edge_force=True, ac_edge_threshold = 0.4,
                 ac_overlap_threshold = 0.4, iterations = 5,
                 balloon_force = 0.5, smoothing = 2):

    p_edge, _, p_interior, p_overlap, _, p_bud = cnn_outputs

    p_over = p_overlap # guarantee existence for AC methods

    if overlap_threshold is None:
        p_overlap = None

    masks = morph_thresh_masks(
        p_interior, interior_threshold=interior_threshold,
        p_overlap=p_overlap, overlap_threshold=overlap_threshold)

    outmasks = []
    if bud_threshold is not None:
        if not bud_overlap:
            p_overlap = None

        budmasks = morph_thresh_masks(
            p_bud, interior_threshold=bud_threshold, dilate=False,
            p_overlap=p_overlap, overlap_threshold=overlap_threshold)

        # Omit interior masks if they overlap with bud masks
        masks = unique_masks(masks, budmasks, threshold=isbud_threshold)

        outmasks += budmasks

    rprops = [regionprops(m.astype('int'), coordinates='rc')[0] for m in masks]
    rprops = [r for r in rprops if
              r.major_axis_length > 0 and r.minor_axis_length > 0]

    e_forces = [get_edge_force(r, p_edge.shape) for r in rprops]
    edge_thresh = p_edge > ac_edge_threshold
    over_thresh = p_over > ac_overlap_threshold

    for j, (force, mask) in enumerate(zip(e_forces, masks)):
        ij_edge_im = p_edge.copy()

        if use_edge_force:
            other_forces = e_forces.copy()
            other_forces.pop(j)
            other_forces = np.sum(np.array(other_forces), 0)
            ef = (force + other_forces)/len(e_forces)
            ij_edge_im[edge_thresh] = ef[edge_thresh]
            ij_edge_im[over_thresh] = force[over_thresh]

        final_mask = morphological_geodesic_active_contour(
            1 - ij_edge_im, iterations, mask, balloon=balloon_force,
            smoothing=smoothing
        )

        outmasks.append(final_mask)

    # Return only the mask outlines
    outlines = [minimum_filter(m, footprint=connect_filt) != m for m in outmasks]

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
    rho = np.concatenate((rho, rho[0,nax]))
    phi = np.concatenate((phi - offset, (2*np.pi,)))

    tck = interpolate.splrep(phi, rho, per=True)
    return interpolate.splev(np.mod(x - offset, 2*np.pi), tck)


def morph_radial_thresh_fit(outline, mask=None, rprops=None):
    if mask is None and rprops is None:
        mask = binary_fill_holes(outline)

    if rprops is None:
        rprops = regionprops(mask.astype('int'), coordinates='rc')[0]

    r_maj = rprops.major_axis_length
    nrays = 4 if r_maj < 5 else 6 if r_maj < 20 else 8

    RL, CL, RU, CU = rprops.bbox
    bbdiag = np.sqrt((RU - RL)**2 + (CU - CL)**2)
    rr_max, cc_max = outline.shape

    astep = 2*np.pi/nrays
    angles = np.mod(rprops.orientation + np.pi, astep) + \
        np.arange(nrays)*astep - np.pi
    centre = np.array(rprops.centroid)

    # Improve accuracy of edge position by smoothing the outline image and using
    # weighted averaging of pixel positions below:
    outline = filters.gaussian(outline, 0.5)

    radii = []
    for angle in angles:
        ray = np.matmul(0.5*np.arange(np.round(2 * bbdiag))[:,nax],
                        np.array((np.cos(angle), np.sin(angle)))[nax,:])
        ray = np.round(centre + ray).astype('int')
        rr, cc = (ray[:,0], ray[:,1])
        ray = ray[(rr>=0) & (rr<rr_max) & (cc>=0) & (cc<cc_max),:]

        edge_pix = np.flatnonzero(np.squeeze(outline[ray[:,0], ray[:,1]]) > 0.01)
        ray = ray[edge_pix, :]
        edge_pix = np.average(ray, weights=outline[ray[:,0], ray[:,1]], axis=0)
        radii.append(np.sqrt(np.sum((edge_pix-centre)**2)))

    return np.array(radii), angles


def draw_radial(radii, angles, centre, shape):
    im = np.zeros(shape, dtype='bool')
    phi = np.linspace(0, 2*np.pi, np.round(4*np.pi*np.max(radii)).astype('int'))
    rho = eval_radial_spline(phi, radii, angles)
    rr = np.round(centre[0] + rho * np.cos(phi)).astype('int')
    cc = np.round(centre[1] + rho * np.sin(phi)).astype('int')
    im[rr, cc] = True
    return im


def morph_radial_thresh_seg(cnn_outputs, interior_threshold=0.9,
                            overlap_threshold=0.9, bud_threshold=0.9,
                            bud_dilate=False, bud_overlap=False,
                            isbud_threshold=0.5):
    """Segment cell outlines from morphology output of CNN by fitting radial
    spline to threshold output

    Specify `overlap_threshold` or `bud_threshold` as `None` to ignore.
    """

    _, _, p_interior, p_overlap, _, p_bud = cnn_outputs

    shape = p_interior.shape

    if overlap_threshold is None:
        p_overlap = None

    if isbud_threshold is None and bud_threshold is not None:
        p_interior = p_interior * (1 - p_bud)

    masks = morph_thresh_masks(
        p_interior, interior_threshold=interior_threshold,
        p_overlap=p_overlap, overlap_threshold=overlap_threshold)

    if bud_threshold is not None:
        if not bud_overlap:
            p_overlap = None

        budmasks = morph_thresh_masks(
            p_bud, interior_threshold=bud_threshold, dilate=False,
            p_overlap=p_overlap, overlap_threshold=overlap_threshold)

        if isbud_threshold is not None:
            # Omit interior masks if they overlap with bud masks
            masks = budmasks + unique_masks(masks, budmasks, threshold=isbud_threshold)

    # Need mask outlines and region properties
    mseg = [minimum_filter(m, footprint=connect_filt) != m for m in masks]
    rprops = [regionprops(m.astype('int'), coordinates='rc')[0] for m in masks]

    outlines = []
    for mask, outline, rp in zip(masks, mseg, rprops):
        radii, angles = morph_radial_thresh_fit(outline, mask, rp)
        if np.any(np.isnan(radii)):
            return mask, outline, rp
        outlines.append(draw_radial(radii, angles, rp.centroid, shape))

    return outlines


def morph_radial_edge_seg(cnn_outputs):
    RL, CL, RU, CU = rp.bbox
    Rext = np.ceil(0.25 * (RU - RL)).astype('int')
    Cext = np.ceil(0.25 * (CU - CL)).astype('int')
    RL = np.max([RL - Rext, 0])
    RU = np.min([RU + Rext, pedge.shape[0]])
    CL = np.max([CL - Cext, 0])
    CU = np.min([CU + Cext, pedge.shape[1]])

    cell_p_edge = p_edge[RL:RU,CL:CU]
    rr, cc = np.where(cell_p_edge>0.3)
    prob = cell_p_edge[rr, cc]
    rr = rr - (rp.centroid[0] - RL)
    cc = cc - (rp.centroid[1] - CL)
    phi = np.arctan2(cc, rr) - rp.orientation
    rho = np.sqrt(rr**2+cc**2)

    ac_seg = np.zeros((RU-RL, CU-CL), dtype='bool')
    ac_phi = np.linspace(0, 2*np.pi, np.round(4*np.pi*np.max(radii)).astype('int'))
    ac_rho = eval_radial_spline(ac_phi, radii, angles)
    ac_rr = np.round(rp.centroid[0] - RL + ac_rho*np.cos(ac_phi)).astype('int')
    ac_cc = np.round(rp.centroid[1] - CL + ac_rho*np.sin(ac_phi)).astype('int')
    ac_seg[ac_rr, ac_cc] = True

