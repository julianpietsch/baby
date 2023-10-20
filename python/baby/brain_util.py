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
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
from os.path import join, isdir
from time import strftime
from uuid import uuid1
import logging
from itertools import islice
import numpy as np
import pickle
from scipy.ndimage import binary_dilation, binary_fill_holes
from skimage import transform
from skimage.morphology import diamond

from .morph_thresh_seg import SegmentationOutput
from .segmentation import draw_radial
from .io import save_tiled_image
from .errors import Clogging, BadParam
from .preprocessing import robust_norm, robust_norm_dw

_logger = None


def _get_logger():
    global _logger
    if _logger is None:
        # TODO: share logging to the main thread using something like a
        # multiprocessing Manager...
        _logger = logging.getLogger('brain_logger')
        _logger.propagate = False
        lfh = logging.StreamHandler()
        lfh.setLevel(logging.ERROR)
        lff = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        lfh.setFormatter(lff)
        _logger.addHandler(lfh)
    return _logger


def _generate_error_dump_id():
    return strftime('%Y-%m-%d_%H-%M-%S_') + str(uuid1())


def _batch_generator(iterable, n=8):
    """Yields batches from an iterator
    Based on
    https://docs.python.org/3/library/itertools.html#itertools-recipes"""
    it = iter(iterable)
    while True:
        batch = list(islice(it, n))
        if not batch:
            return
        yield np.array(batch)


def _tile_generator(imgarray,
                    overlap_size=48,
                    tile_sizes=np.arange(64, 257, 8)):
    """Yields optimal tiles from an image."""

    imH, imW = imgarray.shape[1:3]

    if imH * imW <= np.square(np.max(tile_sizes)):
        # If area is less than that of max tile size we will simply return
        # images without modification
        nH, nW = 1, 1
        tileH, tileW = imH, imW
    else:
        tile_sizes = tile_sizes[tile_sizes > overlap_size + 32]
        if len(tile_sizes) == 0:
            raise BadParam('`overlap_size` is too large for the ' +
                           'specified tile_sizes')

        # For each candidate tile size, calculate the required number of tiles
        # per image (rounding up), including allowance for overlaps
        nH = np.ceil((imH - overlap_size)/(tile_sizes - overlap_size)).astype(int)
        nW = np.ceil((imW - overlap_size)/(tile_sizes - overlap_size)).astype(int)

        # Calculate the new total size given the rounding above
        paddedH = nH * (tile_sizes - overlap_size) + overlap_size
        paddedW = nW * (tile_sizes - overlap_size) + overlap_size

        # Choose the tile size that minimises the fractional padding
        padH = paddedH - imH
        i_best =  np.argmin(padH / tile_sizes)
        nH, tileH = nH[i_best], tile_sizes[i_best]
        padH = padH[i_best]
        padW = paddedW - imW
        i_best =  np.argmin(padW / tile_sizes)
        nW, tileW = nW[i_best], tile_sizes[i_best]
        padW = padW[i_best]

        # Pad image to tiled size
        imgarray = np.pad(
            imgarray, ((0, 0), (0, padH), (0, padW), (0, 0)))

    # First yield tiling details for future stitching
    yield nH, nW, imH, imW, overlap_size

    # Split the images up into tiles
    for img in imgarray:
        for j in range(nH):
            for i in range(nW):
                Hl = j * (tileH - overlap_size)
                Hu = Hl + tileH
                Wl = i * (tileW - overlap_size)
                Wu = Wl + tileW
                yield img[Hl:Hu, Wl:Wu, ...]


def _stitch_tiles(tilestream, nH, nW, imH, imW, nOvlap):
    """Stitches tiles back together"""

    # Regions where tiles overlap use linear decay for weighted averaging
    lindecay = np.linspace(1, 0, nOvlap + 2)[1:-1]
    Hdecay = np.c_[lindecay][..., None]
    Wdecay = np.c_[lindecay].T[..., None]

    # Batch tiles into each complete image
    for tiles in zip(*[iter(tilestream)] * (nW * nH)):
        # Transpose tiles so that we can stitch with h/vstack
        tiles = [tile.transpose((1, 2, 0)) for tile in tiles]

        img = None
        # Batch tiles into rows at a time
        for row in zip(*[iter(tiles)]*nW):
            # Stitch tiles into a row
            rowimg = row[0]
            for t in row[1:]:
                rowimg[:, -nOvlap:] = (rowimg[:, -nOvlap:] * Wdecay
                                       + t[:, :nOvlap] * Wdecay[:, ::-1])
                rowimg = np.hstack((rowimg, t[:, nOvlap:]))

            # Stitch rows together
            if img is None:
                img = rowimg
            else:
                img[-nOvlap:, :] = (img[-nOvlap:, :] * Hdecay +
                                    rowimg[:nOvlap, :] * Hdecay[::-1, :])
                img = np.vstack((img, rowimg[nOvlap:, :]))

        yield img[:imH, :imW, :].transpose((2, 0, 1))


def _apply_preprocessing(bf_imgs, input_norm_dw, pxsize_in, pxsize_out):
    input_norm = robust_norm_dw if input_norm_dw else robust_norm
    bf_imgs = np.stack([input_norm(img, {}) for img in bf_imgs])
    inshape = bf_imgs.shape
    rescaling = None
    if pxsize_in is not None and pxsize_in != pxsize_out:
        rescaling = pxsize_in / pxsize_out
    if rescaling is not None:
        bf_imgs = list(bf_imgs)
        for i, img in enumerate(bf_imgs):
            bf_imgs[i] = transform.rescale(img,
                                           rescaling,
                                           order=1,
                                           channel_axis=2)
        bf_imgs = np.stack(bf_imgs)
        rescaling = 1. / rescaling
    return bf_imgs, rescaling, inshape[1:3]


def _rescale_output(output, rescaling, outshape, cartesian_spline):
    if rescaling is None:
        return

    if 'centres' in output:
        output['centres'] = [[x * rescaling for x in cell]
                             for cell in output['centres']]
    if 'radii' in output:
        output['radii'] = [[r * rescaling for r in cell]
                           for cell in output['radii']]
    has_coords = {'centres', 'angles', 'radii'}.issubset(output.keys())
    if has_coords and ('edgemasks' in output or 'masks' in output):
        edgemasks = np.stack([
            draw_radial(np.array(r), np.array(a), c, outshape,
                        cartesian_spline=cartesian_spline)
            for r, a, c in zip(output['radii'], output['angles'],
                               output['centres'])
        ])
    _0xy = (0,) + outshape
    if 'edgemasks' in output:
        if output['edgemasks'].shape[0] == 0:
            output['edgemasks'] = np.zeros(_0xy, dtype=bool)
        elif has_coords:
            output['edgemasks'] = edgemasks
        else:
            output['edgemasks'] = transform.resize(
                output['edgemasks'], (len(output['edgemasks']),) + outshape, order=0)
    if 'masks' in output:
        if output['masks'].shape[0] == 0:
            output['masks'] = np.zeros(_0xy, dtype=bool)
        elif has_coords:
            output['masks'] = binary_fill_holes(
                edgemasks, diamond(1)[None, ...])
        else:
            output['masks'] = transform.resize(
                output['masks'], (len(output['masks']),) + outshape, order=0)
    if 'preds' in output:
        output['preds'] = transform.resize(
            output['preds'], (len(output['preds']),) + outshape, order=1)
    if 'volumes' in output:
        vscaling = rescaling ** 3
        output['volumes'] = [v * vscaling for v in output['volumes']]


def _segment(segmenter,
             cnn_output,
             refine_outlines,
             yield_volumes,
             yield_masks,
             yield_preds,
             yield_edgemasks,
             clogging_thresh,
             error_dump_dir,
             suppress_errors,
             logger=None):
    if logger is None:
        logger = _get_logger()

    output = {}

    try:
        # Check if clogging has occurred in this image
        flat_interior = []
        for group in segmenter.groups:
            g_interior = group.prediction(cnn_output, 'interior', True)
            g_interior = g_interior > group.interior_threshold
            if group.max_n_erode > 0:
                g_interior = binary_dilation(g_interior,
                                             iterations=group.max_n_erode)
            flat_interior.append(g_interior)
        flat_interior = np.stack(flat_interior).max(axis=0)
        if flat_interior.sum() / flat_interior.size > clogging_thresh:
            raise Clogging('clogging threshold exceeded')

        seg_result = segmenter.segment(cnn_output,
                                       refine_outlines=refine_outlines,
                                       return_volume=yield_volumes)
    except Clogging:
        if suppress_errors:
            seg_result = SegmentationOutput([])
            output['error'] = 'Segmentation error: clogging threshold exceeded'
            output['clogging'] = True
        else:
            raise
    except:
        # Log errors
        err_id = _generate_error_dump_id()
        if (error_dump_dir is not None and isdir(error_dump_dir)):
            save_tiled_image(np.uint16((2**16 - 1) * cnn_output),
                             join(error_dump_dir, err_id + '_cnn_out.png'))
        if suppress_errors:
            err_msg = dict(ID=err_id,
                           refine_outlines=refine_outlines,
                           return_volume=yield_volumes)
            err_msg = ['{}: {}'.format(k, v) for k, v in err_msg.items()]
            err_msg.insert(0, 'Segmentation error')
            logger.exception('\n'.join(err_msg))
            seg_result = SegmentationOutput([])
            output['error'] = 'Segmentation error: ' + err_id
        else:
            raise

    if len(seg_result.coords) > 0:
        centres, radii, angles = zip(*seg_result.coords)
    else:
        centres, radii, angles = 3 * [[]]

    # Return output as a dict
    output.update({
        'centres': [list(c) for c in centres],
        'angles': [a.tolist() for a in angles],
        'radii': [r.tolist() for r in radii]
    })

    _0xy = (0,) + cnn_output.shape[1:3]
    if yield_masks:
        if len(seg_result.masks) > 0:
            output['masks'] = np.stack(seg_result.masks)
        else:
            output['masks'] = np.zeros(_0xy, dtype='bool')
    if yield_edgemasks:
        if len(seg_result.edges) > 0:
            output['edgemasks'] = np.stack(seg_result.edges)
        else:
            output['edgemasks'] = np.zeros(_0xy, dtype='bool')
    if yield_preds:
        output['preds'] = cnn_output
    if yield_volumes:
        output['volumes'] = seg_result.volume

    return output


def _track(tracker,
           seg,
           state,
           i_budneck,
           i_bud,
           assign_mothers,
           return_baprobs,
           yield_edgemasks,
           yield_next,
           error_dump_dir,
           suppress_errors,
           logger=None):
    if logger is None:
        logger = _get_logger()

    if i_budneck is None:
        p_budneck = np.zeros(seg['preds'].shape[1:])
    else:
        p_budneck = seg['preds'][i_budneck]
    if i_bud is None:
        p_bud = np.zeros(seg['preds'].shape[1:])
    else:
        p_bud = seg['preds'][i_bud]
    try:
        tracking = tracker.step_trackers(seg['masks'],
                                         p_budneck,
                                         p_bud,
                                         state=state,
                                         assign_mothers=assign_mothers,
                                         return_baprobs=return_baprobs)
    except:
        # Log errors
        err_id = _generate_error_dump_id()
        masks = seg.get('masks', np.zeros((0, 0, 0)))
        if (error_dump_dir is not None and isdir(error_dump_dir)):
            fprefix = join(error_dump_dir, err_id)
            if masks.size > 0:
                save_tiled_image(
                    seg['masks'].transpose((1, 2, 0)).astype('uint8'),
                    fprefix + '_masks.png')
            save_tiled_image(
                np.uint16((2**16 - 1) * p_budneck[..., None]),
                fprefix + '_budneck_pred.png')
            save_tiled_image(
                np.uint16((2**16 - 1) * p_bud[..., None]),
                fprefix + '_bud_pred.png')
            with open(fprefix + '_state.pkl', 'wb') as f:
                pickle.dump(state, f)
        if suppress_errors:
            err_msg = dict(ID=err_id,
                           assign_mothers=assign_mothers,
                           return_baprobs=return_baprobs)
            err_msg = ['{}: {}'.format(k, v) for k, v in err_msg.items()]
            err_msg.insert(0, 'Tracking error')
            logger.exception('\n'.join(err_msg))
            ncells = len(masks)
            tracking = {
                'cell_label': list(range(1, ncells + 1)),
                'mother_assign': [],
                'p_bud_assign': np.zeros((ncells, ncells)).tolist(),
                'state': state,
            }

            seg['error'] = 'Tracking error: ' + err_id
        else:
            raise

    del seg['preds']
    del seg['masks']
    if not yield_edgemasks and 'edgemasks' in seg:
        del seg['edgemasks']

    seg['cell_label'] = tracking['cell_label']
    state = tracking['state']

    # Use the last added "previous features" in the tracker state to
    # obtain the major and minor ellipsoid axes
    if state is None:
        feats = np.zeros((0, len(tracker.outfeats)))
    else:
        feats = state['prev_feats'][-1]
    ellip_inds = tuple(
        tracker.outfeats.index(p)
        for p in ('major_axis_length', 'minor_axis_length'))
    if feats.shape[0] > 0:
        seg['ellipse_dims'] = feats[:, ellip_inds].tolist()
    else:
        seg['ellipse_dims'] = []

    if assign_mothers:
        seg['mother_assign'] = tracking['mother_assign']

    if return_baprobs:
        seg['p_bud_assign'] = tracking['p_bud_assign']

    if yield_next:
        return seg, state
    else:
        return seg


def _segment_and_track(segmenter,
                       tracker,
                       cnn_output,
                       state,
                       i_budneck,
                       i_bud,
                       rescaling,
                       outshape,
                       refine_outlines,
                       yield_volumes,
                       yield_edgemasks,
                       clogging_thresh,
                       assign_mothers,
                       return_baprobs,
                       yield_next,
                       error_dump_dir,
                       suppress_errors,
                       logger=None):
    if logger is None:
        logger = _get_logger()

    segout = _segment(segmenter,
                      cnn_output,
                      refine_outlines,
                      yield_volumes,
                      True,
                      True,
                      yield_edgemasks,
                      clogging_thresh,
                      error_dump_dir,
                      suppress_errors,
                      logger=logger)
    trackout = _track(tracker,
                      segout,
                      state,
                      i_budneck,
                      i_bud,
                      assign_mothers,
                      return_baprobs,
                      yield_edgemasks,
                      yield_next,
                      error_dump_dir,
                      suppress_errors,
                      logger=logger)

    if yield_next:
        segout, state = trackout
        _rescale_output(segout, rescaling, outshape,
                        segmenter.params.cartesian_spline)
        return segout, state
    else:
        _rescale_output(trackout, rescaling, outshape,
                        segmenter.params.cartesian_spline)
        return trackout


def _segment_and_track_parallel(segmenter, tracker, flattener, morph_preds,
                                tracker_states, rescaling, outshape,
                                refine_outlines, yield_volumes,
                                yield_edgemasks, clogging_thresh,
                                assign_mothers, return_baprobs, yield_next,
                                njobs, error_dump_dir, suppress_errors):

    tnames = flattener.names()
    bud_target = 'sml_fill' if 'sml_fill' in tnames else 'sml_inte'
    assign_buds = 'bud_neck' in tnames and bud_target in tnames
    if assign_buds:
        i_budneck = tnames.index('bud_neck')
        i_bud = tnames.index(bud_target)
    else:
        i_budneck = None
        i_bud = None

    # Run segmentation and tracking in parallel
    from joblib import Parallel, delayed
    return Parallel(n_jobs=njobs, mmap_mode='c')(
        delayed(_segment_and_track)
        (segmenter, tracker, cnn_output, state, i_budneck, i_bud,
         rescaling, outshape, refine_outlines, yield_volumes, yield_edgemasks,
         clogging_thresh, assign_mothers, return_baprobs, yield_next,
         error_dump_dir, suppress_errors)
        for cnn_output, state in zip(morph_preds, tracker_states))
