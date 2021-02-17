from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
from os.path import join, isdir
from time import strftime
from uuid import uuid1
import logging
import numpy as np
import pickle

from .morph_thresh_seg import SegmentationOutput
from .io import save_tiled_image

_logger = None


def _get_logger():
    global _logger
    if _logger is None:
        _logger = logging.getLogger()
    return _logger


def _generate_error_dump_id():
    return strftime('%Y-%m-%d_%H-%M-%S_') + str(uuid1())


def _segment(segmenter,
             cnn_output,
             refine_outlines,
             yield_volumes,
             yield_masks,
             yield_preds,
             yield_edgemasks,
             error_dump_dir,
             suppress_errors,
             logger=None):
    if logger is None:
        logger = _get_logger()

    output = {}

    try:
        seg_result = segmenter.segment(cnn_output,
                                       refine_outlines=refine_outlines,
                                       return_volume=yield_volumes)
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

    try:
        tracking = tracker.step_trackers(seg['masks'],
                                         seg['preds'][i_budneck],
                                         seg['preds'][i_bud],
                                         state=state,
                                         assign_mothers=assign_mothers,
                                         return_baprobs=return_baprobs)
    except:
        # Log errors
        err_id = _generate_error_dump_id()
        if (error_dump_dir is not None and isdir(error_dump_dir)):
            fprefix = join(error_dump_dir, err_id)
            masks = seg.get('masks', np.zeros((0, 0, 0)))
            if masks.size > 0:
                save_tiled_image(
                    seg['masks'].transpose((2, 0, 1)).astype('uint8'),
                    fprefix + '_masks.png')
            save_tiled_image(
                np.uint16((2**16 - 1) * seg['preds'][i_budneck, :, :, None]),
                fprefix + '_budneck_pred.png')
            save_tiled_image(
                np.uint16((2**16 - 1) * seg['preds'][i_bud, :, :, None]),
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
                'cell_label': list(range(ncells)),
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
                       refine_outlines,
                       yield_volumes,
                       yield_edgemasks,
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
                      error_dump_dir,
                      suppress_errors,
                      logger=logger)
    return _track(tracker,
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


def _segment_and_track_parallel(segmenter, tracker, flattener, morph_preds,
                                tracker_states, refine_outlines,
                                yield_volumes, yield_edgemasks,
                                assign_mothers, return_baprobs, yield_next,
                                njobs, error_dump_dir, suppress_errors):

    # logger = _get_logger()

    tnames = flattener.names()
    i_budneck = tnames.index('bud_neck')
    bud_target = 'sml_fill' if 'sml_fill' in tnames else 'sml_inte'
    i_bud = tnames.index(bud_target)

    # Run segmentation and tracking in parallel
    from joblib import Parallel, delayed
    return Parallel(n_jobs=njobs, mmap_mode='c')(
        delayed(_segment_and_track)
        (segmenter, tracker, cnn_output, state, i_budneck, i_bud,
         refine_outlines, yield_volumes, yield_edgemasks, assign_mothers,
         return_baprobs, yield_next, error_dump_dir, suppress_errors)
        for cnn_output, state in zip(morph_preds, tracker_states))
