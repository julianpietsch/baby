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
import numpy as np
from scipy.ndimage import binary_fill_holes

def calc_IoUs(true_segs, pred_segs, fill_holes=True):
    true_segs = [tseg > 0 for tseg in true_segs]
    pred_segs = [pseg > 0 for pseg in pred_segs]

    if fill_holes:
        true_segs = [binary_fill_holes(tseg) for tseg in true_segs]
        pred_segs = [binary_fill_holes(pseg) for pseg in pred_segs]

    nts = len(true_segs)
    nps = len(pred_segs)
    IoUs = np.zeros((nts, nps))

    for t, tseg in enumerate(true_segs):
        for p, pseg in enumerate(pred_segs):
            IoUs[t, p] = np.sum(tseg & pseg) / np.sum(tseg | pseg)

    return IoUs


def best_IoU(IoUs):
    """Return the IoUs for the best-matching pairs

    Note that this function starts with best-matching pairs, unlike what would
    be expected for the precision-recall calculations, where the predictions
    are assumed to be ordered from most confident to least confident.
    """

    IoUs = IoUs.copy() # we will make changes to IoUs
    nts, nps = IoUs.shape

    best = np.zeros(nts)
    assignments = -np.ones(nts, dtype='int')

    if nps > 0:
        for _ in range(nts):
            # Start with the best pair first
            t, p = np.unravel_index(np.argmax(IoUs), IoUs.shape)
            maxIoU =  IoUs[t, p]
            if maxIoU > 0:
                best[t] = maxIoU
                assignments[t] = p
                # This pair has now been claimed
                IoUs[:, p] = -np.Inf
                IoUs[t, :] = -np.Inf

    return best, assignments


def calc_PR(IoUs, iou_thresh=0.5):
    """Calculate precision and recall from IoU matrix

    Note that this function expects the predictions to be ordered from the one
    with highest confidence to the one with lowest confidence.
    """
    IoUs = IoUs.copy()  # we will make changes to IoUs
    nts, nps = IoUs.shape

    nTP = 0  # number of true positives
    precision = np.zeros(nps)
    recall = np.zeros(nps)
    assignments = -np.ones(nps, dtype='int')

    if nts > 0:
        for p in range(nps):
            maxind = np.argmax(IoUs[:, p])
            maxIoU = IoUs[maxind, p]
            if maxIoU > iou_thresh:
                nTP += 1
                IoUs[maxind, :] = -np.Inf  # this object has now been claimed
                assignments[p] = maxind
            precision[p] = nTP / (p + 1)  # nTP / (nTP + nFP)
            recall[p] = nTP / nts  # nTP / (nTP + nFN)

    return precision, recall, assignments


def calc_AP(IoUs, probs=None, iou_thresh=0.5):
    """Calculate the Average Precision from the matrix of IoU terms

    If probs is not specified, then the predicted segmentations are assumed to
    have been provided in order of most to least probable.
    """

    if probs is not None:
        IoUs = IoUs[:, np.argsort(-probs)]

    precision, recall, assignments = calc_PR(IoUs, iou_thresh=iou_thresh)
    nps = len(precision)

    # Make precision monotonic decreasing
    precision_mono = np.array([np.max(precision[p:]) for p in range(nps)])

    # AP is the area under the (step-wise monotonic) precision-recall curve
    AP = np.sum(precision_mono * np.diff(np.concatenate(([0], recall))))

    return AP, assignments


def edge_prob(cnn_output, segs):
    p_edge = cnn_output[0]
    return np.array([np.mean(p_edge[s]) for s in segs])


def flattener_seg_probs(cnn_output, flattener, segs):
    """Estimate the probability of segmentation from flattener CNN

    NB: ignores layers that are bud-only or layers that specify focus
    """
    valid_targets = [not d['budonly'] and not d['focus'] for d in
                     [flattener.getTargetDef(n) for n in flattener.names()]] 
    cnn_output = cnn_output[valid_targets]
    info = {'cellLabels': [1], 'buds': [0]}
    probs = []
    for seg in segs:
        segflat = flattener(binary_fill_holes(seg)[..., None], info)
        segflat = segflat[..., valid_targets].transpose((2, 0, 1))
        target_probs = [
            o[s].mean() for o, s in zip(cnn_output, segflat) if s.any()]
        probs.append(np.mean(target_probs))
    return np.array(probs)
