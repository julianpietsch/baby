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
    IoUs = IoUs.copy() # we will make changes to IoUs
    nts, nps = IoUs.shape

    best = np.zeros(nts)
    assignments = -np.ones(nts, dtype='int')

    for t in range(nts):
        maxind = np.argmax(IoUs[t, :])
        maxIoU =  IoUs[t, maxind]
        if maxIoU > 0:
            best[t] = maxIoU
            IoUs[:, maxind] = -np.Inf  # this object has now been claimed
            assignments[t] = maxind

    return best, assignments


def calc_PR(IoUs, iou_thresh=0.5):
    IoUs = IoUs.copy()  # we will make changes to IoUs
    nts, nps = IoUs.shape

    nTP = 0  # number of true positives
    precision = np.zeros(nps)
    recall = np.zeros(nps)
    assignments = -np.ones(nps, dtype='int')

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
