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
from matplotlib.colors import Normalize
from matplotlib import pyplot as plt


def colour_seg(seg, rgb=(0.5, 0, 0.5), alpha=1):
    seg = (seg > 0).astype('float')
    img = np.zeros(seg.shape + (4,))
    for i, c in enumerate(rgb):
        img[:, :, i] = c * seg
    img[:, :, -1] = alpha * seg
    return img


def colour_segstack(seg_stack, cmap=plt.cm.jet, alpha=0.7):
    """Convert a stack of logical masks to coloured images for display"""
    seg_enum = (seg_stack >
                0) * (1 + np.arange(seg_stack.shape[2], dtype=np.uint16))
    seg_flat = np.apply_along_axis(np.max, 2, seg_enum)
    seg_show = Normalize()(seg_flat)
    seg_show = cmap(seg_show)
    seg_show[:, :, -1] = alpha * (seg_flat > 0)
    return seg_show


def plot_ims(ims, size=4, cmap=plt.cm.gray, **kwargs):
    ncols = len(ims)
    plt.figure(figsize=(ncols * size, size))
    for i in range(ncols):
        plt.subplot(1, ncols, i + 1)
        plt.imshow(np.squeeze(ims[i]), cmap=cmap, **kwargs)
        plt.grid(False)
        plt.xticks([])
        plt.yticks([])
    plt.show()


def plot_IoU(target_seg, ref_segs, index):
    if index < 0:
        match = np.zeros_like(target_seg)
        others = ref_segs + [np.zeros_like(target_seg)]
    else:
        match = ref_segs[index]
        others = ref_segs[:index] + ref_segs[index + 1:]

    others = np.max(np.stack(others), axis=0)

    plt.imshow(colour_seg(others, rgb=(0, 0.3, 0), alpha=0.3))
    plt.imshow(colour_seg(match, rgb=(0, 0, 1), alpha=0.5))
    plt.imshow(colour_seg(target_seg, rgb=(1, 0, 0), alpha=0.5))
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])


def plot_overlaps(model, inputim, target, output):
    lblind = model.output_names.index(output)
    target = target[lblind][..., 0] > 0
    pred = model.predict(inputim)[lblind]
    pred_bin = np.squeeze(pred) > 0.5
    correct = pred_bin & target
    falsepos = pred_bin & ~target
    falseneg = ~pred_bin & target

    ncols = len(inputim)
    plt.figure(figsize=(ncols * 4, 4))
    for i in range(ncols):
        plt.subplot(1, ncols, i + 1)
        plt.imshow(np.squeeze(inputim[i, :, :, 0]), cmap='gray')
        plt.imshow(colour_seg(correct[i], rgb=(0, 1, 0), alpha=0.3))
        plt.imshow(colour_seg(falsepos[i], rgb=(0, 0, 1), alpha=0.3))
        plt.imshow(colour_seg(falseneg[i], rgb=(1, 0, 0), alpha=0.3))
        plt.grid(False)
        plt.xticks([])
        plt.yticks([])
    plt.show()


def plot_best_or_worst(outlist):
    ncols = len(outlist)
    plt.figure(figsize=(ncols * 4, 4))
    for i in range(ncols):
        iou, batch_ind, pred, target, inputim = outlist[i]

        correct = pred & target
        falsepos = pred & ~target
        falseneg = ~pred & target

        plt.subplot(1, ncols, i + 1)
        plt.imshow(np.squeeze(inputim[:, :, 0]), cmap='gray')
        plt.imshow(colour_seg(correct, rgb=(0, 1, 0), alpha=0.3))
        plt.imshow(colour_seg(falsepos, rgb=(0, 0, 1), alpha=0.3))
        plt.imshow(colour_seg(falseneg, rgb=(1, 0, 0), alpha=0.3))
        plt.grid(False)
        plt.xticks([])
        plt.yticks([])
        plt.title('IoU: {:.2f}; Batch: {}'.format(iou, batch_ind))
    plt.show()
