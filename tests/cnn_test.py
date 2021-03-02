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
import pytest
import numpy as np

from baby.preprocessing import robust_norm
from baby.utils import split_batch_pred
from baby.io import save_tiled_image

#from .conftest import IMAGE_DIR

def test_evolve_predict(bb_evolve60, imgs_evolve60, save_cnn_predictions,
                        image_dir):
    # Make a stack of evolve brightfield images
    imgstack = np.stack([robust_norm(*v['Brightfield'])
                         for v in imgs_evolve60.values()])

    preds = bb_evolve60.morph_predict(imgstack)
    assert len(preds) == len(bb_evolve60.flattener.names())
    assert all([p_out.shape[:3] == imgstack.shape[:3] for p_out in preds])

    morph_preds = split_batch_pred(preds)
    assert len(morph_preds) == len(imgstack)

    assert all([pred.max() <= 1 and pred.min() >= 0 for pred in morph_preds])

    if save_cnn_predictions:
        # Save prediction output as 16 bit tiled png
        for pred, (k, v) in zip(morph_preds, imgs_evolve60.items()):
            _, info = v['Brightfield']
            info['channel'] = 'cnnpred'
            save_tiled_image(
                ((2**16 - 1) * pred.transpose([1, 2, 0])).astype('uint16'),
                image_dir / (k + '_preds.png'), info, layout=(1, None)
            )
