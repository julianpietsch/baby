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
