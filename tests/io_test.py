import pytest
import numpy as np

from baby.io import save_tiled_image, load_tiled_image

def test_tiled_save_load(tmp_path):
    img = np.zeros((32, 32, 3), dtype='uint16')
    info = {'experimentID': 'test_experiment'}

    testfile = tmp_path / 'test_image.png'
    save_tiled_image(img, testfile, info=info)
    loaded_img, loaded_info = load_tiled_image(testfile)

    assert (loaded_img == img).all()
    assert 'experimentID' in loaded_info
    assert loaded_info['experimentID'] == info['experimentID']
    assert info['layout'] == (1, 3)

