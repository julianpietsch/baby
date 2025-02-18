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
import pytest
from packaging.version import Version
from pathlib import Path
from functools import partial
import pickle
import json
import numpy as np
import sklearn

from baby.utils import jsonify_sklearn_model, as_sklearn_model
from baby.utils import save_sklearn_model, load_sklearn_model


MODELS_DIR = Path(__file__).parent / 'sklearn-test-data'
PKL_MODEL_FILES = {
    'main_tracker': 'ct_rf_20210201_12.pkl',
    'bak_tracker': 'ct_rf_20210125_9.pkl',
    'mb_assign': 'prime_ovlp_bf_60x_5z_20221219_mb_model.pkl'
}
NPZ_MODEL_FILES = {
    'main_tracker': 'ct_rf_20210201_12.npz',
    'bak_tracker': 'ct_rf_20210125_9.npz',
    'mb_assign': 'prime_ovlp_bf_60x_5z_20221219_mb_model.npz'
}
TEST_DATA_FILES = {
    'main_tracker': 'test-data-ct_rf_20210201_12.npz',
    'bak_tracker': 'test-data-ct_rf_20210125_9.npz',
    'mb_assign': 'test-data-prime_ovlp_bf_60x_5z_20221219_mb_model.npz'
}


if_old_sklearn = pytest.mark.skipif(
    Version(sklearn.__version__) > Version('1.2.2'),
    reason='version of Scikit-learn cannot exceed 1.2.2 for pickled saves'
)


@if_old_sklearn
def test_pkl_matches():
    for k in PKL_MODEL_FILES:
        pkl_file = MODELS_DIR / PKL_MODEL_FILES[k]
        with open(pkl_file, 'rb') as f:
            pkl_model = pickle.load(f)

        loader = np.load(MODELS_DIR / TEST_DATA_FILES[k])
        X, y = loader['X'], loader['y']

        assert np.allclose(pkl_model.predict_proba(X)[:, 1], y)


@if_old_sklearn
def test_pkl_to_npz():
    for k in PKL_MODEL_FILES:
        pkl_file = MODELS_DIR / PKL_MODEL_FILES[k]
        with open(pkl_file, 'rb') as f:
            pkl_model = pickle.load(f)

        npz_dict, npz_arrays = jsonify_sklearn_model(pkl_model)
        npz_json = json.dumps(npz_dict)
        assert all([isinstance(v, np.ndarray) for v in npz_arrays.values()])
        npz_model = json.loads(npz_json,
                               object_hook=partial(as_sklearn_model,
                                                   arrays=npz_arrays))

        loader = np.load(MODELS_DIR / TEST_DATA_FILES[k])
        X, y = loader['X'], loader['y']

        assert np.allclose(npz_model.predict_proba(X)[:, 1], y)


@if_old_sklearn
def test_npz_save_load(tmp_path, save_sklearn_pkls_as_npz):
    for k in PKL_MODEL_FILES:
        pkl_file = MODELS_DIR / PKL_MODEL_FILES[k]
        with open(pkl_file, 'rb') as f:
            pkl_model = pickle.load(f)

        savedir = MODELS_DIR if save_sklearn_pkls_as_npz else tmp_path
        filename = savedir / pkl_file.with_suffix('.npz').name

        save_sklearn_model(pkl_model, filename)
        npz_model = load_sklearn_model(filename)

        loader = np.load(MODELS_DIR / TEST_DATA_FILES[k])
        X, y = loader['X'], loader['y']

        assert np.allclose(npz_model.predict_proba(X)[:, 1], y)


def test_npz_matches():
    for k, filename in NPZ_MODEL_FILES.items():
        npz_model = load_sklearn_model(MODELS_DIR / filename)
        loader = np.load(MODELS_DIR / TEST_DATA_FILES[k])
        X, y = loader['X'], loader['y']
        assert np.allclose(npz_model.predict_proba(X)[:, 1], y)
