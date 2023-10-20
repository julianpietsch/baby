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
from pathlib import Path
import os

from baby import BabyBrain


DEFAULT_MODELSET = 'yeast-alcatras-brightfield-sCMOS-60x-5z'


@pytest.fixture(scope='module')
def remote_modelsets(modelsets):
    return modelsets.remote_modelsets()


def test_remote_modelsets_available(remote_modelsets):
    assert 'models' in remote_modelsets
    assert 'shared' in remote_modelsets


def test_local_cache_dir(modelsets):
    localpath = modelsets.LOCAL_MODELSETS_PATH
    envpath = os.environ.get(modelsets.ENV_VAR_MODELSETS_PATH)
    if not envpath:
        assert localpath != modelsets.DEFAULT_LOCAL_MODELSETS_PATH
    else:
        assert localpath == Path(envpath)
    modelsets._ensure_local_path()
    assert localpath.exists()


def test_meta(modelsets, remote_modelsets):
    modelsets.ids() # trigger creation of local model sets cache
    assert modelsets.LOCAL_MODELSETS_CACHE.exists()

    # Listed models match those on remote
    remote_keys = remote_modelsets['models'].keys()
    assert len(set(modelsets.ids()).difference(remote_keys)) == 0

    # Listed meta data matches that on remote
    remote_meta = {k: v['meta'] for k, v in
                   remote_modelsets['models'].items()}
    assert modelsets.meta() == remote_meta


def test_update(modelsets, remote_modelsets):
    remote_default = remote_modelsets['models'][DEFAULT_MODELSET]
    modelsets.update([DEFAULT_MODELSET])
    localpath = modelsets.LOCAL_MODELSETS_PATH / DEFAULT_MODELSET
    assert localpath.is_dir()
    for filename in remote_default['files']:
        assert (localpath / filename).is_file()


def test_get_params(modelsets, remote_modelsets):
    remote_default = remote_modelsets['models'][DEFAULT_MODELSET]
    local_params = modelsets.get_params(DEFAULT_MODELSET)
    assert local_params == remote_default['brain_params']
    localpath = modelsets.LOCAL_MODELSETS_PATH / DEFAULT_MODELSET
    print(localpath)

    # Test auto-update for missing model set file
    (localpath / modelsets.MODELSET_FILENAME).unlink()
    modelsets.get_params(DEFAULT_MODELSET)
    assert (localpath / modelsets.MODELSET_FILENAME).exists()

    # Test auto-update for missing model file
    target_file = [f for f in remote_default['files']
                   if f != modelsets.MODELSET_FILENAME][0]
    target_file = localpath / target_file
    target_file.unlink()
    modelsets.get_params(DEFAULT_MODELSET)
    assert target_file.exists()


def test_resolve(modelsets):
    params = modelsets.get_params(DEFAULT_MODELSET)
    localpath = modelsets.LOCAL_MODELSETS_PATH / DEFAULT_MODELSET
    sharepath = modelsets.LOCAL_MODELSETS_PATH / modelsets.SHARE_PATH
    localtest = params['morph_model_file']
    sharetest = params['celltrack_model_file']
    assert (localpath / localtest).is_file()
    assert (sharepath / sharetest).is_file()
    assert ((localpath / localtest) == 
            modelsets.resolve(localtest, DEFAULT_MODELSET))
    assert ((sharepath / sharetest) == 
            modelsets.resolve(sharetest, DEFAULT_MODELSET))


def test_get(modelsets):
    bb = modelsets.get(DEFAULT_MODELSET)
    assert type(bb) == BabyBrain


