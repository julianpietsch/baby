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
# SOFTWARE.
import os
from pathlib import Path
import requests
from urllib.parse import urljoin, quote
import json

from .errors import BadParam, BadModel, BadFile
from .utils import jsonify, as_python_object
from .morph_thresh_seg import SegmentationParameters

MODELSETS_FILENAME = 'modelsets.json'
MODELSET_FILENAME = 'modelset.json'

BABY_MODELS_URL = 'https://julianpietsch.github.io/baby-models/'
MODELSETS_URL = urljoin(BABY_MODELS_URL, MODELSETS_FILENAME)

ENV_VAR_MODELSETS_PATH = 'BABY_MODELSETS_PATH'
ENV_LOCAL_MODELSETS_PATH = os.environ.get(ENV_VAR_MODELSETS_PATH)
if (type(ENV_LOCAL_MODELSETS_PATH) == str and
    ENV_LOCAL_MODELSETS_PATH.startswith('~')):
    ENV_LOCAL_MODELSETS_PATH = Path(ENV_LOCAL_MODELSETS_PATH).expanduser()
DEFAULT_LOCAL_MODELSETS_PATH = Path.home() / '.baby_models'
LOCAL_MODELSETS_PATH = ENV_LOCAL_MODELSETS_PATH or DEFAULT_LOCAL_MODELSETS_PATH
LOCAL_MODELSETS_PATH = Path(LOCAL_MODELSETS_PATH)
LOCAL_MODELSETS_CACHE = LOCAL_MODELSETS_PATH / MODELSETS_FILENAME

SHARE_PATH = 'shared'


def remote_modelsets():
    '''Retrieve info on the model sets available for download

    Returns:
        Nested dict hierarchy decoded from remote JSON file. The root of the
        hierarchy has keys 'models' and 'shared'.

        The value of the 'models' key is a dict whose keys are the folder
        names for the model sets and whose values are dicts giving details for
        each model set. Each model set dict has the following items:
        - a 'name' key with value being a str naming the model,
        - a 'files' key with value being a list of str specifying all file
          names in the model set folder,
        - a 'meta' key with value being a dict of meta data on the model,
        - a 'brain_params' key with value being a dict of parameters to
          instantiate a :py:class:`brain.BabyBrain` object.

        The 'shared' key at the root of the hierarchy points to a list of str
        giving file names in the 'shared' folder.

    '''
    msets_text = requests.get(MODELSETS_URL).text
    return json.loads(msets_text, object_hook=as_python_object)


def _ensure_local_path():
    '''Creates the directory to store model sets if it does not exist'''
    if not LOCAL_MODELSETS_PATH.is_dir():
        LOCAL_MODELSETS_PATH.mkdir(parents=True)
    local_share_path = LOCAL_MODELSETS_PATH / SHARE_PATH
    if not local_share_path.is_dir():
        local_share_path.mkdir(parents=True)


def update_local_cache():
    '''Update the locally cached info on model sets available for download'''
    _ensure_local_path()
    with open(LOCAL_MODELSETS_CACHE, 'wt') as f:
        json.dump(jsonify(remote_modelsets()), f)


def _ensure_local_cache(update=False):
    if not LOCAL_MODELSETS_CACHE.exists() or update:
        update_local_cache()


def specifications(update=False, local=False):
    '''Get full specifications for all model sets

    By default returns specifications for all models, including models that
    have not yet been downloaded.

    Args:
        update: whether or not to update the locally-cached list of models
        local: whether to include only locally-available models.

    Returns:
        a `dict` mapping model set IDs to dicts specifying details for each
        model set. Each model set dict has at least the following items:
        - a 'name' key with value being a str naming the model,
        - a 'meta' key with value being a dict of meta data on the model,
        - a 'brain_params' key with value being a dict of parameters to
          instantiate a :py:class:`brain.BabyBrain` object.
    '''
    _ensure_local_cache(update=update)
    if local:
        return local_modelsets()['models']
    with open(LOCAL_MODELSETS_CACHE, 'rt') as f:
        modelsets_info = json.load(f, object_hook=as_python_object)
    return modelsets_info['models']


def ids(update=False, local=False):
    '''List the available model sets by ID

    Args:
        update: whether or not to update the locally-cached list of models
        local: whether to include only locally-available models.

    Returns:
        a `list` of `str` giving the ID of each available model set
    '''
    return list(specifications(update=update, local=local).keys())


def meta(update=False, local=False):
    '''Obtain descriptive meta information on each model set

    Args:
        update: whether or not to update the locally-cached list of models
        local: whether to include only locally-available models.

    Returns:
        a `dict` mapping model set IDs to dicts of meta information associated
        with each model set
    '''
    return {msId: v['meta'] for msId, v in
            specifications(update=update, local=local).items()}


def local_modelsets():
    _ensure_local_path()
    modelsets = {}
    for modelset_file in LOCAL_MODELSETS_PATH.glob('**/' + MODELSET_FILENAME):
        modelset_id = modelset_file.parent.name
        with open(modelset_file, 'rt') as f:
            modelsets[modelset_id] = json.load(f, object_hook=as_python_object)
        modelsets[modelset_id]['files'] = [
            p.name for p in modelset_file.parent.iterdir()]

    local_share_path = LOCAL_MODELSETS_PATH / SHARE_PATH
    shared_files = [p.name for p in local_share_path.iterdir()]

    return {'models': modelsets, 'shared': shared_files}


def resolve(filename, modelset_id):
    '''Resolve the path to a file specified by a model set

    File names are resolved by first assuming the `modelset_id` argument
    specifies a directory containing model files (the model set path). If
    `modelset_id` is not a directory, then it is assumed to be a model ID as
    per :py:func:`ids`, and then the corresponding model set path in the local
    cache will be searched first. 

    If the file is not found in the model set path, it will be searched for in
    the shared directory of the local cache.

    Args:
        filename (str): file name to resolve, typically as specified by one of
            the keys obtained from :py:func:`get_params`.
        modelset_id (str): one of the model IDs as returned by :py:func:`ids`.

    Returns:
        A `Path` object giving the path to the specified file. If the file
        cannot be found a BadParam exception is raised.
    '''
    if Path(modelset_id).is_dir():
        modelset_path = Path(modelset_id)
    else:
        modelset_path = LOCAL_MODELSETS_PATH / modelset_id

    trial_path = modelset_path / filename
    if trial_path.is_file():
        return trial_path

    trial_path = LOCAL_MODELSETS_PATH / SHARE_PATH / filename
    if trial_path.is_file():
        return trial_path
    else:
        raise BadParam(f'The file {filename} could not be resolved for model set {modelset_id}')


def _get_modelset_files(modelset):
    params = modelset['brain_params']
    modelset_files = [v for k, v in params.items() if k.endswith('_file')]
    if type(params['params']) not in {dict, SegmentationParameters}:
        modelset_files.append(params['params'])
    return modelset_files


def _missing_files(modelset, modelset_id):
    modelset_files = _get_modelset_files(modelset)
    missing = []
    for fname in modelset_files:
        try:
            resolve(fname, modelset_id)
        except BadParam:
            missing.append(fname)
    return missing


def update(modelset_ids=None, force=True, cleanup=False, verbose=True):
    '''Updates any outdated model sets that are available remotely

    Args:
        modelset_ids: a list of str specifying one of the model IDs as
            returned by :py:func:`ids`. By default, only updates model sets
            that have already been downloaded locally. Set this to `'all'` to
            download all available model sets. 
        force: whether to replace existing files or just obtain missing ones.
        cleanup: whether to delete model set and shared files that are no
            longer on the remote
        verbose: whether to print download status to standard out.
    '''
    remote_mset_info = remote_modelsets()
    remote_msets = remote_mset_info['models']
    local_mset_info = local_modelsets()  # ensures local path
    local_msets = local_mset_info['models']

    if modelset_ids is None:
        remote_ids = set(remote_msets.keys())
        modelset_ids = list(remote_ids.intersection(local_msets.keys()))
    elif modelset_ids == 'all':
        modelset_ids = list(remote_msets.keys())

    invalid_ids = [msId for msId in modelset_ids if msId not in remote_msets]
    if any(invalid_ids):
        invalid_ids = ', '.join([f'`{msId}`' for msId in invalid_ids])
        raise BadParam(f'Requested model set(s) {invalid_ids} not available')

    # Update model sets from remote host
    for mset_id in modelset_ids:
        mset_meta = remote_msets[mset_id]
        local_mset_dir = LOCAL_MODELSETS_PATH / mset_id
        if not local_mset_dir.exists():
            local_mset_dir.mkdir(parents=True)

        mset_missing = mset_id not in local_msets
        mset_changed = mset_missing or local_msets[mset_id] != mset_meta
        if mset_missing or (mset_changed and not force):
            new_mset_meta = mset_meta.copy()
            del new_mset_meta['files']
            with open(local_mset_dir / MODELSET_FILENAME, 'wt') as f:
                json.dump(jsonify(new_mset_meta), f)

        if not force and not mset_missing:
            # If there is already a local model set and we are not forcing an
            # update, then we will only proceed with file download / cleanup
            # if the local model has missing files. This allows for the case
            # where a local version of a model has different files to those
            # found on the server, and so avoids downloading potentially
            # outdated extras.
            nmissing = len(_missing_files(local_msets[mset_id], mset_id))
            if nmissing == 0:
                continue

        remote_mset_files = set(mset_meta['files'])
        local_mset_files = local_msets.get(mset_id, {}).get('files', [])
        if cleanup:
            # Clean up any old files that are no longer on the remote
            for local_file in local_mset_files:
                if local_file not in remote_mset_files:
                    (local_mset_dir / local_file).unlink()

        remote_mset_files = remote_mset_files.difference({MODELSET_FILENAME})
        remote_mset_dir = urljoin(BABY_MODELS_URL, mset_id + '/')
        if force:
            files_to_download = remote_mset_files
        else:
            files_to_download = remote_mset_files.difference(local_mset_files)
        if len(files_to_download) > 0 and verbose:
            print(f'Downloading files for {mset_id}...')
        for remote_file in files_to_download:
            r = requests.get(urljoin(remote_mset_dir, remote_file))
            with open(local_mset_dir / remote_file, 'wb') as f:
                for chunk in r.iter_content(chunk_size=128):
                    f.write(chunk)

    # Update shared files from remote host
    remote_share_dir = urljoin(BABY_MODELS_URL, SHARE_PATH + '/')
    remote_shared_files = set(remote_mset_info['shared'])
    local_share_path = LOCAL_MODELSETS_PATH / SHARE_PATH
    local_shared_files = local_mset_info['shared']

    if cleanup:
        # Clean up any old files that are no longer on the remote
        for local_file in local_shared_files:
            if local_file not in remote_shared_files:
                (local_share_path / local_file).unlink()

    if force:
        files_to_download = remote_shared_files
    else:
        files_to_download = remote_shared_files.difference(local_shared_files)

    # Download any files that need updating
    if len(files_to_download) > 0 and verbose:
        print('Downloading shared files...')
    for remote_file in files_to_download:
        r = requests.get(urljoin(remote_share_dir, remote_file))
        with open(local_share_path / remote_file, 'wb') as f:
            for chunk in r.iter_content(chunk_size=128):
                f.write(chunk)


def _ensure_modelset(modelset_id):
    '''Ensure that a model set has been downloaded and is ready to use

    Args:
        modelset_id: a `str` specifying one of the model IDs as returned by
        :py:func:`ids`.
    '''
    _ensure_local_path()
    local_path = LOCAL_MODELSETS_PATH / modelset_id
    share_path = LOCAL_MODELSETS_PATH / SHARE_PATH
    local_modelset_file = local_path / MODELSET_FILENAME
    updated = False
    if not local_modelset_file.exists():
        update([modelset_id], force=False)
        updated = True
    with open(local_modelset_file, 'rt') as f:
        modelset = json.load(f, object_hook=as_python_object)
    modelset_files = _get_modelset_files(modelset)
    for fname in modelset_files:
        try:
            resolve(fname, modelset_id)
        except BadParam:
            if updated:
                raise BadModel('Model is corrupt. Contact maintainer.')
            else:
                update([modelset_id], force=False)
                try:
                    resolve(fname, modelset_id)
                except BadParam:
                    raise BadModel('Model is corrupt. Contact maintainer.')
                updated = True


def get_params(modelset_id):
    '''Get model set parameters

    The parameters are designed to be supplied as the argument to
    instantiate a :py:class:`brain.BabyBrain` object.

    The model set will be automatically downloaded if it has not yet been.

    Args:
        modelset_id: a `str` specifying one of the model IDs as returned by
        :py:func:`ids`.
    '''
    _ensure_modelset(modelset_id)
    local_path = LOCAL_MODELSETS_PATH / modelset_id
    local_modelset_file = local_path / MODELSET_FILENAME
    with open(local_modelset_file, 'rt') as f:
        modelset = json.load(f, object_hook=as_python_object)
    return modelset['brain_params']


def get(modelset_id, **kwargs):
    '''Get a model set as a BabyBrain object

    The model set will be automatically downloaded if it has not yet been.

    Args:
        modelset_id: a `str` specifying one of the model IDs as returned by
        :py:func:`ids`.

    Returns:
        A :py:class:`brain.BabyBrain` object instantiated with the model set
        parameters.
    '''
    from .brain import BabyBrain
    return BabyBrain(modelset_path=modelset_id, **get_params(modelset_id),
                     **kwargs)
