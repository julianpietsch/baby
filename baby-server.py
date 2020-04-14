#!/usr/bin/env python3
from __future__ import absolute_import, division, print_function, \
    unicode_literals
import asyncio
import threading
from aiohttp import web
from concurrent.futures import ThreadPoolExecutor

import gc
import time
import json
from os.path import dirname, join
from uuid import uuid4
from functools import reduce
from operator import mul
import numpy as np

from python.baby.brain import BabyBrain
from python.baby.crawler import BabyCrawler

routes = web.RouteTableDef()

DTYPES = {
    8: np.dtype('uint8'),
    16: np.dtype('uint16')
}

SERVER_DIR = dirname(__file__)

MAX_RUNNERS = 3
MAX_SESSIONS = 5
SLEEP_TIME = 0.2  # time between threaded checks for data availability
MAX_ATTEMPTS = 150  # allows for 30s delay before timing out
MAX_IMG_SIZE = 100 * 1024 * 1024  # allows for raw image sizes up to 100 MB

DIMS_ERROR_MSG = '"dims" must be a length 4 integer array: [ntraps, width, height, depth]'


### Helper functions and classes ###

def web_error(message=None, errtype=web.HTTPBadRequest, internalerr=None):
    if internalerr is not None:
        print('ERR: ' + internalerr.message)

    if message is not None:
        print('ERR: ' + message)

    return errtype(text=message)


class Timeout(Exception):
    pass


class PredMissingError(Exception):
    pass


class TaskMaster(object):
    def __init__(self):
        self._lock = threading.Lock()

        self._runner_pool = []
        self.runners = {}
        self._session_pool = []
        self.sessions = {}
        self.tf_session = None
        self.tf_graph = None
        self.tf_version = (0, 0, 0)

    @property
    def modelsets(self):
        with open(join(SERVER_DIR, 'modelsets.json'), 'rt') as f:
            modelsets = json.load(f)
        return modelsets

    def new_session(self, model_name):
        # Clean up old sessions that exceed the maximum allowed number
        nsessions = len(self._session_pool)
        if nsessions >= MAX_SESSIONS:
            with self._lock:
                for i in range(nsessions - MAX_SESSIONS + 1):
                    oldsession = self._session_pool.pop()
                    del self.sessions[oldsession]

        sessionid = str(uuid4())
        sessioninfo = {'id': sessionid, 'model_name': model_name}

        with self._lock:
            self._session_pool.insert(0, sessionid)
            self.sessions[sessionid] = sessioninfo

        return sessionid

    def ensure_runner(self, model_name, modelsets=None):
        if model_name in self.runners:
            print('Model "{}" already loaded. Skipping...'.format(model_name))
            return

        if modelsets is None:
            modelsets = self.modelsets

        assert model_name in modelsets

        # Clean up old runners that exceed the maximum allowed number
        nrunners = len(self._runner_pool)
        with self._lock:
            if nrunners >= MAX_RUNNERS:
                keep = set()
                for sessionid in list(self._session_pool):
                    session_model = self.sessions[sessionid]['model_name']
                    if len(keep) < MAX_RUNNERS:
                        # Keep only runners for the newest sessions
                        keep.add(session_model)
                    if session_model not in keep:
                        # This session is now expired
                        self._session_pool.remove(sessionid)
                        del self.sessions[sessionid]

                # Delete any runners not in keep
                for runner in list(self._runner_pool):
                    if runner not in keep:
                        self._runner_pool.remove(runner)
                        del self.runners[runner]

            # Create a placeholder for the pending runner
            self._runner_pool.insert(0, model_name)
            self.runners[model_name] = 'pending'

        # Make sure we clean up any models that were deleted to free up memory
        gc.collect()  # should in theory work for tf2, but doesn't appear to...

        # Load BabyRunner
        print('Starting new runner for model "{}"...'.format(model_name))

        baby = BabyBrain(**modelsets[model_name],
                         session=self.tf_session, graph=self.tf_graph)

        if self.runners.get(model_name) == 'pending':
            with self._lock:
                self.runners[model_name] = baby

    def get_runner(self, sessionid):
        for i in range(MAX_ATTEMPTS):
            runner = self.runners[self.sessions[sessionid]['model_name']]
            if runner != 'pending':
                break
            time.sleep(SLEEP_TIME)

        if runner == 'pending':
            raise Timeout

        return runner

    def is_valid_session(self, sessionid):
        return sessionid in self.sessions and \
               self.sessions[sessionid]['model_name'] in self.runners and \
               self.sessions[sessionid].get('pred') != 'pending'

    def is_valid_depth(self, sessionid, depth):
        return self.get_runner(sessionid).depth == depth

    def ensure_session_crawler(self, sessionid):
        if 'crawler' in self.sessions[sessionid]:
            pass
        else:
            self.ensure_runner(self.sessions[sessionid]['model_name'])
            while True:
                try:
                    brain = self.get_runner(
                        self.sessions[sessionid]['model_name'])
                except KeyError:
                    continue
                break
            self.sessions[sessionid]['crawler'] = BabyCrawler(brain)

    def segment(self, sessionid, img):
        crawler = self.sessions[sessionid]['crawler']
        #baby = self.get_runner(sessionid)

        with self._lock:
            self.sessions[sessionid]['pred'] = 'pending'

        # if tf.keras.backend.get_session() != self.tf_session:
        #     tf.keras.backend.set_session(self.tf_session)

        pred = crawler.step(img)
        #pred = baby.run(img)

        with self._lock:
            self.sessions[sessionid]['pred'] = pred

    def results(self, sessionid):
        for i in range(MAX_ATTEMPTS):
            pred = self.sessions[sessionid].get('pred')
            if pred is None:
                raise PredMissingError
            if pred != 'pending':
                break
            time.sleep(SLEEP_TIME)

        if pred == 'pending':
            raise Timeout

        return pred


### API routes ###

@routes.get('/')
async def version(request):
    return web.json_response({'baby': 'v1.0'})


@routes.get('/models')
async def get_modelsets(request):
    taskmstr = request.app['TaskMaster']
    return web.json_response(list(taskmstr.modelsets.keys()))


@routes.get('/session/{model}')
async def get_session(request):
    model_name = request.match_info['model']
    taskmstr = request.app['TaskMaster']
    modelsets = taskmstr.modelsets

    if model_name not in modelsets:
        raise web.HTTPNotFound(text='"{}" model is unknown'.format(model_name))

    # Ensure model is loaded in another thread
    loop = asyncio.get_event_loop()
    loop.run_in_executor(request.app['Executor'],
                         taskmstr.ensure_runner, model_name, modelsets)
    sessionid = taskmstr.new_session(model_name)

    print('Creating new session "{}" with model "{}"...'.format(
        sessionid, model_name))

    return web.json_response({'sessionid': sessionid})


@routes.get('/sessions')
async def get_sessions(request):
    taskmstr = request.app['TaskMaster']
    print(taskmstr._session_pool)
    print(taskmstr.sessions)
    print(taskmstr._runner_pool)
    print(taskmstr.runners)
    return web.json_response(taskmstr.sessions)


@routes.post('/segment')
async def segment(request):
    taskmstr = request.app['TaskMaster']
    executor = request.app['Executor']
    loop = asyncio.get_event_loop()

    if 'sessionid' not in request.query:
        raise web_error('"sessionid" must be specified in the query string')

    sessionid = request.query['sessionid']

    if not taskmstr.is_valid_session(sessionid):
        raise web_error('the session is invalid or expired',
                        errtype=web.HTTPForbidden)

    print('Processing query for session "{}"'.format(sessionid))

    await taskmstr.ensure_session_crawler(sessionid)
    reader = await request.multipart()

    field = await reader.next()
    if field.name != 'dims':
        raise web_error('"dims" field must be second')
    dims = await field.read(decode=True)

    try:
        dims = json.loads(dims)
    except json.JSONDecodeError:
        raise Exception(DIMS_ERROR_MSG)

    if type(dims) != list or len(dims) != 4 or \
            not all([type(d) == int and d > 0 for d in dims]):
        raise Exception(DIMS_ERROR_MSG + " {}".format(dims))

    try:
        is_valid_depth = await loop.run_in_executor(
            executor, taskmstr.is_valid_depth, sessionid, dims[3])
    except Timeout:
        raise web_error('session is still loading or has stalled',
                        errtype=web.HTTPRequestTimeout)
    if not is_valid_depth:
        raise web_error('image depth is incorrect for this session')

    field = await reader.next()
    if field.name != 'bitdepth':
        raise web_error('"bitdepth" must be fourth')
    bitdepth = await field.read(decode=True)
    try:
        bitdepth = int(bitdepth)
    except ValueError:
        raise web_error('"bitdepth" must be "8" or "16"')
    if bitdepth not in DTYPES:
        raise web_error('"bitdepth" must be "8" or "16"')

    field = await reader.next()
    if field.name != 'img':
        raise web_error('"img" field must be fourth')

    size = 0
    imgbytes = bytes()
    while True:
        chunk = await field.read_chunk()
        if not chunk:
            break
        size += len(chunk)
        if size > MAX_IMG_SIZE:
            raise web_error('image size cannot exceed 100 MB',
                            errtype=web.HTTPRequestEntityTooLarge)
        imgbytes += chunk

    if len(imgbytes) != reduce(mul, dims) * bitdepth / 8:
        raise web_error(
            'image size ({}) does not match specified dimensions ({})'.format(
                len(imgbytes), json.dumps(dims)
            ))

    try:
        img = np.frombuffer(imgbytes, dtype=DTYPES[bitdepth])
    except Exception as err:
        raise web_error('image data is corrupted')

    img = img.reshape(dims, order='F')

    if request.query.get('test', False):
        print('Data received. Writing test image to "baby-server-test.png"...')
        from imageio import imwrite
        imwrite(join(SERVER_DIR, 'baby-server-test.png'),
                np.squeeze(img[0, :, :, 0]))
        return web.json_response({'status': 'test image written'})

    print('Data received. Segmenting {} images...'.format(len(img)))

    loop.run_in_executor(executor, taskmstr.segment, sessionid, img)

    return web.json_response({
        'status': 'processing {} trap images'.format(len(img))})


@routes.get('/segment')
async def get_segmentation(request):
    taskmstr = request.app['TaskMaster']
    executor = request.app['Executor']
    loop = asyncio.get_event_loop()

    if 'sessionid' not in request.query:
        raise web_error('"sessionid" must be specified in the query string')
    sessionid = request.query['sessionid']

    try:
        pred = await loop.run_in_executor(executor, taskmstr.results,
                                          sessionid)
    except Timeout:
        raise web_error('segmentation is still running or has stalled',
                        errtype=web.HTTPRequestTimeout)

    return web.json_response(pred)


app = web.Application()
app.add_routes(routes)
app['TaskMaster'] = TaskMaster()
app['Executor'] = ThreadPoolExecutor(2)

if __name__ == '__main__':
    import tensorflow as tf

    tf_version = tuple(int(v) for v in tf.version.VERSION.split('.'))
    app['TaskMaster'].tf_version = tf_version

    # Compensate for bug in tensorflow + RTX series NVidia GPUs
    if tf_version[0] == 1:
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        tf_session = tf.Session(config=config)
        tf_graph = tf.get_default_graph()
        app['TaskMaster'].tf_session = tf_session
        app['TaskMaster'].tf_graph = tf_graph
        # tf.keras.backend.set_session(tf_session)
        # assert app['TaskMaster'].tf_session == tf.keras.backend.get_session()
    elif tf_version[0] == 2:
        gpus = tf.config.experimental.list_physical_devices('GPU')
        if gpus:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logical_gpus = tf.config.experimental.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,", len(logical_gpus),
                  "Logical GPUs")

        MAX_RUNNERS = 1
    else:
        raise Exception(
            'Unsupported version of tensorflow encountered ({})'.format(
                tf.version.VERSION
            )
        )

    web.run_app(app, port=5101)
