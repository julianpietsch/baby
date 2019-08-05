#!/usr/bin/env python3
from http.server import HTTPServer, BaseHTTPRequestHandler
import json

from os.path import dirname, join
from functools import reduce
from operator import mul
import numpy as np

from baby.run import BabyRunner

baby = None

matlab_dtype = np.dtype('uint16').newbyteorder('>')

### Helper functions and classes ###

def pred_to_hex(p):
    return np.floor(p.flatten('F')*255.).astype(np.uint8).tobytes().hex()


class JSONError(Exception):
    def __init__(self, status, message):
        """Initialise with the result of the couchdb.Database.upload function"""
        super(JSONError, self).__init__(message)
        self.message = message
        self.status = status


class nnHandler(BaseHTTPRequestHandler):
    def _json_response(self, status, **kwargs):
        self.send_response(status)
        self.send_header('Content-type', 'application/json')
        self.end_headers()
        self.wfile.write(bytes(json.dumps(kwargs), 'ASCII'))

    def do_POST(self):
        try:
            print(self.headers.get('content-type'))
            # matlab produces: application/x-www-form-urlencoded
            # if not self.headers.get('content-type')=='application/json':
            #     raise JSONError(400, 'request should be json encoded')

            data = self.rfile.read(int(self.headers.get('content-length')))
            try:
                json_data = json.loads(data)
            except:
                raise JSONError(400, 'error in json formatting')

            if not 'dims' in json_data or not 'img' in json_data:
                raise JSONError(400, 'json object needs "dims" and "img" fields')

            dims = json_data['dims']
            if not type(dims)==list or len(dims)!=4 or \
                    not all([type(d)==int and d>0 for d in dims]) or dims[3]!=5:
                raise JSONError(400, '"dims" must be a length 4 integer array: [ntps,width,height,5]')

            byteimg = json_data['img']
            if not type(byteimg)==str or not len(byteimg)==4*reduce(mul,dims):
                raise JSONError(400, '"img" must be a hex string of the flattened image')

            try:
                img = np.frombuffer(bytes.fromhex(byteimg), dtype=matlab_dtype)
            except Exception as err:
                print(err.message)
                raise JSONError(400, '"img" must be a hex string of the flattened image')

            img = img.reshape(dims, order='F')
            if json_data.get('test', False):
                from imageio import imwrite
                imwrite('nn-server-test-img.png', np.squeeze(img[0,:,:,0]))
                self._json_response(200, results='test image written')
                return

            print('Data received. Segmenting {} trap images'.format(len(img)))
            global baby
            pred = baby.run(img)
            self._json_response(200, results=pred)

        except JSONError as err:
            print(err.message)
            self._json_response(err.status, error=err.message)


if __name__=='__main__':
    # Compensate for bug in tensorflow + RTX series NVidia GPUs
    import tensorflow as tf
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    tf.keras.backend.set_session(tf.Session(config=config))

    # Load BabyRunner
    baby = BabyRunner()
    print('ready')

    server_address = ('',5101)
    httpd = HTTPServer(server_address, nnHandler)
    httpd.serve_forever()
