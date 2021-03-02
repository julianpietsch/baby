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
from __future__ import absolute_import, division, print_function, unicode_literals

from itertools import repeat


class BatchSizeChangedError(Exception):
    pass


class BabyCrawler(object):
    '''Coordinates incremental segmentation and tracking over a timelapse

    :param baby_brain: an instantiated BabyBrain defining the models
    '''

    def __init__(self, baby_brain):
        self.baby_brain = baby_brain
        self.N_batch = None
        self.tracker_states = None

    def step(self,
             bf_img_batch,
             with_edgemasks=False,
             assign_mothers=False,
             return_baprobs=False,
             refine_outlines=False,
             with_volumes=False,
             parallel=False,
             **kwargs):
        '''Process the next batch of input images

        :param bf_img_batch: a list of ndarray with shape (X, Y, Z), or
            equivalently an ndarray with shape (N_images, X, Y, Z)

        :yields: for each image in `bf_img_batch` a dict with
            - centres: list of float pairs corresponding to (x, y) coords for
              each detected cell,
            - angles: list of lists of floats corresponding, for each cell, to
              angles (radians) used to form active contour outline in radial
              space
            - radii: list of lists of floats corresponding, for each cell, to
              radii used to form active contour outline in radial space
            - cell_label: list of int corresponding to tracked global ID for
              each cell detected in this image (indexed from 1)
            - mother_assign: (optional) list of int specifying for each
              (global) cell label ID, the cell label ID of the corresponding
              mother (0 if no mother was found)
            - p_bud_assign: (optional) matrix as a list of lists of floats,
              specifying the probability that a cell (outer list) is a mother
              to another cell (inner lists) in this image
            - edgemasks: (optional) an ndarray of dtype "bool" with shape
              (N_cells, X, Y) specifying the rasterised edge for each
              segmented cell
        '''
        if self.N_batch is None:
            self.N_batch = len(bf_img_batch)

        if len(bf_img_batch) != self.N_batch:
            raise BatchSizeChangedError(
                'cannot change batch size mid-session')

        if self.tracker_states is None:
            self.tracker_states = list(repeat(None, self.N_batch))

        output = []

        if parallel:
            seg_trk_gen = self.baby_brain.segment_and_track_parallel(
                bf_img_batch,
                tracker_states=self.tracker_states,
                yield_next=True,
                yield_edgemasks=with_edgemasks,
                yield_volumes=with_volumes,
                assign_mothers=assign_mothers,
                return_baprobs=return_baprobs,
                refine_outlines=refine_outlines)
        else:
            seg_trk_gen = self.baby_brain.segment_and_track(
                bf_img_batch,
                tracker_states=self.tracker_states,
                yield_next=True,
                yield_edgemasks=with_edgemasks,
                yield_volumes=with_volumes,
                assign_mothers=assign_mothers,
                return_baprobs=return_baprobs,
                refine_outlines=refine_outlines)

        for i, (seg, state) in enumerate(seg_trk_gen):
            # Update cumulative state
            self.tracker_states[i] = state
            output.append(seg)

        return output
