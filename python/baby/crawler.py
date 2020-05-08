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

    def step(self, bf_img_batch, with_edgemasks=False,
             assign_mothers=False, return_baprobs=False,
             refine_outlines=False, **kwargs):
        '''Process the next batch of input images

        :param bf_img_batch: a list of ndarray with shape (X, Y, Z), or
            equivalently an ndarray with shape (N_images, X, Y, Z)

        :returns: for each image in the batch a dict specifying centres,
        angles, cell labels and the current best estimate for mother-daughter
        pairs
        '''
        if self.N_batch is None:
            self.N_batch = len(bf_img_batch)

        if len(bf_img_batch) != self.N_batch:
            raise BatchSizeChangedError('cannot change batch size mid-session')

        if self.tracker_states is None:
            self.tracker_states = list(repeat(None, self.N_batch))

        output = []

        seg_trk_gen = self.baby_brain.segment_and_track(
            bf_img_batch,
            tracker_states=self.tracker_states, yield_next=True,
            yield_edgemasks=with_edgemasks,
            assign_mothers=assign_mothers, return_baprobs=return_baprobs,
            refine_outlines=refine_outlines
        )

        for i, (seg, state) in enumerate(seg_trk_gen):
            # Update cumulative state
            self.tracker_states[i] = state
            output.append(seg)

        return output
