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
        self.max_lbl_batch = None
        self.prev_cell_lbls_batch = None
        self.prev_feats_batch = None
        self.N_batch = None

    def step(self, bf_img_batch, with_edgemasks=False):
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

        if self.max_lbl_batch is None:
            self.max_lbl_batch = [0 for i in range(self.N_batch)]

        if self.prev_cell_lbls_batch is None:
            self.prev_cell_lbls_batch = [[] for i in range(self.N_batch)]

        if self.prev_feats_batch is None:
            self.prev_feats_batch = [[] for i in range(self.N_batch)]

        output = []

        nstepsback = self.baby_brain.tracker.nstepsback
        seg_trk_gen = self.baby_brain.segment_and_track(
            bf_img_batch, max_lbl_batch=self.max_lbl_batch,
            prev_cell_lbls_batch=self.prev_cell_lbls_batch,
            prev_feats_batch = self.prev_feats_batch,
            yield_edgemasks=with_edgemasks, yield_next=True
        )

        for i, new_seg_trk in enumerate(seg_trk_gen):
            seg, newmaxlbl, newlbls, newfeats = new_seg_trk

            # Update aggregated cell labels and features
            self.max_lbl_batch[i] = newmaxlbl
            self.prev_cell_lbls_batch[i].append(newlbls)
            self.prev_cell_lbls_batch[i] = \
                self.prev_cell_lbls_batch[i][-nstepsback:]
            self.prev_feats_batch[i].append(newfeats)
            self.prev_feats_batch[i] = self.prev_feats_batch[i][-nstepsback:]

            # TODO:
            # self.cumulative_mother_bud_assignment[i] += ...

            # Curate output
            seg['cellLabel'] = newlbls
            # seg['mother_assign'] = [(mother, daughter),...]
            # del seg['ba_probs']

            output.append(seg)

        return output
