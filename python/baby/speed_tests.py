#!/usr/bin/env python3

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

import sys
from contextlib import closing
from itertools import chain, repeat
from time import perf_counter

import numpy as np

import baby

class Tee(object):
    def __init__(self, file=None):
        self._file = file

    def write(self, string):
        if self._file:
            self._file.write(string)
        sys.stdout.write(string)

    def close(self):
        if self._file:
            self._file.close()


class TimingLogger(object):
    def __init__(self, file=sys.stdout):
        if isinstance(file, TimingLogger):
            file = TimingLogger._file
        self._file = file
        self._start_time = perf_counter()
        self._held_output = None

    def start(self, section_name=None, hold_output=False):
        if hold_output or section_name is None:
            self._held_output = section_name
        else:
            self._held_output = None
            print(section_name, end='', file=self._file)
        self._start_time = perf_counter()

    def finish(self):
        if self._held_output:
            print(self._held_output, end='', file=self._file)
        print(' took {:.3f} seconds.'.format(perf_counter() - self._start_time),
                file=self._file)

    def separator(self):
        print('\n--------------------------------\n', file=self._file)

    def log(self, text):
        print(text, file=self._file)

    def copy(self):
        return TimingLogger(file=self._file)


def get_traps_timepoint(img, trap_locs, tile_size=81):
    """Extract tiles from an image

    :param img: ndarray with shape (image_width, image_height, ...)
        An image from which to extract tiles. The first two dimensions must be
        x, y coordinates.
    :param trap_locs: ndarray with shape (n_traps, 2)
        Trap centres (rows) specified in x, y coordinates (columns).
    :param tile_size: width of (square) tile to be centered at each specified
        trap location.
    """
    hw = (tile_size + 1) / 2
    imW, imH = img.shape[0:2]
    imD = img.shape[2:] # will have length 0 if image is 2D
    r_locs, c_locs = np.hsplit(trap_locs, 2)

    # Pad the image with its median value to ensure trap subsets do not exceed image bounds
    pad_val = np.median(img)
    max_loc = max(0, -r_locs.min(), r_locs.max() - imW, -c_locs.min(), c_locs.max() - imH)
    pad_width = np.ceil(max_loc + hw + 1).astype('int')
    pad_widths = 2 * ((pad_width,),) + len(imD) * ((0,),)
    img = np.pad(img, pad_widths, 'constant', constant_values=pad_val)

    # Calculate bounding box coords for each trap
    rl_locs = np.round(pad_width + r_locs.squeeze() - hw - 1).astype('int')
    rh_locs = rl_locs + tile_size
    cl_locs = np.round(pad_width + c_locs.squeeze() - hw - 1).astype('int')
    ch_locs = cl_locs + tile_size

    return np.stack([img[rl:rh, cl:ch, ...] for rl, rh, cl, ch in
                    zip(rl_locs, rh_locs, cl_locs, ch_locs)])


def load_brain(timing, model='evolve_brightfield_60x_5z'):
    import tensorflow as tf
    from baby.brain import BabyBrain

    # Compensate for bug in tensorflow + RTX series NVidia GPUs
    tf_version = tuple(int(v) for v in tf.version.VERSION.split('.'))
    if tf_version[0] == 1:
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        tf_session = tf.Session(config=config)
        tf_graph = tf.get_default_graph()
    elif tf_version[0] == 2:
        tf_session, tf_graph = 2 * (None,)
        gpus = tf.config.experimental.list_physical_devices('GPU')
        if gpus:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logical_gpus = tf.config.experimental.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,", len(logical_gpus),
                    "Logical GPUs")
    else:
        raise Exception(
            'Unsupported version of tensorflow encountered ({})'.format(
                tf.version.VERSION
            )
        )

    timing.start('Loading BabyBrain with "{}" models...'.format(model),
                 hold_output=True)
    modelsets = baby.modelsets()
    bb = BabyBrain(session=tf_session, graph=tf_graph, **modelsets[model])

    timing.separator()
    timing.finish()
    return bb


def load_seg_expt(timing, source_dir, pos=None):
    """Load and track experiment"""
    from core.experiment import Experiment
    from core.segment import SegmentedExperiment

    timing.start('Loading Experiment from "{}"...'.format(source_dir))
    expt = Experiment.from_source(source_dir)
    timing.finish()

    if pos is None:
        pos = expt.positions[0]

    expt.current_position = pos

    timing.start('Tracking traps for pos "{}"...'.format(pos))
    seg_expt = SegmentedExperiment(expt)
    timing.finish()

    return seg_expt


def crawl_expt(timing, seg_expt, bb, ntps=5, refine_outlines=True,
               return_volume=False):
    """Crawl through time points

    :param timing: instance of TimingLogger
    :param seg_expt: instance of SegmentedExperiment
    :param bb: instance of BabyBrain
    :param ntps: maximum number of time points to crawl through if available
    """
    from baby.crawler import BabyCrawler

    expt = seg_expt.raw_expt
    ntps = min(ntps, expt.shape[1])
    channel = expt.channels.index('Brightfield')
    pos = seg_expt.current_position

    outer_timing = timing.copy()
    outer_timing.start('...crawling through {:d} time points'.format(ntps),
                       hold_output=True)

    timing.log('Start crawling...')
    crawler = BabyCrawler(bb)
    output = []
    for tp in range(ntps):
        timing.start('Loading traps for tp {:d}...'.format(tp + 1))
        img = expt[channel,tp,:,:,:].squeeze()
        trap_locs = seg_expt.trap_locations[pos][tp]
        tp_traps = get_traps_timepoint(img, trap_locs)
        timing.finish()

        timing.start('Stepping crawler for {:d} traps...'.format(len(tp_traps)))
        output.append(crawler.step(
            tp_traps, with_edgemasks=True,
            assign_mothers=True, refine_outlines=refine_outlines,
            return_volume=return_volume
        ))
        timing.finish()
    outer_timing.finish()
    return output


def subtask_timings(timing, seg_expt, bb, ntps=5, refine_outlines=True,
                    return_volume=False):
    """Step through sub-tasks of the crawler

    :param timing: instance of TimingLogger
    :param seg_expt: instance of SegmentedExperiment
    :param bb: instance of BabyBrain
    :param ntps: maximum number of time points to crawl through if available
    """
    from baby.preprocessing import robust_norm
    from baby.utils import split_batch_pred, batch_iterator

    expt = seg_expt.raw_expt
    ntps = min(ntps, expt.shape[1])
    channel = expt.channels.index('Brightfield')
    pos = seg_expt.current_position

    outer_timing = timing.copy()
    outer_timing.start('...stepping through {:d} time points'.format(ntps),
                       hold_output=True)

    timing.log('Stepping through sub-tasks for {:d} time points...'.format(ntps))

    # Running time to load and preprocess 5 tps
    trap_locs = seg_expt.trap_locations[pos]
    timing.start('Loading and preprocessing {:d} traps for {:d} tps...'.format(
        len(trap_locs[0]), ntps))
    imgs = expt[channel,0:ntps,:,:,:].squeeze()
    tps_traps = [get_traps_timepoint(img, trap_locs[tp]) for
                 tp, img in enumerate(imgs)]
    tps_traps = [np.stack([robust_norm(img, {}) for img in tp_traps])
                 for tp_traps in tps_traps]
    timing.finish()

    # Running time for CNN prediction
    cnn_outputs = []
    for tp in range(ntps):
        timing.start('Running CNN on time point {:d}...'.format(tp + 1))

        cnn_outputs.append(list(chain(*[
            split_batch_pred(bb.morph_predict(batch))
            for batch in batch_iterator(tps_traps[tp])
        ])))
        timing.finish()

    # Running time for segmentation
    tp_seg_masks = []
    for tp in range(ntps):
        timing.start('Segmenting time point {:d}...'.format(tp + 1))
        seg_masks = []
        for cnn_output in cnn_outputs[tp]:
            seg_result = bb.morph_segmenter.segment(
                cnn_output, refine_outlines=refine_outlines,
                return_volume=return_volume)
            _0xy = (0,) + cnn_output.shape[1:3]
            if len(seg_result.masks) > 0:
                seg_masks.append(np.stack(seg_result.masks))
            else:
                seg_masks.append(np.zeros(_0xy, dtype='bool'))
        tp_seg_masks.append(seg_masks)
        timing.finish()

    # Running time for tracking
    tnames = bb.flattener.names()
    i_budneck = tnames.index('bud_neck')
    bud_target = 'sml_fill' if 'sml_fill' in tnames else 'sml_inte'
    i_bud = tnames.index(bud_target)

    tracker_states = list(repeat(None, len(tps_traps[0])))
    for tp in range(ntps):
        timing.start('Tracking time point {:d}...'.format(tp + 1))
        tgen = zip(cnn_outputs[tp], tp_seg_masks[tp])
        for i, (cnn_output, masks) in enumerate(tgen):
            tracking = bb.tracker.step_trackers(
                masks, cnn_output[i_budneck],
                cnn_output[i_bud], state=tracker_states[i])
            tracker_states[i] = tracking['state']
        timing.finish()

    outer_timing.finish()


def main():
    from optparse import OptionParser
    usage = "usage: %prog [options] experiment_dir"
    parser = OptionParser(usage=usage)
    dflt_model = "evolve_brightfield_60x_5z"
    parser.add_option(
        "-m", metavar="MODELSET", default=dflt_model, dest="modelset",
        help="specify the MODELSET to use (default '{}')".format(dflt_model)
    )
    parser.add_option("-p", dest="pos", metavar="POSNAME",
                      help="specify the position POSNAME to use")
    parser.add_option("-t", "--ntps", dest="ntps",
                      type='int', default=5, metavar="N",
                      help="crawl through (up to) N time points if available")
    parser.add_option("-f", "--file", dest="filename", metavar="FILE",
                      help="write timing info to FILE (also to std out)")
    parser.add_option("-b", "--basic-edges", action="store_false",
                      dest="refine_outlines", default=True,
                      help="Turn outline refinement off")
    parser.add_option("-v", "--volume", dest="return_volume", default=False,
                      action="store_true")
    (options, args) = parser.parse_args()

    if len(args) != 1:
        parser.print_help()
        sys.exit()

    file_handle = None
    if options.filename:
        file_handle = open(options.filename, 'wt')

    with closing(Tee(file_handle)) as f:
        timing = TimingLogger(file=f)

        bb = load_brain(timing, model=options.modelset)

        timing.separator()

        seg_expt = load_seg_expt(timing, args[0], pos=options.pos)

        timing.separator()

        subtask_timings(timing, seg_expt, bb, ntps=options.ntps,
                        refine_outlines=options.refine_outlines,
                        return_volume=options.return_volume)

        timing.separator()

        output = crawl_expt(timing, seg_expt, bb, ntps=options.ntps,
                            refine_outlines=options.refine_outlines,
                            return_volume=options.return_volume)

        timing.separator()
