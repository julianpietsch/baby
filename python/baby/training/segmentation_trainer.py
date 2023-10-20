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
from math import floor, log10
from itertools import combinations, product, chain, repeat, islice
from typing import NamedTuple, Union, Tuple, Any
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from tqdm import tqdm
# from numba import njit

from baby import segmentation
from baby.morph_thresh_seg import (MorphSegGrouped, Group,
                                   SegmentationParameters,
                                   BROADCASTABLE_PARAMS)
from baby.utils import batch_iterator, split_batch_pred
from baby.augmentation import ScalingAugmenter
from baby.performance import calc_IoUs, best_IoU, calc_AP
from baby.errors import BadProcess, BadParam, BadType
from .utils import (SharedParameterContainer, SharedDataContainer,
                    TrainValTestProperty, standard_augmenter)
from .smoothing_model_trainer import SmoothingModelTrainer
from .cnn_trainer import CNNTrainer


# Todo add default parameters to SegTrainer
#       Search space !
DEFAULT_SEG_PARAM_COORDS = {
    'nclosing': [0, 1, 2],
    'nopening': [0, 1, 2],
    'interior_threshold': np.arange(0.3, 1.0, 0.05).tolist(),
    'connectivity': [1, 2],
    'edge_sub_dilations': [0, 1, 2]
}


class SegExample(NamedTuple):
    """CNN output paired with target segmented outlines and info

    Used for optimising segmentation hyperparameters and for training bud
    assigment models
    """
    pred: np.ndarray
    target: np.ndarray
    info: dict
    img: np.ndarray


class Score(NamedTuple):
    """Scoring metrics for segmentation performance.

    Metrics are defined in terms of number of true positives ``TP``, number of
    false positives ``FP`` and number of false negatives ``FN``.

    Attributes:
        precision: ``TP / (TP + FP)``.
        recall: ``TP / (TP + FN)``.
        F1: Balanced F-score ``2 * precision * recall / (precision +
            recall)``.
        F0_5: F-score biased for recall ``1.25 * precision * recall / (0.25 *
            precision + recall)``.
        F2: F-score biased for precision ``5 * precision * recall / (4 *
            precision + recall)``.
        meanIoU: Mean intersection over union between segmented and ground
            truth cell masks.
    """
    precision: float
    recall: float
    F1: float
    F0_5: float
    F2: float
    meanIoU: float


def _example_generator(cnn, dgen):
    # b_iter = batch_iterator(sorted(dgen.ordering), batch_size=dgen.batch_size)
    # Sorting is counterproductive to method of subsample_frac
    b_iter = batch_iterator(dgen.ordering, batch_size=dgen.batch_size)
    with tqdm(total=len(dgen.ordering)) as pbar:
        for b_inds in b_iter:
            batch = dgen.parallel_get_indices(b_inds)

            imgs = [img for img, _ in batch]
            preds = split_batch_pred(
                cnn.predict(np.stack(imgs), verbose=0))
            for pred, (img, (lbl, info)) in zip(preds, batch):
                pbar.update()
                lbl = lbl.transpose(2, 0, 1)
                # Filter out examples that have been augmented away
                valid = lbl.sum(axis=(1, 2)) > 0
                lbl = lbl[valid]
                info = info.copy() # ensure we do not modify original
                clab = info.get('cellLabels', []) or []
                if type(clab) is int:
                    clab = [clab]
                clab = [l for l, v in zip(clab, valid) if v]
                info['cellLabels'] = clab
                buds = info.get('buds', []) or []
                if type(buds) is int:
                    buds = [buds]
                buds = [b for b, v in zip(buds, valid) if v]
                info['buds'] = buds
                yield SegExample(pred, lbl, info, img)


def _sub_params(sub, param_template):
    """Helper function for ``SegFilterParamOptim.fit_filter_params``.

    Substitutes parameters in a ``dict``, possibly group-specific.
    """
    p = {
        k: v.copy() if type(v) == list else v
        for k, v in param_template.items()
    }
    for (k, g), v in sub.items():
        if g is None:
            p[k] = v
        else:
            p[k][g] = v
    return p


# @njit
def _filter_trial(containment, containment_thresh, area, min_area,
                  pedge_thresh, group_thresh_expansion, 
                  gIds, gLs, gUs, gRs, group,
                  p_edge, max_IoU, IoU_thresh, assignId, nT):
    """Numba optimised filter trial implementation.

    TODO: currently the numba version kills the kernel... This implementation
    is nonetheless orders of magnitude faster than the previous.
    """

    rejects = (containment > containment_thresh) | (area < min_area)
    for t_pe, g_ex, g, l, u, gr in zip(pedge_thresh,
                                       group_thresh_expansion,
                                       gIds, gLs, gUs, gRs):
        g_ex = g_ex * gr
        l = max(l - g_ex, 1)
        u = u + g_ex
        rejects |= (group == g) & ((p_edge < t_pe) | (area < l) | (area > u))

    TP_mask = (~rejects) & (max_IoU >= IoU_thresh)

    # Find the maximum IoU in each eid-assignment group following the
    # suggestion on stackoverflow:
    # https://stackoverflow.com/questions/8623047/group-by-max-or-min-in-a-numpy-array
    # NB: the grouping variables and IoUs are presorted when
    # _filter_trial_cache is generated (see below).
    TPs_IoU = max_IoU[TP_mask]
    if TPs_IoU.size > 0:
        grouping = assignId[TP_mask]
        index = np.empty(TPs_IoU.size, 'bool')
        index[-1] = True
        last_in_group = np.any(grouping[1:] != grouping[:-1], axis=1)
        index[:-1] = last_in_group
        TPs_IoU = TPs_IoU[index]

    nPs = np.sum(~rejects)
    nTPs = TPs_IoU.size
    nFPs = nPs - nTPs
    nFNs = nT - nTPs
    precision = nTPs / (nTPs + nFPs)
    recall = nTPs / (nTPs + nFNs)
    # Fbeta = (1 + beta^2) * P * R / (beta^2 * P + R)
    F1 = 2 * precision * recall / (precision + recall)
    F0_5 = 1.25 * precision * recall / (0.25 * precision + recall)
    F2 = 5 * precision * recall / (4 * precision + recall)
    return precision, recall, F1, F0_5, F2, np.mean(TPs_IoU)


def _filter_trial_bootstraps(filter_trial_cache, pedge_thresh,
                             group_thresh_expansion, containment_thresh,
                             min_area, IoU_thresh, return_stderr=False):

    score_boots = []
    for df in filter_trial_cache:
        scr = _filter_trial(df['containment'], containment_thresh,
                            df['area'], min_area,
                            np.array(pedge_thresh, dtype='float'),
                            np.array(group_thresh_expansion,
                                     dtype='float'),
                            df['gIds'], df['gLs'], df['gUs'], df['gRs'],
                            df['group'], df['p_edge'], df['max_IoU'],
                            IoU_thresh, df['assignId'], df['nT'])
        score_boots.append(Score(*scr))

    score_boots = np.array(score_boots)
    mean_score = Score(*score_boots.mean(axis=0))
    if return_stderr:
        stderr = score_boots.std(axis=0) / np.sqrt(score_boots.shape[0])
        return (mean_score, Score(*stderr))
    else:
        return mean_score


def _generate_stat_table(s, seg_ex, segrps, containment_func):
    """Parallelisation helper for ``SegFilterParamOptim.generate_stat_table``."""

    ncells = len(seg_ex.target) if seg_ex.target.any() else 0

    # Perform within-group segmentation
    shape = np.squeeze(seg_ex.pred[0]).shape
    border_rect = np.pad(np.zeros(tuple(x - 2 for x in shape),
                                  dtype='bool'),
                         pad_width=1,
                         mode='constant',
                         constant_values=True)

    # Generate local versions of the Group segmenters to avoid potential race
    # conditions in parallel usage:
    segrps = [Group(grp.targets, grp.params) for grp in segrps]
    masks = []
    for grp in segrps:
        grp.segment(seg_ex.pred, border_rect)
        for cell in grp.cells:
            masks.append(cell.mask)

    # Calculate containment scores across groups
    contained_cells = {}
    paired_groups = zip(segrps, segrps[1:])
    for g, (lower_group, upper_group) in enumerate(paired_groups):
        for l, lower in enumerate(lower_group.cells):
            for u, upper in enumerate(upper_group.cells):
                containment = containment_func(lower.mask, upper.mask)
                if containment > 0:
                    if lower.edge_score > upper.edge_score:
                        contained_cells[(g + 1, u)] = containment
                    else:
                        contained_cells[(g, l)] = containment

    if ncells > 0:
        IoUs = calc_IoUs(seg_ex.target, masks, fill_holes=False)
        max_IoU = IoUs.max(axis=0)
        assignments = IoUs.argmax(axis=0)
        _, best_assignments = best_IoU(IoUs.T)
    else:
        max_IoU = np.zeros(len(masks))
        assignments = np.zeros(len(masks), dtype=np.uint16)
        best_assignments = -np.ones(len(masks), dtype=np.int32)

    ind = 0
    rows = []
    for g, grp in enumerate(segrps):
        for c, cell in enumerate(grp.cells):
            rows.append((s, g, c, cell.area, cell.edge_score,
                         contained_cells.get((g, c), 0.),
                         assignments[ind], max_IoU[ind],
                         best_assignments[ind]))
            ind += 1

    return (s, ncells), rows

def _seg_filter_optim(
        group_id,
        new_values,
        param_names,
        flattener,
        seg_gen,
        base_params=SegmentationParameters(),
        scoring='F0_5',
        n_jobs=4):
    """Parallelisation helper for ``SegmentationTrainer.fit_seg_params``."""

    # Replace default params with specified ones
    new_params = {}
    for k, v in zip(param_names, new_values):
        if k in BROADCASTABLE_PARAMS:
            if k not in new_params:
                gval = getattr(base_params, k)
                if type(gval) != list:
                    raise BadType('Segmentation parameters should '
                                  'have been broadcast.')
                new_params[k] = gval.copy()
            new_params[k][group_id] = v
        else:
            new_params[k] = v
    new_params = base_params._replace(**new_params)

    # Optimise filtering parameters and return score
    sfpo = SegFilterParamOptim(flattener, base_params=new_params,
                               scoring=scoring)
    with np.errstate(all='ignore'):
        sfpo.generate_stat_table(seg_gen, n_jobs=n_jobs)
        sfpo.fit_filter_params(lazy=True, bootstrap=False, n_jobs=n_jobs)
    return {
        'group': group_id,
        'base': {k: getattr(new_params, k) for k in param_names},
        'filter': sfpo.opt_params,
        'score': sfpo.opt_score,
        'scores': sfpo.filter_trial(**sfpo.opt_params, bootstrap=False)
    }


round_to_n = lambda x, n: round(x, -int(floor(log10(x))) + (n - 1))


class SegFilterParamOptim:
    """
    # TODO What does this class do
        * What are the parameters and what do they mean
        * What are the defaults, what are the ranges/admissible options?
    :param flattener:
    :param base_params:
    :param IoU_thresh:
    :param scoring:
    :param nbootstraps:
    :param bootstrap_frac:
    """
    def __init__(self,
                 flattener,
                 base_params=SegmentationParameters(),
                 IoU_thresh=0.5,
                 scoring='F0_5',
                 nbootstraps=10,
                 bootstrap_frac=0.9):

        self.IoU_thresh = IoU_thresh
        self.scoring = scoring
        self.nbootstraps = nbootstraps
        self.bootstrap_frac = bootstrap_frac

        # Force values for parameters optimised by this class
        self._base_params = base_params._replace(
            fit_radial=True, min_area=1, pedge_thresh=None,
            use_group_thresh=False, group_thresh_expansion=0,
            containment_thresh=0.8)

        self.segmenter = MorphSegGrouped(flattener,
                                         params=self._base_params,
                                         return_masks=True)

        self.group_info = []
        for g, group in enumerate(self.segrps):
            lower = min(
                target.definition.get('lower', 1.)
                for target in group.targets)
            upper = max(
                target.definition.get('upper', float('inf'))
                for target in group.targets)
            grange = lower if upper == float('inf') else upper - lower
            self.group_info.append((g, lower, upper, grange))

    @property
    def scoring(self):
        """ The scoring method used during evaluation of the segmentation.

        Accepted values are any of the named attributes of :py:class:`Score`
        specified as a ``str``.
        """
        return self._scoring

    @scoring.setter
    def scoring(self, val):
        if val not in Score._fields:
            raise BadParam('Specified scoring metric not available')
        self._scoring = val

    @property
    def base_params(self):
        return self._base_params

    @property
    def segrps(self):
        return self.segmenter.groups

    @property
    def stat_table(self):
        val = getattr(self, '_stat_table', None)
        if val is None:
            raise BadProcess('"generate_stat_table" has not been run')
        return val

    @property
    def stat_table_bootstraps(self):
        val = getattr(self, '_stat_table_bootstraps', None)
        if val is None:
            raise BadProcess('"generate_stat_table" has not been run')
        return val

    @property
    def truth(self):
        val = getattr(self, '_nPs', None)
        if val is None:
            raise BadProcess('"generate_stat_table" has not been run')
        return val

    @property
    def truth_bootstraps(self):
        val = getattr(self, '_nPs_bootstraps', None)
        if val is None:
            raise BadProcess('"generate_stat_table" has not been run')
        return val

    @property
    def opt_params(self):
        val = getattr(self, '_opt_params', None)
        if val is None:
            raise BadProcess('"fit_filter_params" has not been run')
        return val

    @property
    def opt_score(self):
        val = getattr(self, '_opt_score', None)
        if val is None:
            raise BadProcess('"fit_filter_params" has not been run')
        return val

    def generate_stat_table(self, example_gen, n_jobs=4):
        """Generates unfiltered segmentation results organised as a table.

        The generated output can be accessed as a ``pandas.DataFrame`` from
        :py:attr:`stat_table` and as a list of bootstrap samples from
        :py:attr:`stat_table_bootstraps`.

        Note:
            This function is called from
            :py:meth:`SegmentationTrainer.fit_seg_params` via
            :py:func:`_seg_filter_optim` and also from
            :py:meth:`SegmentationTrainer.refit_filter_seg_params`.
        """
        containment_func = getattr(segmentation,
                                   self.base_params.containment_func)
        rows_truth = []
        rows = []

        from joblib import Parallel, delayed
        rows = Parallel(n_jobs=n_jobs)(
            delayed(_generate_stat_table)(s, seg_ex, self.segrps,
                                          containment_func)
            for s, seg_ex in enumerate(example_gen))

        rows_truth, rows = zip(*rows)
        rows = list(chain(*rows))

        df_truth = pd.DataFrame(rows_truth, columns=['example', 'ncells'])
        df_truth = df_truth.set_index('example')
        self._nPs = df_truth.ncells.to_numpy()

        dtypes = [('example', np.uint16),
                  ('group', np.uint8),
                  ('cell', np.uint16),
                  ('area', np.uint16),
                  ('p_edge', np.float64),
                  ('containment', np.float64),
                  ('assignments', np.uint16),
                  ('max_IoU', np.float64),
                  ('best_assignments', np.int32)]
        df = pd.DataFrame(np.array(rows, dtype=dtypes))

        df['is_best'] = ((df.best_assignments >= 0) &
                         (df.max_IoU >= self.IoU_thresh))
        df['eid'] = df.example
        df['uid'] = tuple(zip(df.example, df.assignments))

        # Generate a set of bootstrapping filters over 90% of the examples
        examples = list(set(df_truth.index.values))
        nperboot = np.round(self.bootstrap_frac * len(examples)).astype(int)
        bootstraps = [
            np.random.choice(examples, nperboot, replace=True)
            for _ in range(self.nbootstraps)
        ]
        self._nPs_bootstraps = [df_truth.loc[b].sum() for b in bootstraps]
        # Limit bootstrap examples to those present in segmentation output
        bootstraps = [b[np.isin(b, df.example)] for b in bootstraps]
        df.set_index('example', drop=False, inplace=True)
        example_counts = df.example.value_counts()
        self._stat_table_bootstraps = []
        for b in bootstraps:
            df_boot = df.loc[b]
            # Renumber examples to handle the case of duplicated examples in
            # the bootstrap:
            df_boot['eid'] = tuple(
                chain(*(repeat(i, example_counts.loc[e])
                        for i, e in enumerate(b))))
            df_boot['uid'] = tuple(zip(df_boot.eid, df_boot.assignments))
            df_boot.set_index('uid', inplace=True)
            self._stat_table_bootstraps.append(df_boot)

        df.set_index('uid', inplace=True)
        self._stat_table = df

    def _ensure_filter_trial_cache(self, bootstrap):
        if getattr(self, '_filter_trial_cache') is None:
            if bootstrap:
                dfs = self.stat_table_bootstraps
                truths = self.truth_bootstraps
            else:
                dfs = [self.stat_table]
                truths = [self.truth.sum()]
            self._filter_trial_cache = []
            for df, nT in zip(dfs, truths):
                # From stackoverflow:
                # https://stackoverflow.com/questions/8623047/group-by-max-or-min-in-a-numpy-array
                # We need to pre-sort the DataFrame by a combined grouping
                # metric according to eid and assignments and IoU
                df = df.sort_values(['eid','assignments','max_IoU'])
                # Also need to reformat the group_info
                gIds, gLs, gUs, gRs = zip(*self.group_info)
                cache = dict(nT=nT, gIds=gIds,
                             gLs=np.array(gLs, dtype='float'),
                             gUs=np.array(gUs, dtype='float'),
                             gRs=np.array(gRs, dtype='float'),
                             containment=df.containment.to_numpy(),
                             area=df.area.to_numpy(),
                             p_edge=df.p_edge.to_numpy(),
                             group=df.group.to_numpy(),
                             max_IoU=df.max_IoU.to_numpy(),
                             assignId=df[['eid','assignments']].to_numpy())
                self._filter_trial_cache.append(cache)

    def filter_trial(self,
                     pedge_thresh,
                     group_thresh_expansion,
                     containment_thresh,
                     min_area,
                     bootstrap=True,
                     return_stderr=False,
                     use_cache=False):

        if not use_cache:
            self._filter_trial_cache = None
        self._ensure_filter_trial_cache(bootstrap)

        return _filter_trial_bootstraps(self._filter_trial_cache,
                                        pedge_thresh, group_thresh_expansion,
                                        containment_thresh, min_area,
                                        self.IoU_thresh,
                                        return_stderr=return_stderr)

    def parallel_filter_trials(self, params_list, score=None, bootstrap=True,
                               return_stderr=False, use_cache=False, n_jobs=4):
        if not use_cache:
            self._filter_trial_cache = None
        self._ensure_filter_trial_cache(bootstrap)
        cache = self._filter_trial_cache

        from joblib import Parallel, delayed
        mean_scores = Parallel(n_jobs=n_jobs)(
            delayed(_filter_trial_bootstraps)(cache, **params,
                                              IoU_thresh=self.IoU_thresh,
                                              return_stderr=return_stderr)
            for params in params_list)

        if return_stderr:
            mean_scores, stderrs = zip(*mean_scores)
            if score is not None:
                mean_scores = [getattr(s, score) for s in mean_scores]
                stderrs = [getattr(s, score) for s in stderrs]
            return mean_scores, stderrs
        else:
            if score is not None:
                mean_scores = [getattr(s, score) for s in mean_scores]
            return mean_scores

    def fit_filter_params(self, lazy=False, bootstrap=False, n_jobs=4):
        # Define parameter grid values, firstly those not specific to a group
        params = {
            ('containment_thresh', None): np.linspace(0, 1, 21),
            ('min_area', None): np.arange(0, 20, 1)
        }

        # Ensure we reset the filter trial cache
        self._filter_trial_cache = None
        self._ensure_filter_trial_cache(bootstrap)

        # Determine the pedge_threshold range based on the observed p_edge
        # range for each group
        if (self.stat_table.is_best.all() or not self.stat_table.is_best.any()):
            t_pe_upper = self.stat_table.groupby('group').p_edge.mean()
        else:
            q_pe = self.stat_table.groupby(['group', 'is_best'])
            q_pe = q_pe.p_edge.quantile([0.25, 0.95]).unstack((1, 2))
            t_pe_upper = q_pe.loc[:, [(False, 0.95), (True, 0.25)]].mean(1)

        t_pe_vals = [
            np.arange(0, u, round_to_n(u / 20, 1)) for u in t_pe_upper
        ]

        # Set group-specific parameter grid values
        g_ex_vals = repeat(np.linspace(0, 0.4, 21))
        for g, (t_pe, g_ex) in enumerate(zip(t_pe_vals, g_ex_vals)):
            params[('pedge_thresh', g)] = t_pe
            params[('group_thresh_expansion', g)] = g_ex

        # Default starting point is with thresholds off and no group expansion
        ngroups = len(self.segrps)
        dflt_params = {
            'containment_thresh': 0,
            'min_area': 0,
            'pedge_thresh': list(repeat(0, ngroups)),
            'group_thresh_expansion': list(repeat(0, ngroups))
        }

        # Search first along each parameter dimension with all others kept at
        # default:
        opt_params = {}
        for k, pvals in params.items():
            params_list = [_sub_params({k: v}, dflt_params) for v in pvals]
            scrs = self.parallel_filter_trials(params_list,
                                               score=self.scoring,
                                               bootstrap=bootstrap,
                                               use_cache=True, n_jobs=n_jobs)
            maxInd = np.argmax(scrs)
            opt_params[k] = pvals[maxInd]

        # Reset the template parameters to the best along each dimension
        base_params = _sub_params(opt_params, dflt_params)

        if lazy:
            # Simply repeat search along each parameter dimension, but now
            # using the new optimum as a starting point
            opt_params = {}
            for k, pvals in params.items():
                params_list = [_sub_params({k: v}, base_params) for v in pvals]
                scrs = self.parallel_filter_trials(params_list,
                                                   score=self.scoring,
                                                   bootstrap=bootstrap,
                                                   use_cache=True, n_jobs=n_jobs)
                maxInd = np.argmax(scrs)
                opt_params[k] = pvals[maxInd]
            opt_params = _sub_params(opt_params, base_params)
            scr = self.filter_trial(**opt_params, bootstrap=bootstrap,
                                    use_cache=True)
            self._opt_params = opt_params
            self._opt_score = getattr(scr, self.scoring)
            return

        # Next perform a joint search for parameters with optimal pairings
        opt_param_pairs = {k: {v} for k, v in opt_params.items()}
        for k1, k2 in combinations(params.keys(), 2):
            vals = list(product(params[k1], params[k2]))
            params_list = [_sub_params({ k1: v1, k2: v2 }, base_params)
                          for v1, v2 in vals]
            scrs = self.parallel_filter_trials(params_list,
                                               score=self.scoring,
                                               bootstrap=bootstrap,
                                               use_cache=True, n_jobs=n_jobs)
            maxInd = np.argmax(scrs)
            p1opt, p2opt = vals[maxInd]
            opt_param_pairs[k1].add(p1opt)
            opt_param_pairs[k2].add(p2opt)

        # Finally search over all combinations of the parameter values found
        # with optimal pairings
        params_list = [_sub_params({k: v for k, v in
                                    zip(opt_param_pairs.keys(), pvals)},
                                   base_params)
                       for pvals in product(*opt_param_pairs.values())]
        scrs = self.parallel_filter_trials(params_list,
                                           score=self.scoring,
                                           bootstrap=bootstrap,
                                           use_cache=True, n_jobs=n_jobs)
        maxInd = np.argmax(scrs)
        self._opt_params = params_list[maxInd]
        self._opt_score = scrs[maxInd]


class SegmentationTrainer(object):
    """Finds optimal segmentation parameters given a trained CNN.

    Args:
        shared_params: Training and segmentation parameters as provided by
            :py:class:`utils.SharedParameterContainer`.
        shared_data: Training data as provided by
            :py:class:`utils.SharedDataContainer`.
        ssm_trainer: SmoothingModelTrainer with optimised model for
            determination of smoothing sigma.
        cnn_trainer: Trainer with optimised CNN.
    """
    def __init__(self,
                 shared_params: SharedParameterContainer,
                 shared_data: SharedDataContainer,
                 ssm_trainer: SmoothingModelTrainer,
                 cnn_trainer: CNNTrainer):
        self._shared_params = shared_params
        self._shared_data = shared_data
        self._ssm_trainer = ssm_trainer
        self._cnn_trainer = cnn_trainer

    @property
    def save_dir(self):
        """Base directory in which to save trained models"""
        return self._shared_params.save_dir

    @property
    def training_parameters(self):
        return self._shared_params.parameters

    @property
    def segment_parameters(self):
        return self._shared_params.segmentation_parameters

    @segment_parameters.setter
    def segment_parameters(self, val):
        self._shared_params.segmentation_parameters = val

    @property
    def segment_parameter_coords(self):
        param_coords = DEFAULT_SEG_PARAM_COORDS.copy()
        param_coords.update(self.training_parameters.seg_param_coords)
        return param_coords

    @property
    def gen(self):
        """Training, validation and test data generators with raw output.

        This attribute provides three :py:class:`ImageLabel` generators as a
        :py:class:`TrainValTestProperty`, with each generator assigned a
        :py:class:`ScalingAugmenter` that outputs an ``(image, label, info)``
        tuple, where ``label`` provides the unflattened ``ndarray`` of cell
        outlines and ``info`` is a ``dict`` of meta-data associated with the
        label (if any). Augmentations are limited to just cropping and scaling
        operations to match the intended pixel size and input size of the CNN.
        """
        # Create an augmenter that returns unflattened label images
        aug = standard_augmenter(
            self._ssm_trainer.model,
            lambda lbl, _: lbl,
            self.training_parameters,
            isval=True)

        def seg_example_aug(img, lbl):
            # Assume that the label preprocessing function also returns info
            _, info = lbl
            img, lbl = aug(img, lbl)
            return img, (lbl > 0, info)

        # In this case, always prefer the validation augmenter
        return self._shared_data.gen_with_aug(seg_example_aug)

    @property
    def examples(self):
        """Training, validation and test segmentation example generators.

        This attribute provides three generators as a
        :py:class:`TrainValTestProperty`, with each generator yielding
        a :py:class:`SegExample` for each image-label pair in the training,
        validation or test image data collections.
        """
        # Ensure that the saved generators are updated if more data is
        # added...
        if getattr(self, '_ncells', None) != self._shared_data.data.ncells:
            self._seg_examples = None
            self._ncells = self._shared_data.data.ncells
        # ...or if data generation parameters change:
        old_gen_params = getattr(self, '_current_gen_params', None)
        new_gen_params = tuple(getattr(self.training_parameters, p) for p in
                               ('in_memory', 'input_norm_dw', 'batch_size',
                                'balanced_sampling', 'use_sample_weights',
                                'xy_out', 'target_pixel_size', 'substacks'))
        if old_gen_params != new_gen_params:
            self._seg_examples = None
            self._current_gen_params = new_gen_params

        cnn = self._cnn_trainer.opt_cnn
        gen = self.gen
        if self.training_parameters.in_memory:
            if getattr(self, '_seg_examples', None) is None:
                self._seg_examples = TrainValTestProperty(
                    list(_example_generator(cnn, gen.train)),
                    list(_example_generator(cnn, gen.val)),
                    list(_example_generator(cnn, gen.test)))
            return TrainValTestProperty(
                (e for e in self._seg_examples.train),
                (e for e in self._seg_examples.val),
                (e for e in self._seg_examples.test))
        else:
            self._seg_examples = None
            return TrainValTestProperty(
                _example_generator(cnn, gen.train),
                _example_generator(cnn, gen.val),
                _example_generator(cnn, gen.test))

    @property
    def flattener(self):
        return self._cnn_trainer.flattener

    @property
    def seg_param_stats(self):
        if getattr(self, '_seg_param_stats', None) is None:
            seg_stats_file = self.training_parameters.segmentation_stats_file
            stats_file = self.save_dir / seg_stats_file
            if not stats_file.is_file():
                raise BadProcess('"fit_seg_params" has not been run yet')
            self._seg_param_stats = pd.read_csv(stats_file, index_col=0)
        return self._seg_param_stats

    def fit_seg_params(self, n_jobs=None, scoring='F0_5', subsample_frac=1.,
                       fit_on_split='val'):
        """Find optimal segmentation hyperparameters.

        Args:
            njobs (int): Number of parallel processes to run.
            scoring (str): Scoring metric to be used to assess segmentation
                performance. The name of any of the attributes in
                :py:class:`Score` may be specified.
        """

        if n_jobs is None:
            n_jobs = self.training_parameters.n_jobs

        # Initialise the default segmenter to determine the number of groups
        # and obtain broadcast base parameters:
        segmenter = MorphSegGrouped(
            self.flattener,
            params=self.segment_parameters)
        ngroups = len(segmenter.groups)
        base_params = segmenter.params

        # Generate parameter search grid according to training parameters
        param_coords = self.segment_parameter_coords
        param_grid = list(product(*param_coords.values()))
        par_names = list(param_coords.keys())

        if type(subsample_frac) == float:
            subsample_frac = (subsample_frac,)
        if type(fit_on_split) == str:
            fit_on_split = (fit_on_split,)
        if len(subsample_frac) == 1 and len(fit_on_split) > 1:
            subsample_frac = subsample_frac * len(fit_on_split)
        if len(fit_on_split) == 1 and len(subsample_frac) > 1:
            fit_on_split = fit_on_split * len(subsample_frac)

        examples = []
        for ssf, split in zip(subsample_frac, fit_on_split):
            step = max(int(np.floor(1. / ssf)), 1)
            # NB: the following essentially assumes that examples are
            # presented in random order (see _example_generator)
            examples.extend(list(islice(
                getattr(self.examples, split), None, None, step)))

        rows = []
        for gind in range(ngroups)[::-1]:
            for pars in tqdm(param_grid):
                rows.append(_seg_filter_optim(gind, pars, par_names,
                                              self.flattener, examples,
                                              base_params=base_params,
                                              scoring=scoring, n_jobs=n_jobs))

        rows_expanded = []
        for row in rows:
            row_details = chain(
                [('group', row['group']), ('score', row['score'])],
                row['scores']._asdict().items(),
                row['base'].items(), row['filter'].items())
            row_expanded = []
            for k, v in row_details:
                if k in BROADCASTABLE_PARAMS and type(v) is list:
                    kvpairs = [('_'.join((k, str(g))), gv)
                               for g, gv in enumerate(v)]
                else:
                    kvpairs = [(k, v)]
                row_expanded.extend(kvpairs)
            rows_expanded.append(dict(row_expanded))

        # TODO: if None values are combined with integer values, the entire
        # column gets converted here to float, with the None values as NaN.
        # This causes errors, for example, with specification of
        # edge_sub_dilations. There is currently no obvious solution to this.
        self._seg_param_stats = pd.DataFrame(rows_expanded)
        stats_file = (self.save_dir /
                      self.training_parameters.segmentation_stats_file)
        self._seg_param_stats.to_csv(stats_file)

        self.refit_filter_seg_params(scoring=scoring, n_jobs=n_jobs)

    def refit_filter_seg_params(self,
                                lazy=False,
                                bootstrap=False,
                                scoring='F0_5',
                                n_jobs=None):

        if n_jobs is None:
            n_jobs = self.training_parameters.n_jobs

        # Initialise the default segmenter to determine the number of groups
        # and obtain broadcast base parameters:
        segmenter = MorphSegGrouped(
            self.flattener,
            params=self.segment_parameters)
        ngroups = len(segmenter.groups)
        base_params = segmenter.params

        # Merge the best parameters from each group into a single parameter set
        par_names = list(self.segment_parameter_coords.keys())
        broadcast_par_names = [k for k in par_names
                               if k in BROADCASTABLE_PARAMS]
        merged_params = {k: getattr(base_params, k) for k in par_names}
        for k in broadcast_par_names:
            merged_params[k] = merged_params[k].copy()
        stats = self.seg_param_stats
        for g, r in enumerate(stats.groupby('group').score.idxmax()):
            for k in broadcast_par_names:
                merged_params[k][g] = stats.loc[r, k + '_' + str(g)]
        merged_params = base_params._replace(**merged_params)

        sfpo = SegFilterParamOptim(self.flattener,
                                   base_params=merged_params,
                                   scoring=scoring)
        with np.errstate(all='ignore'):
            sfpo.generate_stat_table(list(self.examples.val), n_jobs=n_jobs)
            sfpo.fit_filter_params(lazy=lazy, bootstrap=bootstrap,
                                   n_jobs=n_jobs)

        self.segment_parameters = merged_params._replace(**sfpo.opt_params)

    def validate_seg_params(
            self,
            iou_thresh=0.7,
            save=True,
            refine_outlines=True):
        segmenter = MorphSegGrouped(self.flattener,
                                    params=self.segment_parameters,
                                    return_masks=True)
        edge_inds = [
            i for i, t in enumerate(self.flattener.targets)
            if t.prop == 'edge'
        ]
        stats = {}
        dfs = {}
        for k, seg_exs in zip(self.examples._fields, self.examples):
            stats[k] = []
            for seg_ex in seg_exs:
                seg = segmenter.segment(seg_ex.pred,
                                        refine_outlines=refine_outlines)
                edge_scores = np.array([
                    seg_ex.pred[edge_inds, ...].max(axis=0)[s].mean()
                    for s in seg.edges
                    ])
                IoUs = calc_IoUs(seg_ex.target, seg.masks)
                bIoU, _ = best_IoU(IoUs)
                stats[k].append((edge_scores, IoUs, np.mean(bIoU),
                    np.min(bIoU, initial=1),
                    calc_AP(IoUs,
                        probs=edge_scores,
                        iou_thresh=iou_thresh)[0]))
            dfs[k] = pd.DataFrame([s[2:] for s in stats[k]],
                    columns=['IoU_mean', 'IoU_min', 'AP'])

        print({k: df.mean() for k, df in dfs.items()})

        nrows = len(dfs)
        ncols = dfs['val'].shape[1]
        fig, axs = plt.subplots(nrows=nrows,
                ncols=ncols,
                figsize=(ncols * 4, nrows * 4))
        for axrow, (k, df) in zip(axs, dfs.items()):
            for ax, col in zip(axrow, df.columns):
                ax.hist(df.loc[:, col], bins=26, range=(0, 1))
                ax.set(xlabel=col, title=k)
        if save:
            fig.savefig(self.save_dir / 'seg_validation_plot.png')
            plt.close(fig)

