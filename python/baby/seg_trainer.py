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
from math import floor, log10
from itertools import combinations, product, chain, repeat
from typing import NamedTuple, Union, Tuple, Any
import numpy as np
np.seterr(all='ignore')
import pandas as pd

from baby.segmentation import mask_containment
from baby.morph_thresh_seg import MorphSegGrouped
from baby.performance import calc_IoUs, best_IoU
from baby.brain import default_params
from baby.errors import BadProcess, BadParam

BASIC_PARAMS = {
    'interior_threshold', 'nclosing', 'nopening', 'connectivity',
    'edge_sub_dilations'
}

round_to_n = lambda x, n: round(x, -int(floor(log10(x))) + (n - 1))


class Score(NamedTuple):
    precision: float
    recall: float
    F1: float
    F0_5: float
    F2: float
    meanIoU: float


class SegFilterParamOptim:
    """
    # TODO What does this class do
        * What are the parameters and what do they mean
        * What are the defaults, what are the ranges/admissible options?
    :param flattener:
    :param basic_params:
    :param IoU_thresh:
    :param scoring:
    :param nbootstraps:
    :param bootstrap_frac:
    """
    def __init__(self,
                 flattener,
                 basic_params={},
                 IoU_thresh=0.5,
                 scoring='F0_5',
                 nbootstraps=10,
                 bootstrap_frac=0.9):

        self.IoU_thresh = IoU_thresh
        self.scoring = scoring
        self.nbootstraps = nbootstraps
        self.bootstrap_frac = bootstrap_frac
        self._basic_params = default_params.copy()
        self._basic_params.update(
            {k: v for k, v in basic_params.items() if k in BASIC_PARAMS})
        self._basic_params.update({
            'fit_radial': True,
            'min_area': 1,
            'pedge_thresh': None,
            'use_group_thresh': False
        })
        self.segmenter = MorphSegGrouped(flattener,
                                         return_masks=True,
                                         **self._basic_params)

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
        Accepted values are: # TODO define the scoring metrics
        * precision:
        * recall:
        * F1:
        * F0_5:
        * F2:
        * meanIoU:
        :return: str scoring method
        """
        return self._scoring

    @scoring.setter
    def scoring(self, val):
        if val not in Score._fields:
            raise BadParam('Specified scoring metric not available')
        self._scoring = val

    @property
    def basic_params(self):
        return self._basic_params

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

    def generate_stat_table(self, example_gen):
        rows_truth = []
        rows = []
        for s, seg_ex in enumerate(example_gen):
            ncells = len(seg_ex.target) if seg_ex.target.any() else 0
            rows_truth.append((s, ncells))

            # Perform within-group segmentation
            shape = np.squeeze(seg_ex.pred[0]).shape
            border_rect = np.pad(np.zeros(tuple(x - 2 for x in shape),
                                          dtype='bool'),
                                 pad_width=1,
                                 mode='constant',
                                 constant_values=True)
            masks = []
            for group in self.segrps:
                group.segment(seg_ex.pred, border_rect, fit_radial=True)
                for cell in group.cells:
                    masks.append(cell.mask)

            # Calculate containment scores across groups
            contained_cells = {}
            paired_groups = zip(self.segrps, self.segrps[1:])
            for g, (lower_group, upper_group) in enumerate(paired_groups):
                for l, lower in enumerate(lower_group.cells):
                    for u, upper in enumerate(upper_group.cells):
                        containment = mask_containment(lower.mask, upper.mask)
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
            for g, group in enumerate(self.segrps):
                for c, cell in enumerate(group.cells):
                    rows.append((s, g, c, cell.area, cell.edge_score,
                                 contained_cells.get((g, c),
                                                     0.), assignments[ind],
                                 max_IoU[ind], best_assignments[ind]))
                    ind += 1

        df_truth = pd.DataFrame(rows_truth, columns=['example', 'ncells'])
        df_truth = df_truth.set_index('example')
        self._nPs = df_truth.ncells

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

    def filter_trial(self,
                     pedge_thresh,
                     group_thresh_expansion,
                     containment_thresh,
                     min_area,
                     bootstrap=True,
                     return_stderr=False):
        if bootstrap:
            dfs = self.stat_table_bootstraps
            truths = self.truth_bootstraps
        else:
            dfs = [self.stat_table]
            truths = [self.truth.sum()]

        uidcols = ['eid', 'assignments']
        score_boots = []
        for df, nT in zip(dfs, truths):
            rejects = ((df.containment > containment_thresh) |
                       (df.area < min_area))
            for t_pe, g_ex, (g, l, u, gr) in zip(pedge_thresh,
                                                 group_thresh_expansion,
                                                 self.group_info):
                g_ex = g_ex * gr
                l = max(l - g_ex, 1)
                u = u + g_ex
                rejects |= (df.group == g) & ((df.p_edge < t_pe) |
                                              (df.area < l) | (df.area > u))
            TP_mask = (~rejects) & (df.max_IoU >= self.IoU_thresh)

            # # TODO compare speed of:
            # TPs_IoU = df.loc[TP_mask].groupby(uidcols).max_IoU.max()
            # # with speed of:
            TPs_IoU = []
            current_eid = 0
            asgn = {}
            for m, e, a, iou in zip(TP_mask, df.eid, df.assignments,
                    df.max_IoU):
                if e != current_eid:
                    TPs_IoU.extend(asgn.values())
                    current_eid = e
                    asgn = {}
                if not m:
                    continue
                asgn[a] = max(iou, asgn.get(a, 0))
            TPs_IoU.extend(asgn.values())
            # # END TODO

            nPs = np.sum(~rejects)
            nTPs = len(TPs_IoU)
            nFPs = nPs - nTPs
            nFNs = nT - nTPs
            precision = nTPs / (nTPs + nFPs)
            recall = nTPs / (nTPs + nFNs)
            # Fbeta = (1 + beta^2) * P * R / (beta^2 * P + R)
            F1 = 2 * precision * recall / (precision + recall)
            F0_5 = 1.25 * precision * recall / (0.25 * precision + recall)
            F2 = 5 * precision * recall / (4 * precision + recall)
            score_boots.append(
                Score(precision, recall, F1, F0_5, F2, np.mean(TPs_IoU)))

        score_boots = np.array(score_boots)
        mean_score = Score(*score_boots.mean(axis=0))
        if return_stderr:
            stderr = score_boots.std(axis=0) / np.sqrt(score_boots.shape[0])
            return (mean_score, Score(*stderr))
        else:
            return mean_score

    def fit_filter_params(self, lazy=False, bootstrap=False):
        # Define parameter grid values, firstly those not specific to a group
        params = {
            ('containment_thresh', None): np.linspace(0, 1, 21),
            ('min_area', None): np.arange(0, 20, 1)
        }

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
            scrs = []
            for v in pvals:
                p = _sub_params({k: v}, dflt_params)
                scr = self.filter_trial(**p, bootstrap=bootstrap)
                scrs.append(getattr(scr, self.scoring))
            maxInd = np.argmax(scrs)
            opt_params[k] = pvals[maxInd]

        # Reset the template parameters to the best along each dimension
        base_params = _sub_params(opt_params, dflt_params)

        if lazy:
            # Simply repeat search along each parameter dimension, but now
            # using the new optimum as a starting point
            opt_params = {}
            for k, pvals in params.items():
                scrs = []
                for v in pvals:
                    p = _sub_params({k: v}, base_params)
                    scr = self.filter_trial(**p, bootstrap=bootstrap)
                    scrs.append(getattr(scr, self.scoring))
                maxInd = np.argmax(scrs)
                opt_params[k] = pvals[maxInd]
            opt_params = _sub_params(opt_params, base_params)
            scr = self.filter_trial(**opt_params, bootstrap=bootstrap)
            self._opt_params = opt_params
            self._opt_score = getattr(scr, self.scoring)
            return

        # Next perform a joint search for parameters with optimal pairings
        opt_param_pairs = {k: {v} for k, v in opt_params.items()}
        for k1, k2 in combinations(params.keys(), 2):
            scrs = [(v1, v2,
                     getattr(
                         self.filter_trial(**_sub_params({
                             k1: v1,
                             k2: v2
                         }, base_params),
                                           bootstrap=bootstrap),
                         self.scoring))
                    for v1, v2 in product(params[k1], params[k2])]
            p1opt, p2opt, _ = max(scrs, key=lambda x: x[2])
            opt_param_pairs[k1].add(p1opt)
            opt_param_pairs[k2].add(p2opt)

        # Finally search over all combinations of the parameter values found
        # with optimal pairings
        scrs = []
        for pvals in product(*opt_param_pairs.values()):
            p = {k: v for k, v in zip(opt_param_pairs.keys(), pvals)}
            p = _sub_params(p, base_params)
            scrs.append((p,
                         getattr(self.filter_trial(**p, bootstrap=bootstrap),
                                 self.scoring)))

        self._opt_params, self._opt_score = max(scrs, key=lambda x: x[1])


def _sub_params(sub, param_template):
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
