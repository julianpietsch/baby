from math import floor, log10
from itertools import combinations, product, repeat
import numpy as np
import pandas as pd

from baby.segmentation import mask_containment
from baby.morph_thresh_seg import MorphSegGrouped
from baby.performance import calc_IoUs, best_IoU
from baby.brain import default_params
from baby.errors import BadProcess

BASIC_PARAMS = {
    'interior_threshold', 'nclosing', 'nopening', 'connectivity',
    'edge_sub_dilations'
}

round_to_n = lambda x, n: round(x, -int(floor(log10(x))) + (n - 1))


class SegFilterParamOptim:

    def __init__(self, flattener, basic_params={}, IoU_thresh=0.5):
        self.IoU_thresh = IoU_thresh
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
    def opt_params(self):
        val = getattr(self, '_opt_params', None)
        if val is None:
            raise BadProcess('"fit_filter_params" has not been run')
        return val

    @property
    def opt_F1(self):
        val = getattr(self, '_opt_F1', None)
        if val is None:
            raise BadProcess('"fit_filter_params" has not been run')
        return val

    def generate_stat_table(self, example_gen):
        self._nT = 0
        rows = []
        for s, seg_ex in enumerate(example_gen):
            self._nT += len(seg_ex.target)

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

            IoUs = calc_IoUs(seg_ex.target, masks, fill_holes=False)
            max_IoU = IoUs.max(axis=0)
            assignments = IoUs.argmax(axis=0)
            _, best_assignments = best_IoU(IoUs.T)
            ind = 0
            for g, group in enumerate(self.segrps):
                for c, cell in enumerate(group.cells):
                    rows.append((s, g, c, cell.area, cell.edge_score,
                                 contained_cells.get((g, c),
                                                     0.), assignments[ind],
                                 max_IoU[ind], best_assignments[ind]))
                    ind += 1

        df = pd.DataFrame(rows,
                          columns=[
                              'example', 'group', 'cell', 'area', 'p_edge',
                              'containment', 'assignments', 'max_IoU',
                              'best_assignments'
                          ])
        df['is_best'] = ((df.best_assignments >= 0) &
                         (df.max_IoU >= self.IoU_thresh))
        self._stat_table = df

    def filter_trial(self, p_edge_thresh, group_expansion, containment_thresh,
                     min_area):
        df = self.stat_table
        rejects = (df.containment > containment_thresh) | (df.area < min_area)
        for t_pe, g_ex, (g, l, u, gr) in zip(p_edge_thresh, group_expansion,
                                             self.group_info):
            g_ex = g_ex * gr
            l = max(l - g_ex, 1)
            u = u + g_ex
            rejects |= (df.group == g) & ((df.p_edge < t_pe) | (df.area < l) |
                                          (df.area > u))
        TP_mask = (~rejects) & (df.max_IoU >= self.IoU_thresh)
        TPs_IoU = df.loc[TP_mask].groupby(['example',
                                           'assignments']).max_IoU.max()
        nPs = sum(~rejects)
        nTPs = len(TPs_IoU)
        nFPs = nPs - nTPs
        nFNs = self._nT - nTPs
        precision = nTPs / (nTPs + nFPs)
        recall = nTPs / (nTPs + nFNs)
        F1 = 2 * precision * recall / (precision + recall)
        return precision, recall, F1, TPs_IoU.mean()

    def fit_filter_params(self, lazy=False):
        # Define parameter grid values, firstly those not specific to a group
        params = {
            ('containment_thresh', None): np.linspace(0, 1, 21),
            ('min_area', None): np.arange(0, 20, 1)
        }

        # Determine the p_edge_threshold range based on the observed p_edge
        # range for each group
        q_pe = self.stat_table.groupby(['group', 'is_best'])
        q_pe = q_pe.p_edge.quantile([0.25, 0.95]).unstack((1, 2))
        t_pe_upper = q_pe.loc[:, [(False, 0.95), (True, 0.25)]].mean(1)
        t_pe_vals = [
            np.arange(0, u, round_to_n(u / 20, 1)) for u in t_pe_upper
        ]

        # Set group-specific parameter grid values
        g_ex_vals = repeat(np.linspace(0, 0.4, 21))
        for g, (t_pe, g_ex) in enumerate(zip(t_pe_vals, g_ex_vals)):
            params[('p_edge_thresh', g)] = t_pe
            params[('group_expansion', g)] = g_ex

        # Default starting point is with thresholds off and no group expansion
        ngroups = len(self.segrps)
        dflt_params = {
            'containment_thresh': 0,
            'min_area': 0,
            'p_edge_thresh': list(repeat(0, ngroups)),
            'group_expansion': list(repeat(0, ngroups))
        }

        # Search first along each parameter dimension with all others kept at
        # default:
        opt_params = {}
        for k, pvals in params.items():
            F1s = []
            for v in pvals:
                p = _sub_params({k: v}, dflt_params)
                _, _, F1, _ = self.filter_trial(**p)
                F1s.append(F1)
            maxInd = np.argmax(F1s)
            opt_params[k] = pvals[maxInd]

        # Reset the template parameters to the best along each dimension
        base_params = _sub_params(opt_params, dflt_params)

        if lazy:
            # Simply repeat search along each parameter dimension, but now
            # using the new optimum as a starting point
            opt_params = {}
            for k, pvals in params.items():
                F1s = []
                for v in pvals:
                    p = _sub_params({k: v}, base_params)
                    _, _, F1, _ = self.filter_trial(**p)
                    F1s.append(F1)
                maxInd = np.argmax(F1s)
                opt_params[k] = pvals[maxInd]
            opt_params = _sub_params(opt_params, base_params)
            _, _, F1, _ = self.filter_trial(**opt_params)
            self._opt_params = opt_params
            self._opt_F1 = F1
            return

        # Next perform a joint search for parameters with optimal pairings
        opt_param_pairs = {k: {v} for k, v in opt_params.items()}
        for k1, k2 in combinations(params.keys(), 2):
            F1s = [(v1, v2,
                    self.filter_trial(**_sub_params({
                        k1: v1,
                        k2: v2
                    }, base_params))[2])
                   for v1, v2 in product(params[k1], params[k2])]
            p1opt, p2opt, F1opt = max(F1s, key=lambda x: x[2])
            opt_param_pairs[k1].add(p1opt)
            opt_param_pairs[k2].add(p2opt)

        # Finally search over all combinations of the parameter values found
        # with optimal pairings
        F1s = []
        for pvals in product(*opt_param_pairs.values()):
            p = {k: v for k, v in zip(opt_param_pairs.keys(), pvals)}
            p = _sub_params(p, base_params)
            F1s.append((p, self.filter_trial(**p)[2]))

        self._opt_params, self._opt_F1 = max(F1s, key=lambda x: x[1])


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
