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
from warnings import warn
import numpy as np
# import gaussianprocessderivatives as gp
import gaussianprocessderivatives as gp
import h5py
from tqdm import trange, tqdm
from collections import namedtuple

def by_thresh(diff, gr, start, thresh, max_thresh, d_idx=None, debug=False):
    """Recursively define a crossing point based on a threshold range. 
    
    Requirements for a valid 'crossing' point: 
    * the difference between mother and daughter growth rate (diff) is less than the threshold
    * the daughter growth rate (gr) is greater than the threshold
    
    The range of thresholds from thresh to max_thresh is checked from low to high.
    As soon as a point satisfies the requirements above, a value is returend. 
    If there is no valid point for any of the thresholds within the range, 0 is returned
    """
    # In case there are NaN values, we can safely ignore the warning that
    # usually comes up
    with np.errstate(invalid='ignore'):
        zero_crossings = np.where((np.abs(diff) <= thresh) & (gr >= 1.0 * thresh))[0]
    if len(zero_crossings) > 0: 
        if debug:
            print(f"{thresh} succeeded: {zero_crossings[0] + start}")
        return zero_crossings[0] + start
    elif thresh < max_thresh: 
        if debug:
            print(f"{thresh} failed for {d_idx}, dist is {diff}")
        return by_thresh(diff, gr, start, thresh + 1, max_thresh, d_idx, debug)
    else:
        if debug:
            print(f"All thresholds failed")
        return None


def cytokinesis_from_growth_rate_crossing(d_gr, m_gr, start, end,
        d_idx=None, ofs=2, lb=2, ub=15, debug=False):
    """Find cytokinesis by growth rate crossing within a range of thresholds. 
    
    This method enforces that cytokinesis is not within ofs time points of start or end.
    It only uses the mean growth rate values to determine the crossing point.
    """
    ostart = start + ofs
    oend = end - ofs
    diff = d_gr[ostart:oend] - m_gr[ostart:oend]
    cyto = by_thresh(diff, d_gr[ostart:oend], ostart, lb, ub, d_idx, debug)
    return cyto


def dist(x, std_x, y, std_y):
    """Distance between two gaussian variables given mean and standard deviation."""
    return np.sqrt(std_x**2 + std_y**2 + (x-y)**2)


def cytokinesis_from_growth_rate(d_gr, d_std, m_gr, m_std,
        start, end, d_idx=None, ofs=2, lb=2, ub=15, debug=False):
    """Find cytokinesis by growth rate crossing within a range of thresholds, noisy.
    
    This method enforces that cytokinesis is not within ofs time points of start or end.
    It uses both the mean and standard deviation of growth rate to determine the crossing point.
    """
    ostart = start + ofs
    oend = end - ofs
    diff = d_gr[ostart:oend] - m_gr[ostart:oend]
    diff = dist(d_gr[ostart:oend], d_std[ostart:oend], m_gr[ostart:oend], m_std[ostart:oend])
    cyto = by_thresh(diff, d_gr[ostart:oend], ostart, lb, ub, d_idx, debug)
    return cyto


def growth_rates_from_volumes(times, volumes):
    fvals = ['f', 'df']
    fits = {}
    for f in fvals:
        fits[f] = {}
        fits[f]['mean'] = np.full(volumes.shape, np.nan)
        fits[f]['std'] = np.full(volumes.shape, np.nan)

    # Use parameter bounds suited to raw and not log volume
    hyperpar_bounds = {0: (-2, 3), 1: (-2, 0), 2: (0, 3)}

    # Loop over all cells and fit Gaussian process
    for i in range(volumes.shape[0]):
        iV = volumes[i]
        i_filt = ~np.isnan(iV)
        if i_filt.sum() < 3:
            continue

        i_gp = gp.maternGP(hyperpar_bounds, times[i_filt], iV[i_filt])
        i_gp.findhyperparameters(noruns=5)
        i_gp.predict(times[i_filt], derivs=1)
        for f in fvals:
            fits[f]['mean'][i, i_filt] = getattr(i_gp, f)
            fits[f]['std'][i, i_filt] = np.sqrt(getattr(i_gp, f + 'var'))

    gr_tuple = namedtuple('gp_out', 'vol, vol_std, grs, grs_std')
    return gr_tuple(*fits['f'].values(), *fits['df'].values())


def fit_gr_data(filename, run_moth=True, run_dght=True, split_at_birth=False,
        log_volume=False):
    data = h5py.File(filename, 'a')
    if log_volume:
        hyperpar_bounds = {0: (-2, 1), 1: (-2, 2), 2: (-4, -2)}
    else:
        hyperpar_bounds = {0: (-2, 3), 1: (-2, 0), 2: (0, 3)}

    if run_moth:
        allT = data['M/times'][()]
        allV = data['M/vol'][()]
        if log_volume:
            allV = np.log(allV)

        if split_at_birth:
            ntps, ncells = allV.shape
            assert ncells < 2**16 - 1
            pnum = data['M/posNum'][()].astype(int).flatten() - 1
            tnum = data['M/trapNum'][()].astype(int).flatten() - 1
            cnum = data['M/cellNum'][()].astype(int).flatten() - 1
            mnum = data['M/mothers'][()].flatten()
            dlabs = data['M/daughterLabel'][()].astype(int)
            births = data['M/births'][()].astype(bool)

            cmap = np.zeros((np.max(pnum) + 1, np.max(tnum) + 1,
                np.max(cnum) + 1), dtype='uint16')
            cmap[pnum, tnum, cnum] = np.arange(1, ncells + 1)
            mother_inds = np.empty(ncells)
            mother_inds[:] = np.nan
            hasm = ~np.isnan(mnum)
            mother_inds[hasm] = cmap[pnum[hasm], tnum[hasm], mnum[hasm].astype(int) - 1]
            if np.any(mother_inds[hasm] == 0):
                warn('Some mothers are missing...')
                mother_inds[hasm & (mother_inds == 0)] = np.nan
            mother_inds -= 1  # convert to Python indexing
            mothers = np.unique(mother_inds[~np.isnan(mother_inds)])

        fvals = ['f', 'df', 'ddf']
        fits = {}
        for f in fvals:
            fits[f] = {}
            fits[f]['mean'] = np.empty(allV.shape)
            fits[f]['mean'][:] = np.nan
            fits[f]['std'] = np.empty(allV.shape)
            fits[f]['std'][:] = np.nan

        # Loop over all mothers and run smoothing
        for m in trange(allV.shape[1]):
            if split_at_birth:
                # Split time series at each birth event
                btps = np.flatnonzero(births[:,m])
                daughter_inds = cmap[pnum[m], tnum[m], dlabs[btps, m].astype(int) - 1]
                if np.any(daughter_inds == 0):
                    warn(f'Some daughters are missing for cell {m + 1}')
                    daughter_inds = daughter_inds[daughter_inds != 0]
                daughter_inds -= 1
                assert set(daughter_inds) == set(np.flatnonzero(mother_inds == m))
                # Where birth is uncertain, the daughter may appear and
                # disappear over its first few time points, with the mother
                # dividing and merging in complement. Take care to add all
                # mother-like tps to time series of undivided cell, and keep
                # only daughter-like tps in time series of divided cell
                flicker_mask = np.zeros(ntps, dtype='bool')
                for btp, nbtp, d in zip(btps, np.hstack((btps[1:], ntps)), daughter_inds):
                    ltp = max(nbtp, np.argmax(births[:,d]))
                    flicker_mask[btp:ltp] = np.isnan(allV[btp:ltp, d])
                mVs = []
                btps = np.hstack((0, btps, ntps))
                for s, b, e in zip(btps[:-2], btps[1:-1], btps[2:]):
                    mV = allV[:, m].copy()
                    mV[:s] = np.nan
                    mV[e:] = np.nan
                    mV[s:b][flicker_mask[s:b]] = np.nan
                    mV[b:e][~flicker_mask[b:e]] = np.nan
                    mVs.append(mV)

                mV = allV[:, m].copy()
                mV[:btps[-2]] = np.nan
                mV[btps[-2]:][flicker_mask[btps[-2]:]] = np.nan
                mVs.append(mV)

            else:
                # Treat the mother as one time series
                mVs = [allV[:, m]]

            for mV in mVs:
                mf = ~np.isnan(mV)
                if mf.sum() < 3:
                    continue
                mT = allT[mf, m]
                mV = mV[mf]
                mg = gp.maternGP(hyperpar_bounds, mT, mV)
                mg.findhyperparameters(noruns=5)
                mg.predict(mT, derivs=2)
                for f in fvals:
                    fits[f]['mean'][mf, m] = getattr(mg, f)
                    fits[f]['std'][mf, m] = np.sqrt(getattr(mg, f + 'var'))

        # Save predictions
        dsprops = dict(shape=allV.shape, dtype=allV.dtype,
                       chunks=True, compression='gzip', fillvalue=np.nan)
        data['M'].require_dataset('svol', **dsprops)
        data['M/svol'][()] = fits['f']['mean']
        data['M'].require_dataset('svol_std', **dsprops)
        data['M/svol_std'][()] = fits['f']['std']
        data['M'].require_dataset('grs', **dsprops)
        data['M/grs'][()] = fits['df']['mean']
        data['M'].require_dataset('grs_std', **dsprops)
        data['M/grs_std'][()] = fits['df']['std']
        data['M'].require_dataset('d2v', **dsprops)
        data['M/d2v'][()] = fits['ddf']['mean']
        data['M'].require_dataset('d2v_std', **dsprops)
        data['M/d2v_std'][()] = fits['ddf']['std']

    if run_dght:
        fvals = ['f', 'df', 'ddf']
        for g in tqdm(data['D']):
            grp = data['D'][g]
            allT = grp['times']
            allV = grp['vol']
            if log_volume:
                allV = np.log(allV)

            fits = {}
            for f in fvals:
                fits[f] = {}
                fits[f]['mean'] = np.empty(allV.shape)
                fits[f]['mean'][:] = np.nan
                fits[f]['std'] = np.empty(allV.shape)
                fits[f]['std'][:] = np.nan

            # Loop over all daughters and run smoothing
            for d in range(allV.shape[1]):
                dV = allV[:, d]
                df = ~np.isnan(dV)
                if not df.any():
                    continue
                dT = allT[df, d]
                dV = dV[df]
                dg = gp.maternGP(hyperpar_bounds, dT, dV)
                dg.findhyperparameters(noruns=5)
                dg.predict(dT, derivs=2)
                for f in fvals:
                    fits[f]['mean'][df, d] = getattr(dg, f)
                    fits[f]['std'][df, d] = np.sqrt(getattr(dg, f + 'var'))

            # Save predictions
            dsprops = dict(shape=allV.shape, dtype=allV.dtype,
                           chunks=True, compression='gzip', fillvalue=np.nan)
            grp.require_dataset('svol', **dsprops)
            grp['svol'][()] = fits['f']['mean']
            grp.require_dataset('svol_std', **dsprops)
            grp['svol_std'][()] = fits['f']['std']
            grp.require_dataset('grs', **dsprops)
            grp['grs'][()] = fits['df']['mean']
            grp.require_dataset('grs_std', **dsprops)
            grp['grs_std'][()] = fits['df']['std']
            grp.require_dataset('d2v', **dsprops)
            grp['d2v'][()] = fits['ddf']['mean']
            grp.require_dataset('d2v_std', **dsprops)
            grp['d2v_std'][()] = fits['ddf']['std']

    data.close()


def main():
    from optparse import OptionParser
    usage = "usage: %prog [options] grdata_file"
    parser = OptionParser(usage=usage)
    parser.add_option('-d','--daughters-only', dest='run_moth',
            help='only fit daughter data', action='store_false',
            default=True)
    parser.add_option('-m','--mothers-only', dest='run_dght',
            help='only fit mother data', action='store_false',
            default=True)
    parser.add_option('-s','--split-at-birth', dest='split_at_birth',
            help='split mother traces according to birth times',
            action='store_true', default=False)
    parser.add_option('-l','--log-volume', dest='log_volume',
            help='fit gaussian process to log(V)',
            action='store_true', default=False)
    (options, args) = parser.parse_args()

    if len(args) < 1:
        parser.print_help()
        sys.exit()

    for grdata_file in args:
        fit_gr_data(grdata_file, run_moth=options.run_moth,
                run_dght=options.run_dght,
                split_at_birth=options.split_at_birth,
                log_volume=options.log_volume)
