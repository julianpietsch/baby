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
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from baby.augmentation import _filled_canny, _apply_crop, SmoothingSigmaModel
from baby.errors import BadProcess, BadType
from baby.preprocessing import dwsquareconn
from baby.generator import augmented_generator
from baby.segmentation import (binary_edge, squareconn, mask_to_knots,
                               draw_radial, mask_iou)
from baby.morph_thresh_seg import SegmentationParameters
from baby.utils import find_file
from numpy.polynomial import Polynomial
from scipy.ndimage import binary_fill_holes
from scipy.optimize import curve_fit
from skimage import filters, transform
from skimage.measure import regionprops
from tqdm import trange

from .utils import TrainValProperty


def _generate_smoothing_sigma_stats(gen, params):
    sigmas = np.arange(0.4, 5.0, 0.20)
    rotations = np.arange(7, 45, 7)
    scaling = np.linspace(0.5, 1.5, 6)

    square_edge = lambda m: binary_edge(m, squareconn)
    smoothing_stats = []
    for t in trange(len(gen.paths)):
        segs = gen.get_by_index(t)

        if segs.shape[2] == 1 and segs.sum() == 0:
            continue

        ncell = segs.shape[2]
        segs_fill = binary_fill_holes(segs, dwsquareconn)
        segs_edge = binary_edge(segs, dwsquareconn)
        for c in range(ncell):
            sfill = segs_fill[..., c]
            sedge = segs_edge[..., c]
            nedge = segs[..., c].sum()
            area = sfill.sum()

            # Add padding to make input image square
            nx, ny = sfill.shape[:2]
            if nx != ny:
                rprops = regionprops(sfill.astype('int'))[0]
                min_rr, min_cc, max_rr, max_cc = rprops.bbox
                pad = np.round(0.25 * max(max_rr - min_rr, max_cc - min_cc)).astype(int)
                min_rr = max(min_rr - pad, 0)
                min_cc = max(min_cc - pad, 0)
                max_rr = min(max_rr + pad, nx)
                max_cc = min(max_cc + pad, ny)
                sfill = sfill[min_rr:max_rr, min_cc:max_cc]
                sedge = sedge[min_rr:max_rr, min_cc:max_cc]
                nx, ny = sfill.shape[:2]

            # Add padding to ensure features do not rotate out of image limits
            xpad = max(nx, ny) - nx
            ypad = max(nx, ny) - ny
            xlpad = xpad // 2
            ylpad = ypad // 2
            sfill = np.pad(sfill, ((xlpad, xpad - xlpad),
                (ylpad, ypad - ylpad)), mode='constant')
            sedge = np.pad(sedge, ((xlpad, xpad - xlpad),
                (ylpad, ypad - ylpad)), mode='constant')

            rprops = regionprops(sfill.astype('int'))[0]
            centre = np.array(rprops.centroid)

            # fit spline to generate accurate reference edges for resize
            # transformation. We do not have a predicted edge in this case, so
            # supply the ground truth edge for the fitting routine.
            (_, radii, angles), genedge = mask_to_knots(
                sfill, p_edge=sedge, **params._asdict())
            genfill = binary_fill_holes(genedge, squareconn)

            # Limit the number of rotation and scaling operations by
            # randomly choosing one per cell:
            r = np.random.choice(rotations)
            z = np.random.choice(scaling)
            for s in sigmas:
                # Use gaussian blurred filled image for augmentations
                sblur = filters.gaussian(sfill, s)
                genblur = filters.gaussian(genfill, s)

                # Identity transformation (on raw edge)
                spf = _filled_canny(sblur)
                smoothing_stats += [{
                    'ind': t,
                    'cell': c,
                    'sigma': s,
                    'rotation': 0,
                    'scaling': 1,
                    'nedge': nedge,
                    'area': area,
                    'iou': mask_iou(spf, sfill),
                    'edge_iou': mask_iou(square_edge(spf), sedge)
                    }]

                # Rotation transformation (on raw edge)
                sr = transform.rotate(sblur,
                        angle=r,
                        mode='reflect',
                        resize=True)
                sr = transform.rotate(sr,
                        angle=-r,
                        mode='reflect',
                        resize=False)
                srf = _filled_canny(_apply_crop(sr, spf.shape))
                smoothing_stats += [{
                    'ind': t,
                    'cell': c,
                    'sigma': s,
                    'rotation': r,
                    'scaling': 1,
                    'nedge': nedge,
                    'area': area,
                    'iou': mask_iou(srf, sfill),
                    'edge_iou': mask_iou(square_edge(srf), sedge)
                    }]

                # Scaling transformation (on generated edge)
                insize = np.array(spf.shape)
                outsize = np.round(insize * z).astype('int')
                centre_sc = outsize / 2 + z * (centre - insize / 2)
                genedge_sc = draw_radial(z * radii, angles, centre_sc,
                        outsize, cartesian_spline=params.cartesian_spline)
                genfill_sc = binary_fill_holes(genedge_sc, squareconn)
                sd = transform.resize(genblur, outsize, anti_aliasing=False)
                sdf = _filled_canny(sd)
                smoothing_stats += [{
                    'ind': t,
                    'cell': c,
                    'sigma': s,
                    'rotation': 0,
                    'scaling': z,
                    'nedge': nedge,
                    'area': area,
                    'iou': mask_iou(sdf, genfill_sc),
                    'edge_iou': mask_iou(square_edge(sdf), genedge_sc)
                    }]

    return pd.DataFrame(smoothing_stats)

class SmoothingModelTrainer:
    """Trains the smoothing model for augmentations on binary masks.

    Args:
        shared_params (utils.SharedParameterContainer): training and
            segmentation parameters as provided by
            :py:class:`utils.SharedParameterContainer`.
        shared_data (utils.SharedDataContainer): Access to training data.
    """
    def __init__(self, shared_params, shared_data):
        self._shared_params = shared_params
        self._shared_data = shared_data
        self._model = None
        self._stats = None

    @property
    def save_dir(self):
        return self._shared_params.save_dir

    @property
    def stats_file(self):
        return (self.save_dir /
                self._shared_params.parameters.smoothing_sigma_stats_file)

    @property
    def model_file(self):
        return (self.save_dir /
                self._shared_params.parameters.smoothing_sigma_model_file)

    @property
    def segment_params(self):
        return self._shared_params.segmentation_parameters

    def generate_smoothing_sigma_stats(self, aug=lambda x, y: y[0]):
        train_gen, val_gen, _ = self._shared_data.gen
        with augmented_generator(train_gen, aug) as gen:
            sss_train = _generate_smoothing_sigma_stats(gen, self.segment_params)
        with augmented_generator(val_gen, aug) as gen:
            sss_val = _generate_smoothing_sigma_stats(gen, self.segment_params)
        sss_train['validation'] = False
        sss_val['validation'] = True
        sss = pd.concat((sss_train, sss_val))
        sss.to_csv(self.stats_file)

    @property
    def stats(self):
        if self._stats is None:
            if not self.stats_file.exists():
                raise BadProcess(
                        'smoothing sigma stats have not been generated')
            self._stats = pd.read_csv(self.stats_file)
        return TrainValProperty(self._stats[~self._stats['validation']],
                self._stats[self._stats['validation']])

    @property
    def model(self):
        if self._model is None:
            if self.model_file.is_file():
                smoothing_sigma_model = SmoothingSigmaModel()
                smoothing_sigma_model.load(self.model_file)
                self._model = smoothing_sigma_model
            else:
                raise BadProcess(
                        'The "smoothing_sigma_model" has not been assigned yet')
        return self._model

    @model.setter
    def model(self, ssm):
        if isinstance(ssm, str):
            ssm_file = find_file(ssm, self.save_dir, 'smoothing_sigma_model')
            ssm = SmoothingSigmaModel()
            ssm.load(ssm_file)
        if not isinstance(ssm, SmoothingSigmaModel):
            raise BadType(
                    '"smoothing_sigma_model" must be of type "baby.augmentation.SmoothingSigmaModel"'
                    )
        ssm.save(self.model_file)
        self._model = ssm

    def _get_grouped_sss(self):
        group_best_iou = lambda x: x.loc[x['iou'].idxmax(), :]
        idcols = ['ind', 'cell', 'scaling', 'rotation']
        stats = self.stats.train
        stats = stats.groupby(idcols).apply(group_best_iou)
        filts = {
                'identity': (stats.scaling == 1) & (stats.rotation == 0),
                'scaling': stats.scaling != 1,
                'rotation': stats.rotation != 0
                }
        return stats, filts

    def fit(self, filt='identity'):
        stats, filts = self._get_grouped_sss()

        if filt:
            stats = stats[filts[filt]]

        # Get initial parameters from linear fit of log transformed nedge
        b = 10  # initial guess for offset term in final model
        # Fit s = c + m * log(n - b); want n = b + exp((s - c)/m)
        pinv = Polynomial.fit(np.log(np.clip(stats.nedge - b, 1, None)),
                stats.sigma,
                deg=1)
        c = pinv(0)
        m = pinv(1) - c

        # Fit n = b + a * exp(p * s); inverse: s = log(n - c) / p - log(a) / p
        model = lambda s, a, p, b: b + a * np.exp(p * s)
        p0 = (np.exp(-c / m), 1 / m, b)
        params, _ = curve_fit(model, stats.sigma, stats.nedge, p0=p0)

        self.model = SmoothingSigmaModel(*params)

    def plot_fitted_model(self):
        stats, filts = self._get_grouped_sss()
        model = lambda s, a, p, b: b + a * np.exp(p * s)
        params = (self.model._a, self._model._b, self.model._c)

        fig, axs = plt.subplots(2,
                len(filts),
                figsize=(12, 12 * 2 / len(filts)))
        sigma_max = stats.sigma.max()
        nedge_max = stats.nedge.max()
        sigma = np.linspace(0, sigma_max, 100)
        for ax, (k, f) in zip(axs[0], filts.items()):
            ax.scatter(stats[f].sigma,
                    stats[f].nedge,
                    16,
                    alpha=0.05,
                    edgecolors='none')
            ax.plot(sigma, model(sigma, *params), 'r')
            ax.set(title=k.title(),
                    xlabel='sigma',
                    ylabel='nedge',
                    ylim=[0, nedge_max])

        nedge = np.linspace(1, nedge_max, 100)
        for ax, (k, f) in zip(axs[1], filts.items()):
            ax.scatter(stats[f].nedge,
                    stats[f].sigma,
                    16,
                    alpha=0.05,
                    edgecolors='none')
            ax.plot(nedge, [self.model(n) for n in nedge], 'r')
            ax.set(title=k.title(),
                    xlabel='nedge',
                    ylabel='sigma',
                    ylim=[0, sigma_max])
        fig.tight_layout()
        fig.savefig(self.save_dir / 'fitted_smoothing_sigma_model.png')
