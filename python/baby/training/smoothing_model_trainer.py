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
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from baby.augmentation import _filled_canny, _apply_crop, SmoothingSigmaModel
from baby.errors import BadProcess, BadType
from baby.preprocessing import dwsquareconn
from baby.segmentation import binary_edge, squareconn, morph_radial_thresh_fit, \
    draw_radial, mask_iou
from baby.utils import find_file
from numpy.polynomial import Polynomial
from scipy.ndimage import binary_fill_holes
from scipy.optimize import curve_fit
from skimage import filters, transform
from skimage.measure import regionprops
from tqdm import trange

from .utils import augmented_generator, TrainValProperty


def _generate_smoothing_sigma_stats(gen):
    sigmas = np.arange(0.4, 5.0, 0.20)
    rotations = np.arange(7, 45, 7)
    scaling = np.linspace(0.5, 1.5, 6)

    square_edge = lambda m: binary_edge(m, squareconn)
    smoothing_stats = []
    for t in trange(len(gen.paths)):
        _, (segs, _) = gen.get_by_index(t)

        if segs.shape[2] == 1 and segs.sum() == 0:
            continue

        ncell = segs.shape[2]
        segs_fill = binary_fill_holes(segs, dwsquareconn)
        segs_edge = binary_edge(segs, dwsquareconn)
        for c in range(ncell):
            sfill = segs_fill[..., c]
            sedge = segs_edge[..., c]
            nedge = segs[..., c].sum()

            # fit radial spline to generate accurate reference edges for
            # resize transformation
            rprops = regionprops(sfill.astype('int'))[0]
            centre = np.array(rprops.centroid)
            radii, angles = morph_radial_thresh_fit(sedge, sfill, rprops)
            genedge = draw_radial(radii, angles, centre, sedge.shape)
            genfill = binary_fill_holes(genedge, squareconn)

            # Limit the number of rotation and scaling operations by
            # randomly choosing one per cell:
            r = np.random.choice(rotations)
            z = np.random.choice(scaling)
            for s in sigmas:
                # Use gaussian blurred filled image for augmentations
                sblur = filters.gaussian(sfill, s)
                genblur = filters.gaussian(genfill, s)

                spf = _filled_canny(sblur)
                smoothing_stats += [{
                    'ind': t,
                    'cell': c,
                    'sigma': s,
                    'rotation': 0,
                    'scaling': 1,
                    'nedge': nedge,
                    'iou': mask_iou(spf, sfill),
                    'edge_iou': mask_iou(square_edge(spf), sedge)
                }]

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
                    'iou': mask_iou(srf, sfill),
                    'edge_iou': mask_iou(square_edge(srf), sedge)
                }]

                insize = np.array(spf.shape)
                outsize = np.round(insize * z).astype('int')
                centre_sc = outsize / 2 + z * (centre - insize / 2)
                genedge_sc = draw_radial(z * radii, angles, centre_sc,
                                         outsize)
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
                    'iou': mask_iou(sdf, genfill_sc),
                    'edge_iou': mask_iou(square_edge(sdf), genedge_sc)
                }]

    return pd.DataFrame(smoothing_stats)

class SmoothingModelTrainer:
    def __init__(self, save_dir, stats_file, model_file):
        self.save_dir = save_dir
        self.stats_file = save_dir / stats_file
        self.model_file = save_dir / model_file
        self._model = None
        self._stats = None

    def generate_smoothing_sigma_stats(self, train_gen, val_gen):
        with augmented_generator(train_gen, lambda x, y: (x, y)) as gen:
            sss_train = _generate_smoothing_sigma_stats(gen)
        with augmented_generator(val_gen, lambda x, y: (x, y)) as gen:
            sss_val = _generate_smoothing_sigma_stats(gen)
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
