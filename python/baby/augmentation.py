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
"""
A set of augmentation functions for on-the-fly preprocessing of images before
input into the Neural Network. All augmentation functions are meant to work
with unstructured arrays (in particular, elastic_deform, turn, and the shifts
probably will return errors if used on structured arrays. They all take exactly
two arrays as input, and will perform identical transformations on both arrays.
"""
from __future__ import absolute_import, division, print_function
import json
import numpy as np
from itertools import permutations, repeat
from scipy.ndimage import map_coordinates, gaussian_filter, shift
from scipy.ndimage.morphology import binary_fill_holes
from scipy import interpolate
from skimage import transform
from skimage.feature import canny
from skimage.filters import gaussian
from skimage.draw import rectangle_perimeter
import elasticdeform
# import cv2

from .preprocessing import segoutline_flattening, connect_pixel_gaps
from .errors import BadParam, BadFile

AUGMENTATION_ORDER = ('substacks', 'vshift', 'hshift', 'rough_crop',
                      'downscale', 'shadows', 'blur', 'elastic_deform',
                      'rotate', 'crop', 'vflip', 'hflip',
                      'movestacks', 'noise', 'pad', 'gamma',
                      'hist_deform', 'hist_scale')


class Augmenter(object):

    def __init__(self, xy_out=80, probs={}, p_noop=0.05, substacks=None,
            only_basic_augs=True):
        """
        Random data augmentation of img and lbl.

        Parameters
        ----------
        xy_out : int or pair of ints as a tuple
            The intended width and height of the final augmented image.

        probs : dict of floats in [0, 1] or None
            Specify non-default probabilities for the named augmentations.
            Augmentations with zero probability are omitted.  To get all
            augmentations with equal probability, simply specify an empty
            dict.

        p_noop : float in [0, 1]
            Adjusts the probability for no augmentation operation in the
            default case. If set to 1, then all operations will be omitted
            by default (i.e., unless a non-zero probability is specified
            for that operation in probs).

        only_basic_augs: bool
            If True, then elastic_deform and img intensity augmentations are
            omitted (i.e., "shadows", "elastic_deform", "gamma",
            "hist_deform", "hist_scale"). This was the default for the BABY
            paper. Specify False to include all augmentations. NB: this has
            lower priority than the `probs` arg, e.g., if `shadows=0.2` is
            specified in probs, then the shadows aug will be active.
        """

        if type(xy_out) is int:
            self.xy_out = (xy_out, xy_out)
        elif len(xy_out) == 2:
            self.xy_out = tuple(xy_out)
        else:
            raise Exception('"xy_out" must be an int or pair of ints')

        self.xy_in = None
        self.pad_value = None
        self.aug_log = []
        self._vshift = None
        self._hshift = None

        self.preserve_bool_lbl = True

        # Interpolation order for label operations
        # This should be 0 for bitmask labels
        self.lbl_order = 0

        if only_basic_augs:
            custom_probs = probs
            probs = dict(shadows=0, elastic_deform=0, gamma=0, hist_deform=0,
                    hist_scale=0, blur=0)
            probs.update(**custom_probs)

        # Treat 'crop', 'pad' and 'substacks' specially to have p = 1
        guaranteed_augs = ('substacks', 'rough_crop', 'pad', 'crop')
        self.aug_order = [
            a for a in AUGMENTATION_ORDER
            if probs.get(a, 1) > 0 or a in guaranteed_augs
        ]
        n_augs = len([1 for a in self.aug_order
                      if probs.get(a, 0) < 1
                      and a not in guaranteed_augs])
        p_default = 1. - p_noop**(1/n_augs)
        self.probs = np.array([
            1 if a in guaranteed_augs else probs.get(a, p_default)
            for a in self.aug_order
        ])

        if substacks is not None:
            if type(substacks) != int:
                raise BadParam('"substacks" must be None or an int')
            if substacks < 1:
                raise BadParam('"substacks" must be a positive integer')
        self.nsubstacks = substacks

    def __call__(self, img, lbl_info):
        """
        Random data augmentation of img and lbl.

        Parameters
        ----------
        img, lbl : np.ndarray, unstructured and in 3 dimensions.
            The input (img) and true output (lbl) to the Neural Network.

        Return
        -------
        (np.ndarray, np.ndarray)
            The augmented images, unstructured and in 3 dimensions

        """

        # Ensure that img and lbl have the same xy dimensions
        if isinstance(lbl_info, tuple):
            lbl, info = lbl_info
        else:
            lbl = lbl_info
        assert img.shape[:2] == lbl.shape[:2], \
            'xy dimensions of img and lbl are mismatched'

        lbl_is_bool = lbl.dtype == 'bool' and self.preserve_bool_lbl

        self.xy_in = img.shape[:2]
        self.pad_value = None
        self.aug_log = []

        for a, p in zip(self.aug_order, self.probs):
            if p == 0.0:
                continue
            elif p == 1.0:
                self.aug_log.append(a)
                img, lbl = getattr(self, a)(img, lbl)
            else:
                if np.random.uniform() < p:
                    self.aug_log.append(a)
                    img, lbl = getattr(self, a)(img, lbl)
            if lbl_is_bool and lbl.dtype != 'bool':
                lbl = lbl > 0.5  # ensure label stays boolean

        self.xy_in = None
        self.pad_value = None

        # Ensure that x and y dimensions match the intended output size
        assert img.shape[:2] == self.xy_out and lbl.shape[:2] == self.xy_out, \
            'xy dimensions do not match intended size after augmentation'

        return (img, lbl)

    def substacks(self, img, lbl):
        """Randomly pick Z-sections to match the chosen substack size

        By default all Z-sections are used. If `substacks` was specified in
        the constructor, then Z-sections are randomly chosen to limit to that
        number of stacks. All subsets are considered except those where
        Z-sections are separated by more than one omitted Z-slice.

        An error will be raised if the image has fewer Z-sections than the
        number specified in `substacks`.

        Images presented to the augmenter call do not necessarily all need to
        have the same number of Z-sections; this will result in consistent
        output if `substacks` is specified, but will potentially result in
        inconsistent output in the default case.
        """

        self.refslice = 0
        nsub = self.nsubstacks

        if nsub:
            # nz = number of z-sections in input image
            nz = 1 if len(img.shape) < 3 else img.shape[2]
            if nz < self.nsubstacks:
                raise BadParam(
                    '"img" has fewer Z-sections than the specified "substacks"')
            
            # Obtain set of permuted filters (logical masks) to subset the
            # current Z-sections to that specified by `nsubstacks`
            template_filter = tuple(repeat(True, nsub)) + tuple(repeat(False, nz - nsub))
            ss_filters = tuple(np.array(f) for f in set(permutations(template_filter)))

            # Exclude filters that select Z-sections separated by more than one
            # omitted Z-section. The logic is:
            # - `f[:-1] != f[1:]` finds transition points between blocks of
            #   True/False
            # - Prepending logic with `f[0]` ensures that odd counts in diff
            #   will be False (pairs with `[::2]`)
            # - Blocks of False at the end are ignored by not appending
            #   anything to the logic
            # - Blocks of False at the start are ignored by subsetting the
            #   results of `where`
            ss_filters = [
                f for f in ss_filters if
                (np.diff(np.where(np.concatenate(([f[0]],f[:-1] != f[1:])))[0][1:])[::2] < 2).all()
            ]

            ss = ss_filters[np.random.randint(len(ss_filters))]
            img = img[:, :, ss]
            self.refslice = int(np.median(np.where(ss))) + 1

        self.pad_value = np.median(img)
        return img, lbl

    def pad(self, img, lbl, update_size=True):
        """Ensure image/label size are large enough

        A guaranteed augmentation that pads image to ensure that 
        as required Guaranteed augmentation simply run to ensure image/label sizes 
        """

        # We store a pad value to ensure consistent padding if this routine is
        # called more than once in the augmentation chain
        if self.pad_value is None:
            self.pad_value = np.median(img)

        # Ensure that img and lbl xy dimensions are at least as large as the
        # requested output
        Xin, Yin = img.shape[:2]
        Xout, Yout = self.xy_out
        Xpad = Xout - Xin if Xin < Xout else 0
        Ypad = Yout - Yin if Yin < Yout else 0
        if Xpad > 0 or Ypad > 0:
            XpadL = Xpad // 2
            Xpad = (XpadL, Xpad - XpadL)
            YpadL = Ypad // 2
            Ypad = (YpadL, Ypad - YpadL)
            img = np.pad(img, (Xpad, Ypad, (0, 0)),
                mode='constant', constant_values=self.pad_value)
            lbl = np.pad(lbl, (Xpad, Ypad, (0, 0)), mode='constant')

        # Update input size by default
        if update_size:
            self.xy_in = img.shape[:2]

        return img, lbl

    def rotate(self, img, lbl):
        """Random rotation

        Example image:

        .. image:: ../report/figures/augmentations/turn.*
        """
        angle = np.random.choice(360)
        return (transform.rotate(img,
                                 angle=angle,
                                 mode='reflect',
                                 resize=True),
                transform.rotate(lbl,
                                 angle=angle,
                                 mode='reflect',
                                 resize=True))

    def vshift(self, img, lbl, maxpix=None):
        """Shift along height, max of 10px by default

        Only integer shifts are made (i.e., no interpolation)

        Example image:

        .. image:: ../report/figures/augmentations/vshift.*
        """

        if maxpix is None:
            if self.xy_in is None:
                inshape = img.shape[0]
            else:
                inshape = self.xy_in[0]
            maxpix = np.max([0, (inshape - self.xy_out[0]) // 2])

        self._vshift = np.random.choice(np.arange(-maxpix, maxpix + 1, dtype='int'))
        return img, lbl

    def hshift(self, img, lbl, maxpix=None):
        """Shift along width, max of 10px by default

        Only integer shifts are made (i.e., no interpolation)

        Example image:

        .. image:: ../report/figures/augmentations/hshift.*
        """

        if maxpix is None:
            if self.xy_in is None:
                inshape = img.shape[1]
            else:
                inshape = self.xy_in[1]
            maxpix = np.max([0, (inshape - self.xy_out[1]) // 2])

        self._hshift = np.random.choice(np.arange(-maxpix, maxpix + 1, dtype='int'))
        return img, lbl

    def rough_crop(self, img, lbl, xysize=None, nonempty_slices=None):
        if xysize is None:
            xysize = np.array(self.xy_out, dtype=float) * 1.7
            xysize = np.ceil(xysize).astype(int)
        if self.pad_value is None:
            self.pad_value = np.median(img)
        vshift = 0 if self._vshift is None else self._vshift
        hshift = 0 if self._hshift is None else self._hshift
        img = _shift_and_crop(img, vshift, hshift, xysize, self.pad_value)
        lbl = _shift_and_crop(lbl, vshift, hshift, xysize, 0,
                              nonempty_slices=nonempty_slices)
        self._vshift, self._hshift = None, None
        return img, lbl

    def crop(self, img, lbl, xysize=None):
        if xysize is None:
            xysize = self.xy_out

        return _apply_crop(img, xysize), _apply_crop(lbl, xysize)

    def downscale(self, img, lbl, maxpix=None):
        if maxpix is None:
            if self.xy_in is None:
                inshape = img.shape[:2]
            else:
                inshape = self.xy_in
            maxpix = np.max([0, np.min(inshape) - np.max(self.xy_out)])

        pix = np.random.choice(maxpix + 1)
        pix = maxpix
        img_crop, lbl_crop = self.crop(img,
                                       lbl,
                                       xysize=np.array(self.xy_out) + pix)

        return (transform.resize(img_crop, self.xy_out),
                transform.resize(
                    lbl_crop, self.xy_out, anti_aliasing_sigma=0.05) > 0.3)

    def vflip(self, img, lbl):
        """Vertical flip

        Example image:

        .. image:: ../report/figures/augmentations/vflip.*
        """
        return np.flip(img, axis=0), np.flip(lbl, axis=0)

    def hflip(self, img, lbl):
        """Horizontal flip

        Example image:

        .. image:: ../report/figures/augmentations/hflip.*
        """
        return np.flip(img, axis=1), np.flip(lbl, axis=1)

    def movestacks(self, img, lbl):
        """Translate stacks in img to increase robustness to shifts in focus

            Only movements up or down by one stack are made, and the boundary is
            repeated
        """

        if np.random.uniform() < 0.5:
            # Move img stacks up by one
            img[:, :, 1:] = img[:, :, :-1]
        else:
            # Move img stacks down by one
            img[:, :, :-1] = img[:, :, 1:]

        return img, lbl

    def noise(self, img, lbl):
        """Add gaussian noise to the img (not the lbl)
        """

        img += np.random.normal(scale=np.random.exponential(0.003),
                                size=img.shape)
        return img, lbl

    def elastic_deform(self, img, lbl):
        """Random deformation based on elasticdeform package from PyPI

        We aim for relatively weak deformation, so assign grid points
        approximately every 32 pixels and keep the sigma low.

        The interpolation order for the lbl is set from the `lbl_order`
        property of this class. It should be 0 if the lbl is a bitmask and no
        smoothing has been applied, but can be the default 3 if smoothing of
        a bitmask image has been applied.
        """

        # Want only light deformation, so approximately one grid point every
        # 32 pixels:
        npoints = np.maximum(np.round(np.array(lbl.shape[:2])/32), 1)
        npoints = npoints.astype('int').tolist()
        return elasticdeform.deform_random_grid(
            [img, lbl], sigma=2, points=npoints,
            order=[3, self.lbl_order], mode='reflect', axis=[(0, 1), (0, 1)])

    def shadows(self, img, lbl):
        """Location-dependent intensity deformation

        Introduce local changes in intensity by fitting a bivariate spline to
        a grid of random intensities. 
        """

        # Want only slow changes in intensity, so approximately one grid point
        # every 64 pixels:
        npoints = np.maximum(np.round(np.array(img.shape[:2])/64), 1)
        npoints = npoints.astype('int').tolist()

        # Set up grid with random intensities and interpolate
        x = np.linspace(0, img.shape[0] - 1, np.maximum(npoints[0], 2))
        y = np.linspace(0, img.shape[1] - 1, np.maximum(npoints[1], 2))
        I = np.random.normal(size=(x.size, y.size))
        mapping = interpolate.RectBivariateSpline(
                x, y, I, kx=np.minimum(x.size, 4) - 1,
                ky=np.minimum(y.size, 4) - 1, s=(x.size * y.size) * 0.25)
        # Scaled so that 90% of distribution sits within 0.5- to 2-fold
        I_scaler = np.exp(0.4 * mapping(np.arange(img.shape[0]),
                np.arange(img.shape[1]), grid=True))
        # Truncate to a two-fold change in intensity
        I_scaler = np.minimum(np.maximum(I_scaler, 0.5), 2)
        # Ensure that the mean intensity of the entire image stays
        # approximately constant
        I_scaler /= np.median(I_scaler)
        # Apply scaling consistently across z-sections
        I_scaler = I_scaler[..., None]
        img_min = img.min()
        img_aug = img_min + (img - img_min) * I_scaler
        return img_aug, lbl

    def gamma(self, img, lbl):
        """Adjust image gamma

        Picks a random gamma transform to apply to the image
        """

        # Pick a random gamma
        g = np.random.uniform(0.3, 2.0)

        # Attempt to robustly maintain the intensity range
        iQ = np.quantile(img, (0, 0.02, 0.5, 0.98, 1))
        imin, iptp = iQ[0], iQ[4] - iQ[0]
        iQn = (iQ - imin) / iptp
        # aim for consistent difference between 2/98 quantiles
        sc = np.diff(iQ[[1, 3]]) / np.diff(iQn[[1, 3]] ** g)
        # aim for consistent median
        off = iQ[2] - iQn[2] ** g

        return off + sc * ((img - imin) / iptp) ** g, lbl

    def hist_deform(self, img, lbl):
        """Random deformation of intensity histogram

        Maps equally-spaced intensity ranges to new ranges sampled from the
        Dirichlet distribution (of equal average lengths), but uses
        interpolation to smooth the mapping.

        Inspired by the `histogram_voodoo` implementation provided by Daniel
        Eaton from the Paulsson lab for the Delta segmentation package.
        """
        npoints = 5  # includes end points
        control_points = np.linspace(0, 1, num=npoints)
        # split into `npoints - 1` segments with roughly equal lengths
        # i.e., want relatively high alpha parameter:
        mapped_points = np.random.dirichlet((2 * npoints,) * (npoints - 1))
        mapped_points = np.cumsum(np.concatenate([[0], mapped_points]))
        mapping = interpolate.PchipInterpolator(control_points, mapped_points)

        # Attempt to robustly maintain the intensity range
        iQ = np.quantile(img, (0, 0.02, 0.5, 0.98, 1))
        imin, iptp = iQ[0], iQ[4] - iQ[0]
        iQn = (iQ - imin) / iptp
        # aim for consistent difference between 2/98 quantiles
        sc = np.diff(iQ[[1, 3]]) / np.diff(mapping(iQn[[1, 3]]))
        # aim for consistent median
        off = iQ[2] - mapping(iQn[2])

        return off + sc * mapping((img - imin) / iptp), lbl

    def hist_scale(self, img, lbl):
        """Randomly offset and scale the intensity histogram
        """

        # Sample an offset within 25% of the range of image intensities
        imgrange = np.diff(np.quantile(img, [0.02, 0.98]))
        off = np.random.uniform(-0.25*imgrange, 0.25*imgrange)

        # Fairly sample a scaling factor between roughly 0.5 and 2
        sc = np.random.lognormal(sigma=0.4)
        # Truncate to a minimum of 0.5 and maximum of 2
        sc = np.minimum(np.maximum(sc, 0.5), 2)

        return (img + off) * sc, lbl

    def blur(self, img, lbl):
        """Apply gaussian filter to img to simulate loss of focus

        Draw the sigma from an uniform distribution between 1 and 4.
        """
        return gaussian(img, np.random.uniform(1, 4)), lbl


class SmoothingSigmaModel(object):
    """Model for picking a smoothing sigma for gaussian filtering
    
    There are two model types, 'exponential' and 'constant'

    For the exponential model, `a`, `b` and `c` should be obtained by fitting
    data to the following model:
        nedge = c + a * exp(b * sigma)

    For the constant model, set `a` such that:
        sigma = a
    i.e., sigma is independent of nedge
    """

    def __init__(self, a=None, b=None, c=None, model='exponential'):
        self._a = a
        self._b = b
        self._c = c
        if model == 'exponential':
            self._formula = 'sigma = log((nedge - c) / a) / b'
        elif model == 'constant':
            self._formula = 'sigma = a'
        else:
            raise BadParam('Unrecognised model type')

    def save(self, filename):
        with open(filename, 'wt') as f:
            json.dump(
                {
                    'a': self._a,
                    'b': self._b,
                    'c': self._c,
                    'formula': self._formula
                }, f)

    def load(self, filename):
        with open(filename, 'rt') as f:
            model = json.load(f)
        formula = model.get('formula')
        if formula == 'sigma = m*log(nedge) + c':
            m, c = (model.get(k, 0) for k in ('m', 'c'))
            self._a = np.exp(-c * m)
            self._b = 1 / m
            self._c = 0
            self._formula = 'sigma = log((nedge - c) / a) / b'
        elif formula == 'sigma = a':
            self._a = model.get('a', 0)
            self._b, self._c = None, None
            self._formula = formula
        elif formula == 'sigma = log((nedge - c) / a) / b':
            self._a, self._b, self._c = (
                model.get(k, 0) for k in ('a', 'b', 'c'))
            self._formula = formula
        else:
            raise BadFile('Model with unrecognised formula encountered')

    def __repr__(self):
        keys = ('a', 'b', 'c')
        coefs = (getattr(self, '_' + k) for k in keys)
        coefs = [
            f'{k} = {v:.2f}' for k, v in zip(keys, coefs) if v is not None]
        return 'SmoothingSigmaModel: {}; {}'.format(
            self._formula, ', '.join(coefs))

    def __call__(self, s, scaling=1.):
        if self._formula == 'sigma = a':
            return self._a
        elif self._formula == 'sigma = log((nedge - c) / a) / b':
            nedge = np.sum(s) * scaling
            return np.log(np.clip(
                (nedge - self._c) / self._a, 1, None)) / self._b
        else:
            raise BadParam('Unrecognised formula encountered')


class SmoothedLabelAugmenter(Augmenter):

    def __init__(self,
                 sigmafunc,
                 targetgenfunc=segoutline_flattening,
                 canny_padding=2,
                 **kwargs):
        super(SmoothedLabelAugmenter, self).__init__(**kwargs)
        self.sigmafunc = sigmafunc
        self.targetgenfunc = targetgenfunc
        self.canny_padding = canny_padding
        # Since label smoothing is applied, interpolation order for label
        # transformations can be increased
        self.lbl_order = 3

    def __call__(self, img, lbl_info):
        """This augmenter needs to be used in combination with a label
        preprocessing function that returns both images and info.
        """

        lbl, self.current_info = lbl_info

        # Smooth filled label to avoid anti-aliasing artefacts
        lbl = lbl.astype('float')
        for l in range(lbl.shape[2]):
            o = lbl[..., l]  # slice a single outline
            if o.sum() > 0 and self.sigmafunc(o) > 0:
                lbl[..., l] = gaussian(binary_fill_holes(o),
                                       self.sigmafunc(o))

        return super(SmoothedLabelAugmenter, self).__call__(img, lbl)

    def rotate(self, img, lbl):
        """Random rotation

        Example image:

        .. image:: ../report/figures/augmentations/turn.*
        """
        angle = np.random.choice(360)
        return (transform.rotate(img,
                                 angle=angle,
                                 order=3,
                                 mode='reflect',
                                 resize=True),
                transform.rotate(lbl,
                                 angle=angle,
                                 order=3,
                                 mode='reflect',
                                 resize=True))

    def crop(self, img, lbl, xysize=None):
        # Overload the crop function to restore filled cell masks and generate
        # the flattened targets. Performing these operations before applying
        # cropping helps to avoid boundary effects and misclassification of
        # size group.
        #
        if xysize is None:
            xysize = np.array(self.xy_out)

        # Find edges from blurred images and fill
        for s in range(lbl.shape[2]):
            lbl[:, :, s] = _filled_canny(lbl[:, :, s], self.canny_padding)

        info = self.current_info
        if 'focusStack' in info:
            # Modify info dict that gets passed to flattener in order to
            # specify focus relative to reference slice formed from substack
            info = info.copy()
            cellFocus = info['focusStack']
            if type(cellFocus) != list:
                cellFocus = [cellFocus]
            # print('old focus = {}'.format(', '.join([str(f) for f in cellFocus])))
            info['focusStack'] = [f - self.refslice for f in cellFocus]
            # print('new focus = {}'.format(', '.join([str(f) for f in info['focusStack']])))

        # Generate flattened targets from segmentation outlines
        lbl = self.targetgenfunc(lbl, info)

        return _apply_crop(img, xysize), _apply_crop(lbl, xysize)

    def downscale(self, img, lbl, maxpix=None):
        inshape = img.shape[:2]
        if maxpix is None:
            if self.xy_in is not None:
                inshape = self.xy_in
            maxpix = np.max([0, np.min(inshape) - np.max(self.xy_out)])

        # Allow for small chance of no scaling, but robust if image sizes equal
        pix = np.random.choice(maxpix + 1)
        scaling = (np.min(inshape) - pix) / np.min(inshape)
        outshape = np.floor(np.array(img.shape[:2]) * scaling)

        return (transform.resize(img, outshape),
                transform.resize(lbl, outshape, anti_aliasing=False))


class DownscalingAugmenter(SmoothedLabelAugmenter):

    def __init__(self, *args, pixdev=4, xy_scaled=None, **kwargs):
        super(DownscalingAugmenter, self).__init__(*args, **kwargs)
        self.probs[self.aug_order.index('downscale')] = 1
        self.pixdev = pixdev
        self.xy_scaled = xy_scaled or self.xy_out

    def downscale(self, img, lbl, maxpix=None):
        inshape = img.shape[:2]
        if maxpix is None:
            if self.xy_in is not None:
                inshape = self.xy_in
            maxpix = np.max([0, np.min(inshape) - np.max(self.xy_scaled)])

        pix = np.arange(np.max([0, maxpix - self.pixdev]), maxpix + 1)
        if len(pix) == 0 or len(pix) == 1:
            pix = maxpix
        else:
            pix = np.random.choice(pix)
        scaling = (np.min(inshape) - pix) / np.min(inshape)
        outshape = np.floor(np.array(img.shape[:2]) * scaling)

        return (transform.resize(img, outshape),
                transform.resize(lbl, outshape, anti_aliasing=False))


class ScalingAugmenter(SmoothedLabelAugmenter):
    """Augmenter with a target pixel size

    Assumes that images have `pixel_size` property in their meta info. If it
    is missing, images are assumed to be at the target size.

    NB: The `downscale` augmentation can now also upscale the image.
    """

    def __init__(self,
                 *args,
                 target_pixel_size=0.263,
                 scale_frac=0.05,
                 **kwargs):
        super(ScalingAugmenter, self).__init__(*args, **kwargs)
        self.target_pixel_size = target_pixel_size
        self.scale_frac = scale_frac
        scale_index = self.aug_order.index('downscale')
        self.scale_prob = self.probs[scale_index]
        self.probs[scale_index] = 1
        self.preserve_bool_lbl = False

    def __call__(self, img, lbl_info):
        lbl, info = lbl_info
        self.current_info = info
        self._input_pix_size = info.get('pixel_size', self.target_pixel_size)
        self._scaling = self._input_pix_size / self.target_pixel_size
        self._outshape = np.round(np.array(self.xy_out) / self._scaling)

        return super(SmoothedLabelAugmenter, self).__call__(img, lbl)

    def vshift(self, img, lbl, maxpix=None):
        if maxpix is None:
            if self.xy_in is None:
                inshape = img.shape[0]
            else:
                inshape = self.xy_in[0]
            maxpix = np.max([0, (inshape - self._outshape[0]) // 2])
        return super(ScalingAugmenter, self).vshift(img, lbl, maxpix=maxpix)

    def hshift(self, img, lbl, maxpix=None):
        if maxpix is None:
            if self.xy_in is None:
                inshape = img.shape[1]
            else:
                inshape = self.xy_in[1]
            maxpix = np.max([0, (inshape - self._outshape[1]) // 2])
        return super(ScalingAugmenter, self).hshift(img, lbl, maxpix=maxpix)

    def rough_crop(self, img, lbl):
        xysize = np.array(self.xy_out, dtype=float) * 1.7 / self._scaling
        xysize = np.ceil(xysize).astype(int)

        nonempty = np.ones(lbl.shape[2], dtype=bool)
        n_total = nonempty.size if lbl.any() else 0
        img, lbl = super(ScalingAugmenter, self).rough_crop(
            img, lbl, xysize=xysize, nonempty_slices=nonempty)

        # Remove empty sections of lbl and sync with info
        if lbl.shape[2] == 0:
            lbl = np.zeros(lbl.shape[:2] + (1,), dtype=bool)
        info = self.current_info.copy()
        for k in ['cellLabels', 'focusStack', 'buds']:
            if k in info and info[k] is not None:
                l = info[k]
                if type(l) != list:
                    l = [l]
                if len(l) != n_total:
                    raise Exception(f'"{k}" info does not match label image shape ({str(info)})')
                info[k] = [x for x, m in zip(l, nonempty) if m]
        self.current_info = info

        # Smooth filled label to avoid anti-aliasing artefacts
        lbl = lbl.astype('float')
        for l in range(lbl.shape[2]):
            o = lbl[..., l]  # slice a single outline
            sigma = max(self.sigmafunc(o), 1)
            if o.sum() > 0:
                lbl[..., l] = gaussian(_bordered_fill_holes(o), sigma)

        return img, lbl

    def downscale(self, img, lbl, maxpix=None):
        # Scale image and label to target pixel size
        inshape = img.shape[:2]
        scaling = self._scaling

        # Apply random scaling according to probability for this op
        # also need to correct the aug_log
        last_aug = self.aug_log.pop()
        assert last_aug == 'downscale'
        p = self.scale_prob
        if p == 1.0 or (p > 0 and np.random.uniform() < p):
            self.aug_log.append('downscale')
            scaling += scaling * self.scale_frac * np.random.uniform(-1, 1)

        outshape = np.round(np.array(img.shape[:2]) * scaling).astype(int)
        # OpenCV is much faster but much less accurate...
        # img = cv2.resize(img, outshape, interpolation=cv2.INTER_LINEAR)
        # if img.ndim == 2:
        #     img = img[:, :, None]
        # lbl = cv2.resize(lbl, outshape, interpolation=cv2.INTER_LINEAR)
        # if lbl.ndim == 2:
        #     lbl = lbl[:, :, None]
        img = transform.resize(img, outshape, mode='edge', anti_aliasing=False)
        lbl = transform.resize(lbl, outshape, order=3, mode='edge', anti_aliasing=False)
        return img, lbl


# =============== UTILITY FUNCTIONS ====================== #


def _shift_and_crop(stack, vshift, hshift, xysize, pad_value,
                    nonempty_slices=None):
    r_in, c_in = stack.shape[:2]
    r_out, c_out = xysize
    rb_in = (r_in - r_out) // 2 - vshift
    cb_in = (c_in - c_out) // 2 - hshift
    re_in = rb_in + r_out
    ce_in = cb_in + c_out
    rb_out = max(0, -rb_in)
    re_out = min(r_out, r_out - re_in + r_in)
    cb_out = max(0, -cb_in)
    ce_out = min(c_out, c_out - ce_in + c_in)
    rb_in = max(0, rb_in)
    re_in = min(r_in, re_in)
    cb_in = max(0, cb_in)
    ce_in = min(c_in, ce_in)
    slicedims = stack.shape[2:]
    if nonempty_slices is not None:
        nonempty_slices[()] = stack[rb_in:re_in, cb_in:ce_in].any(axis=(0,1))
        slicedims = (int(nonempty_slices.sum()),) + slicedims[1:]
    out = np.full_like(stack, pad_value, shape=tuple(xysize) + slicedims)
    if nonempty_slices is None:
        out[rb_out:re_out, cb_out:ce_out] = stack[rb_in:re_in, cb_in:ce_in]
    else:
        out[rb_out:re_out, cb_out:ce_out, :] = stack[
            rb_in:re_in, cb_in:ce_in, nonempty_slices]
    return out

def _apply_crop(stack, xysize):
    cropy, cropx = xysize
    starty, startx = stack.shape[:2]
    if startx > cropx:
        startx = (startx - cropx) // 2
        stack = stack[:, startx:(startx + cropx), ...]
    if starty > cropy:
        starty = (starty - cropy) // 2
        stack = stack[starty:(starty + cropy), :, ...]
    return stack


def _bordered_fill_holes(edgemask, pad_width=1, prepadded=False):
    """Fill holes in a mask treating the boundary as an edge

    NB: this function assumes that no cells intersect with opposing borders.
    """

    bp = pad_width
    if not prepadded:
        edgemask = np.pad(edgemask, bp)

    mask = np.zeros(edgemask.shape, dtype='bool')
    mask[bp:-bp, bp:-bp] = edgemask[bp:-bp, bp:-bp]

    # The following assumes that the cell does not intersect with opposing
    # borders, filling bordering with two U-shaped border edges:
    mask[:bp, :] = True
    mask[-bp:, :] = True
    mask[:, :bp] = True
    mask = binary_fill_holes(mask)

    mask[:, :bp] = False
    mask[:, -bp:] = True
    mask = binary_fill_holes(mask)

    return mask[bp:-bp, bp:-bp]


def _filled_canny(segblur, pad_width=2):
    """Use canny to find edge and fill object

    Handles intersections with border by assuming that the object cannot
    intersect all borders at once.

    segblur:  segmentation image that has been gaussian blurred
    pad_width:       border padding
    """
    se = canny(np.pad(segblur, pad_width, 'edge'), sigma=0)
    return _bordered_fill_holes(se, pad_width=pad_width, prepadded=True)
