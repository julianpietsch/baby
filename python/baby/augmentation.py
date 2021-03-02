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
"""
A set of augmentation functions for on-the-fly preprocessing of images before
input into the Neural Network. All augmentation functions are meant to work
with unstructured arrays (in particular, elastic_deform, turn, and the shifts
probably will return errors if used on structured arrays. They all take exactly
two arrays as input, and will perform identitcal transformations on both arrays.
"""
from __future__ import absolute_import, division, print_function
import json
import numpy as np
from itertools import permutations, repeat
from scipy.ndimage import map_coordinates, gaussian_filter, shift
from scipy.ndimage.morphology import binary_fill_holes
from skimage import transform
from skimage.feature import canny
from skimage.filters import gaussian
from skimage.draw import rectangle_perimeter

from .preprocessing import segoutline_flattening
from .errors import BadParam

AUGMENTATION_ORDER = ('substacks', 'rotate', 'vshift', 'hshift', 'downscale',
                      'crop', 'vflip', 'hflip', 'movestacks', 'noise')


class Augmenter(object):

    def __init__(self, xy_out=80, probs={}, p_noop=0.05, substacks=None):
        """
        Random data augmentation of img and lbl.

        Parameters
        ----------
        xy_out : int or pair of ints as a tuple
            The intended width and height of the final augmented image.

        probs : dict of floats in [0, 1]
            Specify non-default probabilities for the named augmentations.
            Augmentations with zero probability are omitted.

        p_noop : float in [0, 1]
            Adjusts the probability for no augmentation operation in the
            default case. If set to 1, then all operations will be omitted
            by default (i.e., unless a non-zero probability is specified
            for that operation in probs).
        """

        if type(xy_out) is int:
            self.xy_out = (xy_out, xy_out)
        elif len(xy_out) == 2:
            self.xy_out = xy_out
        else:
            raise Exception('"xy_out" must be an int or pair of ints')

        self.xy_in = None

        self.aug_order = [
            a for a in AUGMENTATION_ORDER
            if probs.get(a, 1) > 0 or a == 'crop'
        ]

        # Treat 'crop' and 'substacks' specially to have p = 1
        guaranteed_augs = ('crop', 'substacks')
        n_augs = len(self.aug_order) - len(guaranteed_augs)
        p_default = (1. - p_noop)**n_augs / n_augs
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

        lbl_is_bool = lbl.dtype == 'bool'

        self.xy_in = img.shape[:2]

        for a, p in zip(self.aug_order, self.probs):
            if p == 0.0:
                continue
            elif p == 1.0:
                img, lbl = getattr(self, a)(img, lbl)
            else:
                if np.random.uniform() < p:
                    img, lbl = getattr(self, a)(img, lbl)
            if lbl_is_bool and lbl.dtype != 'bool':
                lbl = lbl > 0.5  # ensure label stays boolean

        self.xy_in = None

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

        pix = np.random.choice(np.arange(-maxpix, maxpix + 1, dtype='int'))
        return (shift(img, [pix, 0, 0], mode='reflect', order=0),
                shift(lbl, [pix, 0, 0], mode='reflect', order=0))

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

        pix = np.random.choice(np.arange(-maxpix, maxpix + 1, dtype='int'))
        return (shift(img, [0, pix, 0], mode='reflect', order=0),
                shift(lbl, [0, pix, 0], mode='reflect', order=0))

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

    def elastic_deform(self, img, lbl, params={}):
        """Slight deformation

        Elastic deformation of images as described in
        Simard, Steinkraus and Platt, "Best Practices for
        Convolutional Neural Networks applied to Visual Document Analysis", in
        Proc. of the International Conference on Document Analysis and
        Recognition, 2003.
        Adapted from:
        https://gist.github.com/chsasank/4d8f68caf01f041a6453e67fb30f8f5a

        Example image:

        .. image:: ../report/figures/augmentations/elastic_deform.*
        """
        alpha = 0.75 * img.shape[1]
        sigma = 0.08 * img.shape[1]
        x_y = _elastic_deform(np.dstack([img, lbl]), alpha=alpha, sigma=sigma)
        return np.split(x_y, [img.shape[2]], axis=2)

    def identity(self, img, lbl):
        """Do nothing

        Example image:

        .. image:: ../report/figures/augmentations/identity.*
        """
        return img, lbl


class SmoothingSigmaModel(object):
    """Model for picking a smoothing sigma for gaussian filtering
    
    a, b and c should be obtained by fitting data to the following model:
        nedge = c + a * exp(b * sigma)
    """

    def __init__(self, a=None, b=None, c=None):
        self._a = a
        self._b = b
        self._c = c
        self._formula = 'sigma = log((nedge - c) / a) / b'

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
        if model.get('formula') == 'sigma = m*log(nedge) + c':
            m, c = (model.get(k, 0) for k in ('m', 'c'))
            self._a = np.exp(-c * m)
            self._b = 1 / m
            self._c = 0
        elif self._formula != model.get('formula'):
            raise BadFile('Model formula does not match SmoothingSigmaModel')
        else:
            self._a, self._b, self._c = (
                model.get(k, 0) for k in ('a', 'b', 'c'))

    def __repr__(self):
        return 'SmoothingSigmaModel: {}; a = {:.2f}, b = {:.2f}, c = {:.2f}'.format(
            self._formula, self._a, self._b, self._c)

    def __call__(self, s):
        return np.log(np.clip(
            (np.sum(s) - self._c) / self._a, 1, None)) / self._b


class SmoothedLabelAugmenter(Augmenter):

    def __init__(self,
                 sigmafunc,
                 targetgenfunc=segoutline_flattening,
                 **kwargs):
        super(SmoothedLabelAugmenter, self).__init__(**kwargs)
        self.sigmafunc = sigmafunc
        self.targetgenfunc = targetgenfunc

    def __call__(self, img, lbl_info):
        """This augmenter needs to be used in combination with a label
        preprocessing function that returns both images and info.
        """

        lbl, info = lbl_info

        # Smooth filled label for to avoid anti-aliasing artefacts
        lbl = lbl.astype('float')
        for l in range(lbl.shape[2]):
            o = lbl[..., l]  # slice a single outline
            if o.sum() > 0 and self.sigmafunc(o) > 0:
                lbl[..., l] = gaussian(binary_fill_holes(o),
                                       self.sigmafunc(o))

        img, lbl = super(SmoothedLabelAugmenter, self).__call__(img, lbl)

        # NB: to limit open shapes, the crop operation has been overloaded to
        # find edges before cropping

        # Finally generate flattened targets from segmentation outlines
        if 'focusStack' in info:
            info = info.copy()
            cellFocus = info['focusStack']
            if type(cellFocus) != list:
                cellFocus = [cellFocus]
            # print('old focus = {}'.format(', '.join([str(f) for f in cellFocus])))
            info['focusStack'] = [f - self.refslice for f in cellFocus]
            # print('new focus = {}'.format(', '.join([str(f) for f in info['focusStack']])))

        lbl = self.targetgenfunc(lbl, info)

        return img, lbl

    def crop(self, img, lbl, xysize=None):
        if xysize is None:
            xysize = np.array(self.xy_out)

        # Find edges and fill cells before cropping
        for s in range(lbl.shape[2]):
            lbl[:, :, s] = _filled_canny(lbl[:, :, s])

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

    def __call__(self, img, lbl_info):
        lbl, info = lbl_info
        self._input_pix_size = info.get('pixel_size', self.target_pixel_size)
        self._scaling = self._input_pix_size / self.target_pixel_size
        self._outshape = np.round(np.array(self.xy_out) / self._scaling)
        img, lbl = super(ScalingAugmenter, self).__call__(img, lbl_info)
       # self._input_pix_size = None
       # iself.scaling = None
       # self._outshape = None
        return img, lbl

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

    def downscale(self, img, lbl, maxpix=None):
        # Scale image and label to target pixel size
        inshape = img.shape[:2]
        scaling = self._scaling

        # Apply random scaling according to probability for this op
        p = self.scale_prob
        if p == 1.0 or (p > 0 and np.random.uniform() < p):
            scaling += scaling * self.scale_frac * np.random.uniform(-1, 1)

        outshape = np.round(np.array(img.shape[:2]) * scaling)
        outshape = np.maximum(outshape, self.xy_out)

        return (transform.resize(img, outshape),
                transform.resize(lbl, outshape, anti_aliasing=False))


# =============== UTILITY FUNCTIONS ====================== #


def _apply_crop(stack, xysize):
    cropy, cropx = xysize
    starty, startx = stack.shape[:2]
    startx = (startx - cropx) // 2
    starty = (starty - cropy) // 2
    return stack[starty:(starty + cropy), startx:(startx + cropx), ...]


def _elastic_deform(image, alpha, sigma, random_state=None):
    """
    Elastic deformation of images as described in [Simard2003]_.
    [Simard2003] Simard, Steinkraus and Platt, "Best Practices for
    Convolutional Neural Networks applied to Visual Document Analysis", in
    Proc. of the International Conference on Document Analysis and
    Recognition, 2003.
    Adapted from:
    https://gist.github.com/chsasank/4d8f68caf01f041a6453e67fb30f8f5a
    """
    if random_state is None:
        random_state = np.random.RandomState(None)

    shape = image[:, :, 0].shape
    dx = gaussian_filter(
        (random_state.rand(*shape) * 2 - 1), sigma, mode="constant",
        cval=0) * alpha
    dy = gaussian_filter(
        (random_state.rand(*shape) * 2 - 1), sigma, mode="constant",
        cval=0) * alpha

    _x, _y = np.meshgrid(np.arange(shape[0]),
                         np.arange(shape[1]),
                         indexing='ij')
    indices = np.reshape(_x + dx, (-1, 1)), np.reshape(_y + dy, (-1, 1))
    if len(image.shape) == 3:
        result = np.empty_like(image)
        for d in range(image.shape[2]):
            # iterate over depth
            cval = np.median(image[:, :, d])
            result[:, :, d] = map_coordinates(image[:, :, d],
                                              indices,
                                              order=1,
                                              cval=cval).reshape(shape)
        result
    else:
        cval = np.median(image)
        result = map_coordinates(image, indices, order=1,
                                 cval=cval).reshape(shape)
    return result


def _filled_canny(segblur, bp=2):
    """Use canny to find edge and fill object

    Handles intersections with border by assuming that the object cannot
    intersect all borders at once.

    segblur:  segmentation image that has been gaussian blurred
    bp:       border padding
    """

    se = canny(np.pad(segblur, bp, 'edge'), sigma=0)
    sf = np.zeros(se.shape, dtype='bool')
    sf[bp:-bp, bp:-bp] = se[bp:-bp, bp:-bp]

    # The following assumes that the cell does not intersect with opposing
    # borders, filling bordering with two U-shaped border edges:
    sf[:bp, :] = True
    sf[-bp:, :] = True
    sf[:, :bp] = True
    sf = binary_fill_holes(sf)

    sf[:, :bp] = False
    sf[:, -bp:] = True
    sf = binary_fill_holes(sf)

    return sf[bp:-bp, bp:-bp]
