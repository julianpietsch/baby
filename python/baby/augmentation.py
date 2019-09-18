"""
A set of augmentation functions for on-the-fly preprocessing of images before
input into the Neural Network. All augmentation functions are meant to work
with unstructured arrays (in particular, elastic_deform, turn, and the shifts
probably will return errors if used on structured arrays. They all take exactly
two arrays as input, and will perform identitcal transformations on both arrays.
"""
from __future__ import absolute_import, division, print_function
import numpy as np
from scipy.ndimage import map_coordinates, gaussian_filter, shift
from scipy.ndimage.morphology import binary_fill_holes
from skimage import transform
from skimage.feature import canny
from skimage.filters import gaussian
from skimage.draw import rectangle_perimeter

from .preprocessing import segoutline_flattening


AUGMENTATION_ORDER = ('substacks', 'rotate', 'vshift', 'hshift', 'downscale',
                      'crop', 'vflip', 'movestacks', 'noise')


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
        elif len(xy_out)==2:
            self.xy_out = xy_out
        else:
            raise Exception('"xy_out" must be an int or pair of ints')

        self.xy_in = None

        self.aug_order = [a for a in AUGMENTATION_ORDER
                          if probs.get(a,1)>0 or a=='crop']

        # Treat 'crop' and 'substacks' specially to have p = 1
        guaranteed_augs = ('crop', 'substacks')
        n_augs = len(self.aug_order) - len(guaranteed_augs)
        p_default = (1.-p_noop)**n_augs/n_augs
        self.probs = np.array([
            1 if a in guaranteed_augs else probs.get(a, p_default)
            for a in self.aug_order
        ])

        self.nsubstacks = substacks


    def __call__(self, img, lbl):
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
        assert img.shape[:2] == lbl.shape[:2], \
            'xy dimensions of img and lbl are mismatched'

        lbl_is_bool = lbl.dtype == 'bool'

        self.xy_in = img.shape[:2]

        for a, p in zip(self.aug_order, self.probs):
            if p==0.0:
                continue
            elif p==1.0:
                img, lbl = getattr(self, a)(img, lbl)
            else:
                if np.random.uniform()<p:
                    img, lbl = getattr(self, a)(img, lbl)
                    if lbl_is_bool and lbl.dtype != 'bool':
                        lbl = lbl > 0.5 # ensure label stays boolean

        self.xy_in = None

        # Ensure that x and y dimensions match the intended output size
        assert img.shape[:2] == self.xy_out and lbl.shape[:2] == self.xy_out, \
            'xy dimensions do not match intended size after augmentation'

        return (img, lbl)


    def substacks(self, img, lbl):
        if self.nsubstacks == 1:
            img = img[:,:,np.random.choice(range(img.shape[2])),None]
        elif self.nsubstacks == 3:
            choices = [[True, False, True, False, True],
                       [True, True, False, False, True],
                       [True, False, False, True, True],
                       [False, True, True, False, True],
                       [False, True, False, True, True],
                       [True, False, True, True, False],
                       [True, True, False, True, False]]
            img = img[:,:,np.random.choice(choices)]

        return img, lbl


    def rotate(self, img, lbl):
        """Random rotation

        Example image:

        .. image:: ../report/figures/augmentations/turn.*
        """
        angle = np.random.choice(360)
        return (
            transform.rotate(img, angle=angle, mode='reflect', resize=True),
            transform.rotate(lbl, angle=angle, mode='reflect', resize=True)
        )


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
            maxpix = np.max([0, (inshape - self.xy_out[0])//2])

        pix = np.random.choice(np.arange(-maxpix, maxpix+1, dtype='int'))
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
            maxpix = np.max([0, (inshape - self.xy_out[1])//2])

        pix = np.random.choice(np.arange(-maxpix, maxpix+1, dtype='int'))
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

        pix = np.random.choice(maxpix+1)
        pix = maxpix
        img_crop, lbl_crop = self.crop(
            img, lbl, xysize=np.array(self.xy_out)+pix)

        return (
            transform.resize(img_crop, self.xy_out),
            transform.resize(
                lbl_crop, self.xy_out, anti_aliasing_sigma=0.05) > 0.3
        )


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

        if np.random.uniform()<0.5:
            # Move img stacks up by one
            img[:,:,1:] = img[:,:,:-1]
        else:
            # Move img stacks down by one
            img[:,:,:-1] = img[:,:,1:]

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


class SmoothedLabelAugmenter(Augmenter):
    def __init__(self, sigmafunc, targetgenfunc=segoutline_flattening, **kwargs):
        super(SmoothedLabelAugmenter, self).__init__(**kwargs)
        self.sigmafunc = sigmafunc
        self.targetgenfunc = targetgenfunc


    def __call__(self, img, lbl_info):
        """This augmenter needs to be used in combination with a label
        preprocessing function that returns both images and info.
        """

        lbl, info = lbl_info

        lbl_stack = []
        for s in np.dsplit(lbl, lbl.shape[2]):
            s = np.squeeze(s)
            # Only smooth label if it is non-empty (allows for empty traps)
            if s.sum() > 0:
                s = gaussian(binary_fill_holes(s), self.sigmafunc(s))
            lbl_stack += [s[..., np.newaxis]]
        lbl = np.concatenate(lbl_stack, axis=2)

        img, lbl = super(SmoothedLabelAugmenter, self).__call__(img, lbl)

        # NB: to limit open shapes, the crop operation has been overloaded to
        # find edges before cropping

        # Finally generate flattened targets from segmentation outlines
        lbl = self.targetgenfunc(lbl, info)

        return img, lbl


    def crop(self, img, lbl, xysize=None):
        if xysize is None:
            xysize = np.array(self.xy_out)

        # Find edges and fill cells before cropping
        for s in range(lbl.shape[2]):
            lbl[:,:,s] = _filled_canny(lbl[:,:,s])

        return _apply_crop(img, xysize), _apply_crop(lbl, xysize)


    def downscale(self, img, lbl, maxpix=None):
        inshape = img.shape[:2]
        if maxpix is None:
            if self.xy_in is not None:
                inshape = self.xy_in
            maxpix = np.max([0, np.min(inshape) - np.max(self.xy_out)])

        # Allow for small chance of no scaling, but robust if image sizes equal
        pix = np.random.choice(maxpix+1)
        scaling = (np.min(inshape)-pix)/np.min(inshape)
        outshape = np.floor(np.array(img.shape[:2])*scaling)

        return (
            transform.resize(img, outshape),
            transform.resize(lbl, outshape, anti_aliasing=False)
        )


# =============== UTILITY FUNCTIONS ====================== #


def _apply_crop(stack, xysize):
    cropy, cropx = xysize
    starty, startx = stack.shape[:2]
    startx = (startx - cropx)//2
    starty = (starty - cropy)//2
    return stack[starty:(starty+cropy), startx:(startx+cropx), ...]


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
    dx = gaussian_filter((random_state.rand(*shape) * 2 - 1),
                        sigma, mode="constant", cval=0) * alpha
    dy = gaussian_filter((random_state.rand(*shape) * 2 - 1),
                        sigma, mode="constant", cval=0) * alpha

    _x, _y = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]),
                        indexing='ij')
    indices = np.reshape(_x + dx, (-1, 1)), np.reshape(_y + dy, (-1, 1))
    if len(image.shape) == 3:
        result = np.empty_like(image)
        for d in range(image.shape[2]):
            # iterate over depth
            cval = np.median(image[:, :, d])
            result[:, :, d] = map_coordinates(image[:, :, d],
                                            indices, order=1,
                                            cval=cval).reshape(shape)
        result
    else:
        cval = np.median(image)
        result = map_coordinates(image, indices,
                                order=1, cval=cval).reshape(shape)
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
    sf[bp:-bp,bp:-bp] = se[bp:-bp,bp:-bp]

    # The following assumes that the cell does not intersect with opposing
    # borders, filling bordering with two U-shaped border edges:
    sf[:bp,:] = True
    sf[-bp:,:] = True
    sf[:,:bp] = True
    sf = binary_fill_holes(sf)

    sf[:,:bp] = False
    sf[:,-bp:] = True
    sf = binary_fill_holes(sf)

    return sf[bp:-bp,bp:-bp]

