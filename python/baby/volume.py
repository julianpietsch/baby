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
Volume estimation utilities

Notes: I don't have statistics on ranges of radii for each of the knots in
the radial spline representation, but we regularly extract the average of
these radii for each cell. So, depending on camera/lens, we get:
    * 60x evolve: mean radii of 2-14 pixels (and measured areas of 30-750
    pixels^2)
    * 60x prime95b: mean radii of 3-24 pixels (and measured areas of 60-2000
	pixels^2)

And I presume that for a 100x lens we would get an ~5/3 increase over those
values.

In terms of the current volume estimation method, it's currently only
implemented in the AnalysisToolbox repository, but it's super simple:

mVol = 4/3*pi*sqrt(mArea/pi).^3

where mArea is simply the sum of pixels for that cell.
"""
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scipy import ndimage
from skimage.morphology import erosion, ball
from skimage import measure, draw


def my_ball(radius):
    """Generates a ball-shaped structuring element.

    This is the 3D equivalent of a disk.
    A pixel is within the neighborhood if the Euclidean distance between
    it and the origin is no greater than radius.

    Parameters
    ----------
    radius : int
        The radius of the ball-shaped structuring element.

    Other Parameters
    ----------------
    dtype : data-type
        The data type of the structuring element.

    Returns
    -------
    selem : ndarray
        The structuring element where elements of the neighborhood
        are 1 and 0 otherwise.
    """
    n = 2 * radius + 1
    Z, Y, X = np.mgrid[-radius:radius:n * 1j,
              -radius:radius:n * 1j,
              -radius:radius:n * 1j]
    X **= 2
    Y **= 2
    Z **= 2
    X += Y
    X += Z
    return X <= radius * radius


def circle_outline(r):
    return ellipse_perimeter(r, r)


def ellipse_perimeter(x, y):
    im_shape = int(2 * max(x, y) + 1)
    img = np.zeros((im_shape, im_shape), dtype=np.uint8)
    rr, cc = draw.ellipse_perimeter(int(im_shape // 2), int(im_shape // 2),
                                    int(x), int(y))
    img[rr, cc] = 1
    return np.pad(img, 1)


def capped_cylinder(x, y):
    max_size = (y + 2 * x + 2)
    pixels = np.zeros((max_size, max_size))

    rect_start = ((max_size - x) // 2, x + 1)
    rr, cc = draw.rectangle_perimeter(rect_start, extent=(x, y),
                                      shape=(max_size, max_size))
    pixels[rr, cc] = 1
    circle_centres = [(max_size // 2 - 1, x),
                      (max_size // 2 - 1, max_size - x - 1)]
    for r, c in circle_centres:
        rr, cc = draw.circle_perimeter(r, c, (x + 1) // 2,
                                       shape=(max_size, max_size))
        pixels[rr, cc] = 1
    pixels = ndimage.morphology.binary_fill_holes(pixels)
    pixels ^= erosion(pixels)
    return pixels


def volume_of_sphere(radius):
    return 4 / 3 * np.pi * radius ** 3


def plot_voxels(voxels):
    verts, faces, normals, values = measure.marching_cubes_lewiner(
        voxels, 0)
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    mesh = Poly3DCollection(verts[faces])
    mesh.set_edgecolor('k')
    ax.add_collection3d(mesh)
    ax.set_xlim(0, voxels.shape[0])
    ax.set_ylim(0, voxels.shape[1])
    ax.set_zlim(0, voxels.shape[2])
    plt.tight_layout()
    plt.show()


# Volume estimation
def union_of_spheres(outline, shape='my_ball', debug=False):
    filled = ndimage.binary_fill_holes(outline)
    nearest_neighbor = ndimage.morphology.distance_transform_edt(
        outline == 0) * filled
    voxels = np.zeros((filled.shape[0], filled.shape[1], max(filled.shape)))
    c_z = voxels.shape[2] // 2
    for x, y in zip(*np.where(filled)):
        radius = nearest_neighbor[(x, y)]
        if radius > 0:
            if shape == 'ball':
                b = ball(radius)
            elif shape == 'my_ball':
                b = my_ball(radius)
            else:
                raise ValueError(f"{shape} is not an accepted value for "
                                 f"shape.")
            centre_b = ndimage.measurements.center_of_mass(b)

            I, J, K = np.ogrid[:b.shape[0], :b.shape[1], :b.shape[2]]
            voxels[I + int(x - centre_b[0]), J + int(y - centre_b[1]),
                   K + int(c_z - centre_b[2])] += b
    if debug:
        plot_voxels(voxels)
    return voxels.astype(bool).sum()


def improved_uos(outline, shape='my_ball', debug=False):
    filled = ndimage.binary_fill_holes(outline)
    nearest_neighbor = ndimage.morphology.distance_transform_edt(
        outline == 0) * filled
    voxels = np.zeros((filled.shape[0], filled.shape[1], max(filled.shape)))
    c_z = voxels.shape[2] // 2

    while np.any(nearest_neighbor != 0):
        radius = np.max(nearest_neighbor)
        x, y = np.argwhere(nearest_neighbor == radius)[0]
        if shape == 'ball':
            b = ball(np.ceil(radius))
        elif shape == 'my_ball':
            b = my_ball(np.ceil(radius))
        else:
            raise ValueError(f"{shape} is not an accepted value for shape")
        centre_b = ndimage.measurements.center_of_mass(b)

        I, J, K = np.ogrid[:b.shape[0], :b.shape[1], :b.shape[2]]
        voxels[I + int(x - centre_b[0]), J + int(y - centre_b[1]),
               K + int(c_z - centre_b[2])] += b

        # Use the central disk of the ball from voxels to get the circle
        # = 0 if nn[x,y] < r else nn[x,y]
        rr, cc = draw.circle(x, y, np.ceil(radius), nearest_neighbor.shape)
        nearest_neighbor[rr, cc] = 0
    if debug:
        plot_voxels(voxels)
    return voxels.astype(bool).sum()


def conical(outline, debug=False):
    filled = ndimage.binary_fill_holes(outline)
    nearest_neighbor = ndimage.morphology.distance_transform_edt(
        outline == 0) * filled
    if debug:
        hf = plt.figure()
        ha = hf.add_subplot(111, projection='3d')

        X, Y = np.meshgrid(np.arange(nearest_neighbor.shape[0]),
                           np.arange(nearest_neighbor.shape[1]))
        ha.plot_surface(X, Y, nearest_neighbor)
        plt.show()
    return 4 * nearest_neighbor.sum() + filled.sum()


def volume(outline, method='spheres'):
    if method == 'conical':
        return conical(outline)
    elif method == 'spheres':
        return union_of_spheres(outline)
    else:
        raise ValueError(f"Method {method} not implemented.")
