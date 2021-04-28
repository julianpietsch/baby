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
import numpy as np
from skimage import img_as_float
from scipy.ndimage import (
    generic_filter, minimum_filter, maximum_filter,
    binary_fill_holes, binary_dilation, binary_erosion, binary_closing
)
from skimage.measure import regionprops
from skimage.draw import polygon
from skimage.morphology import diamond
from itertools import chain
from typing import NamedTuple, Union
import json

from .segmentation import binary_edge
from .utils import EncodableNamedTuple, jsonify, as_python_object

# Depth-wise structuring elements for square or full connectivity
dwsquareconn = diamond(1)[..., None]
dwfullconn = np.ones((3,3,1), dtype='uint8')

def raw_norm(img, info):
    """Keep raw values but scale non-float images to [0, 1]"""
    return img_as_float(img, force_copy=True)


def standard_norm(img, info):
    """Normalise intensity to [0, 1]"""
    img = img.copy().astype('float');
    imin, imax = np.min(img), np.max(img)
    imrange = imax - imin
    imrange = 1 if imrange == 0 else imrange
    return (img - imin) / imrange


def logical_norm(img, info):
    return img > 0


def robust_norm(img, info, q_outliers=0.04):
    """Robust normalisation of intensity to [-1,1]"""
    img = img.copy().astype('float')
    hq = q_outliers / 2
    low, mid, high = np.quantile(img, (hq, 0.5, 1-hq))
    imrange = high - low
    imrange = 1 if imrange == 0 else imrange
    return (img - mid) * (1 - q_outliers) / imrange


def seg_norm(img, info):
    img = img > 0
    # Connect any 1-pixel gaps:
    imconn = generic_filter(img.astype('int'), np.sum, footprint=dwfullconn)
    imconn = binary_erosion(imconn > 1, dwsquareconn) | img
    return imconn, info


def segoutline_flattening(fill_stack, info):
    """Returns a stack of images in order:
        edge: edges for all cells flattened into a single layer
        filled: filled area for all cells flattened into a single layer
        interiors: filled area excluding the edges
        overlap: regions occupied by more than one cell
        budring: edges located between mother and bud centres
        bud: buds that are are smaller than a fractional threshold of the mother
    """
    imsize = fill_stack.shape[0:2]
    ncells = fill_stack.shape[2]
    imout = np.zeros(imsize+(6,), dtype='bool')

    edge_stack = binary_edge(fill_stack, dwsquareconn)

    edge_flat = np.any(edge_stack, axis=2)
    fill_flat = np.any(fill_stack, axis=2)
    overlap = np.sum(fill_stack, axis=2)>1

    imout[:,:,0] = edge_flat  # edge
    imout[:,:,1] = fill_flat  # filled
    imout[:,:,2] = fill_flat & ~edge_flat & ~overlap  # interiors
    imout[:,:,3] = overlap  # overlap

    bud_pairs = [(m, np.nonzero(np.array(info.get('cellLabels', []))==b)[0][0])
                 for m, b in enumerate(info.get('buds', []) or []) if b>0]

    cell_info = [
        regionprops(fill_stack[:,:,i].astype('int32'), coordinates='rc')
        for i in range(fill_stack.shape[2])
    ]
    cell_info = [p[0] if len(p)>0 else None for p in cell_info]

    for m, b in bud_pairs:
        if cell_info[m] is None or cell_info[b] is None:
            # Label possible transformed outside field of view by augmentation
            continue
        if m == b:
            raise Exception('a mother cannot be its own bud')
        m_centre = np.array(cell_info[m].centroid).T
        b_centre = np.array(cell_info[b].centroid).T
        r_width = cell_info[b].minor_axis_length*0.25
        r_hvec = b_centre-m_centre
        r_wvec = np.matmul(np.array([[0,-1],[1,0]]), r_hvec)
        if np.linalg.norm(r_wvec) == 0:
            raise Exception('mother and bud have coincident centres')
        r_wvec = r_width*r_wvec/np.linalg.norm(r_wvec)
        r_points = np.zeros((2,4))
        r_points[:,0] = m_centre-0.5*r_wvec
        r_points[:,1] = r_points[:,0] + r_hvec
        r_points[:,2] = r_points[:,1] + r_wvec
        r_points[:,3] = r_points[:,2] - r_hvec
        r_inds, c_inds = polygon(r_points[0,:], r_points[1,:], imsize)
        r_im = np.zeros(fill_stack.shape[0:2], dtype='bool')
        r_im[r_inds, c_inds] = 1

        # Bud junction
        bj = (edge_stack[:,:,m] | edge_stack[:,:,b]) & r_im
        imout[:,:,4] |= binary_dilation(binary_closing(bj))

        # Smaller buds
        if (cell_info[b].area / cell_info[m].area) < 0.7:
            imout[:,:,5] |= fill_stack[:,:,b]

    return imout


@EncodableNamedTuple
class CellGroup(NamedTuple):
    """Defines a cell group for creation of SegmentationFlattening targets
    """
    lower: Union[int, float] = 1
    upper: Union[int, float] = np.Inf
    budonly: bool = False
    focus: Union[int, float, None] = None


@EncodableNamedTuple
class PredTarget(NamedTuple):
    """Defines a target for SegmentationFlattening objects
    """
    name: str
    group: str
    prop: str
    nerode: int = 0  # Erosions applied to cells before flattening
    ndilate: int = 0  # Dilations applied to cells before flattening
    ndilate_overlaps: int = 0  # Dilations applied after determining overlaps
    ndilate_mother: int = 2 # Dilations applied only to mothers


class UnrecognisedProp(Exception):
    pass


class SegmentationFlattening(object):
    def __init__(self, filename=None):
        self.propdepends = {
            'filled': set(),
            'filledsum': set(),
            'edge': set(),
            'overlap': set(),
            'interior': {'overlap'},
            'budneck': set()
        }

        self.groupdef = {}
        self.groupprops = {}
        self.targets = []

        if filename is not None:
            self.load(filename)

    def addGroup(self, name, lower=1, upper=np.Inf,
                 budonly=False, focus=None):
        """Add a new cell group to this flattener

        :param name: a unique name to identify this group
        :param lower: the lower size threshold (in pixels) of cells to include
            in this group
        :param upper: the upper size threshold (in pixels) of cells to include
            in this group
        :param budonly: whether to limit this group to cells annotated as buds
        :param focus: a float specifying the focal plane that this group
            should correspond to. Cells with a focus annotation will be
            allocated to the group with the closest focal match.
        """
        assert name not in self.groupdef, \
            '"{}" group already exists'.format(name)
        self.groupdef[name] = CellGroup(lower, upper, budonly, focus)
        self.groupprops[name] = set()

    def addTarget(self, name, group, prop, nerode=0, ndilate=0,
                  focusStacks=[]):
        """Add a new prediction target to this flattener

        :param name: a unique name to identify this target
        :param group: the name identifying the cell group from which this
            target should be generated
        :param prop: the type of mask that should be generated for this
            target. Valid values are currently:
            - 'filled': True for all edge and interior pixels in the specified
              group, False otherwise;
            - 'edge': True for all edge pixels of cells in the specified
              group, False otherwise;
            - 'overlap': True for all pixels corresponding to at least two
              cells in the specified group, False otherwise;
            - 'interior': True for all pixels
            - 'filledsum': The integer number of cells present at each pixel;
            - 'budneck': For all cells in this group, if a mother has been
              annotated, then set to True those pixels where the daughter
              overlaps with a dilated version of its mother
        :param nerode: the number of erosions that should be applied for
            generation of this target.
        :param ndilate: the number of dilations that should be applied for
            generation of this target.
        :param focusStacks: a list of floats specifying focal planes. If
            non-empty, this is shorthand for creating a new group for each of
            the specified focal planes (copying the properties of the
            specified cell group) and creating a corresponding target.
        """
        assert name not in {t.name for t in self.targets}, \
            '"{}" target already exists'.format(name)
        assert group in self.groupdef, \
            '"{}" group does not exist'.format(group)
        assert prop in self.propdepends, \
            '"{}" is not a valid property'.format(prop)

        if len(focusStacks) > 0:
            gdict = self.groupdef[group]._asdict()
            del gdict['focus']
            for f in focusStacks:
                fgroup = '_'.join([group, str(f)])
                self.addGroup(fgroup, focus=f, **gdict)
                self.addTarget('_'.join([name, str(f)]),
                               fgroup, prop, nerode=nerode,
                               ndilate=ndilate, focusStacks=[])
        else:
            self.targets.append(PredTarget(name, group, prop, nerode, ndilate))
            self.groupprops[group] = self.groupprops[group].union(
                {prop}, self.propdepends[prop])

    def names(self):
        return tuple(t.name for t in self.targets)

    def getGroupTargets(self, group, propfilter=None):
        assert group in self.groupdef, \
            '"{}" group does not exist'.format(group)
        if type(propfilter) == list:
            # Return targets in order of properties provided
            grouptargets = {t.prop: t.name for t in self.targets if t.group == group}
            return [grouptargets.get(p) for p in propfilter]
        else:
            if propfilter is None:
                propfilter = self.propdepends

            # Return filtered targets in order of addition
            return [t.name for t in self.targets
                    if t.group == group and t.prop in propfilter]

    def getTargetDef(self, name):
        target = [t for t in self.targets if t.name == name]
        assert len(target) == 1 , \
            '"{}" target does not exist'.format(name)

        target = target[0]
        tdef = self.groupdef[target.group]._asdict()
        tdef.update(target._asdict())
        return tdef

    def save(self, filename):
        with open(filename, 'wt') as f:
            json.dump(jsonify({
                'groupdef': self.groupdef,
                'groupprops': self.groupprops,
                'targets': self.targets
            }), f)

    def load(self, filename):
        with open(filename, 'rt') as f:
            data = json.load(f, object_hook=as_python_object)

        # Map any legacy versions of the group definitions to CellGroup
        gdefs = data.get('groupdef', {})
        self.groupdef = {k: g if isinstance(g, CellGroup)
                         else CellGroup(*(g[:3] + g[4:5]))
                         for k, g in gdefs.items()}
        # Map any legacy versions of the target definitions to PredTarget
        self.targets = [t if isinstance(t, PredTarget)
                        else PredTarget(*(t[0:3] + gdefs[t[1]][3:4]))
                        for t in data.get('targets', [])]
        self.groupprops = data.get('groupprops', {})

    def __call__(self, filled_stack, info):
        filled_stack = filled_stack > 0
        shape = filled_stack.shape
        if len(shape) == 2:
            filled_stack = filled_stack[..., None]
            shape = filled_stack.shape

        areas = filled_stack.sum(axis=(0,1))
        if shape[2] == 1 and areas.sum() == 0:
            ncell = 0
        else:
            ncell = shape[2]

        shape = shape[0:2]

        props = set(chain.from_iterable(self.groupprops.values()))

        if 'edge' in props:
            edge_stack = binary_edge(filled_stack, dwsquareconn)

        if ncell == 0:
            bmpairs = []
            isbud = np.array([], dtype='bool')
        else:
            buds = info.get('buds', ()) or ()
            if type(buds) == int:
                buds = [buds]
            labels = info.get('cellLabels', ()) or ()
            if type(labels) == int:
                labels = [labels]
            assert len(labels) == len(areas)
            bmpairs = [(m, labels.index(bl))
                       for m, bl in enumerate(buds)
                       if bl > 0 and bl in labels]
            buds = set(buds).difference((0,))
            labels = list(labels)
            isbud = np.array([l in buds for l in labels], dtype='bool')

        budonly = {True: isbud, False: np.ones(isbud.shape, dtype='bool')}

        # If the group does not limit to focus, then all cells are candidates
        focusAssignments = {}
        focusAssignments[None] = np.ones(areas.shape, dtype='bool')

        # Process focus if any groups require it
        focusNums = list({g.focus for g in self.groupdef.values()
                          if g.focus is not None})
        if len(focusNums) > 0:
            cellFocus = info.get('focusStack', [])
            if type(cellFocus) != list:
                cellFocus = [cellFocus]
            assert len(cellFocus) == ncell, 'image with bad focus label found'

            # print('focusNums = {}'.format(', '.join([str(f) for f in focusNums])))
            fNumArray = np.array(focusNums)
            cellAssignment = np.array([np.argmin(np.abs(fNumArray - f))
                                    for f in cellFocus])
            # print('cellAssignment = {}'.format(', '.join([str(f) for f in cellAssignment])))
            focusAssignments = {f: cellAssignment == i
                                for i, f in enumerate(focusNums)}

        groupinds = {
            k: np.flatnonzero((areas >= g.lower) & (areas < g.upper)
                              & budonly[g.budonly] & focusAssignments[g.focus])
            for k, g in self.groupdef.items()
        }

        targetims = [filled_stack[..., []]]
        for t in self.targets:
            g = t.group
            inds = groupinds[g].tolist()

            if t.prop in {'edge'}:
                imstack = edge_stack[:, :, inds]
            else:
                imstack = filled_stack[:, :, inds]

            # Apply specified dilations and/or erosions to each cell
            # independently:
            if t.ndilate > 0:
                imstack = binary_dilation(
                    imstack, dwsquareconn, iterations=t.ndilate)
            if t.nerode > 0:
                imstack = binary_erosion(
                    imstack, dwsquareconn, iterations=t.nerode)

            # Apply property-specific flattening operations
            if t.prop in {'filled', 'edge'}:
                imflat = imstack.any(axis=2)
            elif t.prop == 'filledsum':
                imflat = imstack.sum(axis=2)
            elif t.prop in {'overlap', 'interior'}:
                # Overlaps between cells in this group
                # NB: gets reused below in for 'interior' calculation
                imflat = imstack.sum(axis=2) > 1
                if t.ndilate_overlaps > 0:
                    imflat = binary_dilation(
                        imflat, iterations=t.ndilate_overlaps)
            elif t.prop == 'budneck':
                imflat = np.zeros(shape, dtype='bool')
                for m, b in bmpairs:
                    # Skip buds outside the specified group:
                    if b not in inds:
                        continue
                    # Get bud mask from imstack (with erosions/dilations)
                    bim = imstack[..., inds.index(b)]
                    # Get mother mask from complete stack and apply
                    # mother-specific dilation:
                    mim = binary_dilation(filled_stack[..., m],
                                          iterations=t.ndilate_mother)
                    imflat |= mim & bim
            else:
                raise UnrecognisedProp(
                    'Unrecognised prop "{}"'.format(t.prop))

            if t.prop == 'interior':
                # Like 'filled' except that overlaps are also excluded
                # Overlaps are used as calculated and stored in `imflat`
                imflat = imstack.any(axis=2) & (~imflat)

            targetims.append(imflat)

        return np.dstack(targetims)


def flattener_norm_func(flattener):
    def norm_func(img, info):
        img, info = seg_norm(img, info)
        img = binary_fill_holes(img, dwsquareconn)
        return flattener(img, info)

    return norm_func
