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
import itertools

from scipy import ndimage

from typing import Union, Iterable, Any, Optional, List, NamedTuple

from .errors import BadParam
from .segmentation import mask_containment, iterative_erosion, thresh_seg, \
    binary_edge, single_region_prop, outline_to_radial, get_edge_scores, \
    refine_radial_grouped, iterative_dilation, draw_radial
from .volume import volume


# class ContainmentFunction:
#     def __init__(self, threshold: float = .8):
#         self.threshold = threshold
#
#     def __call__(self, *args, **kwargs):
#         return mask_containment(*args, **kwargs) > self.threshold


# Making a callable object instance:
#
# a = lambda a, b: a + b
#
# def a(a, b):
#     return a + b
#
# class A:
#     def __call__(self, a, b):
#         return a + b
# a = A()

class Cell:
    def __init__(self, area, mask, predicted_edge, border_rect,
                 fit_radial=True):
        self.area = area
        self.mask = mask
        self.predicted_edge = predicted_edge
        self.fit_radial = fit_radial
        self.border_rect = border_rect

        self._coords = None
        self._edge = None
        self._edge_score = None

    @property
    def edge_score(self):
        if self._edge_score is None:
            self._edge_score = get_edge_scores((self.edge,),
                                               self.predicted_edge)[0]
        return self._edge_score

    def _calculate_properties(self, fit_radial):
        self._edge = binary_edge(self.mask)
        if fit_radial:
            rprop = single_region_prop(self.mask)
            coords, edge = outline_to_radial(self.edge, rprop,
                                             return_outline=True)
            self.mask = ndimage.binary_fill_holes(edge)
        else:
            edge = self._edge | (self.border_rect & self.mask)
            coords = tuple()
        self._coords = coords
        self._edge = edge

    @property
    def edge(self):
        if self._edge is None:
            self._calculate_properties(fit_radial=self.fit_radial)
        return self._edge

    @property
    def coords(self):
        if self._coords is None:
            self._calculate_properties(fit_radial=self.fit_radial)
        return self._coords

    @property
    def volume(self, method='conical'):
        return volume(self.edge, method=method)


class Target:
    def __init__(self, name, flattener, desired_targets=None):
        self.name = name
        self.flattener = flattener
        self._definition = None
        self._available_targets = dict()

        if desired_targets is None:
            desired_targets = ['interior', 'filled', 'edge', 'overlap']
        self.__desired_targets = desired_targets

    @property
    def available_targets(self):
        if not self._available_targets:
            self._calculate_targets()
        return self._available_targets

    def __getitem__(self, item):
        if item in self.available_targets and self.available_targets[item] \
                is None:
            self._calculate_targets()
        return self.available_targets[item]

    def __contains__(self, item):
        return item in self.available_targets

    def _calculate_targets(self):
        targets = self.flattener.getGroupTargets(self.name,
                                                 self.__desired_targets)
        self._available_targets.update(dict(zip(
            self.__desired_targets,
            targets)))

        none_targets = [k for k, v in self._available_targets.items()
                        if v is None]
        for none_target in none_targets:
            del self._available_targets[none_target]

        if 'interior' not in self._available_targets:
            if 'filled' in self._available_targets:
                self._available_targets['interior'] = self._available_targets[
                    'filled']
            else:
                raise ValueError('No interior or filled target specified in '
                                 f'available targets {self.available_targets}')

    @property
    def definition(self):
        if self._definition is None:
            self._definition = self.flattener.getTargetDef(
                self['interior'])
        return self._definition

    def prediction(self, pred, target_name):
        if target_name in self:
            result = pred[self.flattener.names().index(self[target_name])]
        else:
            result = np.zeros((pred.shape[1], pred.shape[2]))
        return result


class Group:

    def __init__(self,
                 targets,
                 min_area=10.,
                 use_thresh=False,
                 thresh_expansion=0.,
                 pedge_thresh=None,
                 interior_threshold=0.5,
                 n_closing=0,
                 n_opening=0,
                 connectivity=2,
                 edge_sub_dilations=None):
        # Parameter assignment
        self.__connectivity = connectivity
        self.__min_area = min_area
        self.__use_thresh = use_thresh
        self.__thresh_expansion = thresh_expansion
        self.__pedge_thresh = pedge_thresh
        self.__n_closing = n_closing
        self.__n_opening = n_opening
        self.__interior_threshold = interior_threshold
        self.edge_sub_dilations = edge_sub_dilations

        # Subgroup targets
        self.targets = targets

        # Computed members
        self._n_erode = None
        self._lower = None
        self._upper = None

        # Dunno yet, probably functions
        self.cells = []

    def _calculate_bounds(self):
        # get bounds
        if self.__use_thresh:
            lower = min(target.definition.get('lower', 1.) for target in
                        self.targets)
            upper = max(target.definition.get('upper', float('inf')) for
                        target in self.targets)
            expansion = self.__thresh_expansion * (lower
                                                   if upper == float('inf')
                                                   else upper - lower)
            lower = max(lower - expansion, self.__min_area)
            upper += expansion
        else:
            lower, upper = self.__min_area, float('inf')
        self._lower = lower
        self._upper = upper

    @property
    def lower(self):
        if self._lower is None:
            self._calculate_bounds()
        return self._lower

    @property
    def upper(self):
        if self._upper is None:
            self._calculate_bounds()
        return self._upper

    @property
    def n_erode(self):
        if self._n_erode is None:
            self._n_erode = [target.definition.get('nerode', 0)
                             for target in self.targets]
        return self._n_erode

    @property
    def max_n_erode(self):
        max_n_erode = max(self.n_erode)
        if self.edge_sub_dilations is not None and max_n_erode == 0:
            return 1
        else:
            return max_n_erode  # Todo: add number of dilations?

    @property
    def n_cells(self):
        return len(self.cells)

    @property
    def interior_threshold(self):
        return self.__interior_threshold or 0.5

    def prediction(self, pred, target_name, erode=False):
        predictions = [target.prediction(pred, target_name)
                       for target in self.targets]
        if erode:
            predictions = self.erode_predictions(predictions)
        result = np.dstack(predictions)
        return result.max(axis=2)

    def erode_predictions(self, predictions: List) -> List:
        # Perform morphological erosion on any group members that do not
        # match the maximum (NB: erosion here is non-binary):
        predictions = [iterative_erosion(p, self.max_n_erode - n_erode)
                       for p, n_erode in zip(predictions, self.n_erode)]
        return predictions

    def segment(self, pred, border_rect, fit_radial=False):
        """
        Obtain the cell masks, areas, edges, and coordiantes from the
        prediction of the interior of the cell group.


        :param fit_radial:
        :param pred: The neural network's prediction
        :param border_rect: A boolean array delimiting the border of the
            prediction arrays
        """

        # TODO this is the bit where the use of the overlap informations
        #  from morph_thresh_masks should be added
        pred_interior = self.prediction(pred, 'interior', erode=True)
        pred_edge = self.prediction(pred, 'edge', erode=False)

        if self.edge_sub_dilations is not None:
            pred_edge = iterative_dilation(pred_edge,
                                           self.edge_sub_dilations)
            pred_interior *= (1 - pred_edge)

        masks_areas = [(m, a) for m, a in thresh_seg(
            pred_interior, interior_threshold=self.__interior_threshold or 0.5,
            nclosing=self.__n_closing or 0,
            nopening=self.__n_opening or 0,
            ndilate=self.max_n_erode,
            return_area=True, connectivity=self.__connectivity)
                       if self.lower <= a < self.upper]
        self.cells = [Cell(a, m, pred_edge, border_rect, fit_radial=fit_radial)
                      for m, a in masks_areas]
        # Remove cells that do not exceed the p_edge threshold
        if self.__pedge_thresh is not None:
            self.cells = [cell for cell in self.cells
                          if cell.edge_score > self.__pedge_thresh]


def broadcast_arg(arg: Union[Iterable, Any],
                  argname: Optional[str] = None,
                  n_groups: Optional[int] = 3):
    if argname is None:
        argname = 'Argument'
    try:
        if len(arg) != n_groups:
            raise ValueError(f'"{argname}" is of incorrect length')
        else:
            return arg
    except TypeError:
        return [arg] * n_groups


class MorphSegGrouped:
    def __init__(self, flattener, cellgroups=None,
                 interior_threshold=0.5, nclosing=0, nopening=0,
                 connectivity=2,
                 min_area=10, pedge_thresh=None, fit_radial=False,
                 use_group_thresh=False, group_thresh_expansion=0.,
                 edge_sub_dilations=None,
                 containment_thresh=0.8, containment_func=mask_containment,
                 return_masks=False, return_coords=False, return_volume=False):
        """

        :param flattener:
        :param cellgroups:
        :param interior_threshold:
        :param nclosing:
        :param nopening:
        :param connectivity:
        :param min_area:
        :param pedge_thresh:
        :param fit_radial:
        :param use_group_thresh:
        :param group_thresh_expansion:
        :param edge_sub_dilations:
        :param containment_thresh:
        :param containment_func:
        :param return_masks:
        :param return_coords:
        """
        # Todo: assertions about valid options
        #  (e.g. 0 < interior_threshold < 1)
        # Assign options and parameters
        self.pedge_thresh = pedge_thresh
        self.fit_radial = fit_radial
        self.containment_thresh = containment_thresh
        self.containment_func = containment_func
        self.return_masks = return_masks
        self.return_coords = return_coords
        self.return_volume = return_volume

        self.flattener = flattener

        # Define  group parameters
        if cellgroups is None:
            cellgroups = ['large', 'medium', 'small']
        cellgroups = [(g,) if isinstance(g, str) else g for g in cellgroups]
        n_groups = len(cellgroups)
        interior_threshold = broadcast_arg(interior_threshold,
                                           'interior_threshold', n_groups)
        n_closing = broadcast_arg(nclosing, 'nclosing', n_groups)
        n_opening = broadcast_arg(nopening, 'nopening', n_groups)
        min_area = broadcast_arg(min_area, 'min_area', n_groups)
        connectivity = broadcast_arg(connectivity, 'connectivity', n_groups)
        pedge_thresh = broadcast_arg(pedge_thresh, 'pedge_thresh', n_groups)
        group_thresh_expansion = broadcast_arg(group_thresh_expansion,
                                               'group_thresh_expansion',
                                               n_groups)
        edge_sub_dilations = broadcast_arg(edge_sub_dilations,
                                           'edge_substraction_dilations',
                                           n_groups)

        # Minimum area must be larger than 1 to avoid generating cells with
        # no size:
        min_area = [np.max([a, 1]) for a in min_area]

        # Initialize the different groups and their targets
        self.groups = []
        for i, target_names in enumerate(cellgroups):
            targets = [Target(name, flattener) for name in target_names]
            self.groups.append(Group(targets, min_area=min_area[i],
                                     use_thresh=use_group_thresh,
                                     thresh_expansion=group_thresh_expansion[i],
                                     pedge_thresh=pedge_thresh[i],
                                     interior_threshold=interior_threshold[i],
                                     n_closing=n_closing[i],
                                     n_opening=n_opening[i],
                                     connectivity=connectivity[i],
                                     edge_sub_dilations=edge_sub_dilations[i]))
        self.group_segs = None

    # Todo: This is ideally the form of the input argument
    def contains(self, a, b):
        return self.containment_func(a, b) > self.containment_thresh

    def remove_duplicates(self):
        """
        Resolve any cells duplicated across adjacent groups:

        :param group_segs:
        :return: The group segmentations with duplicates removed
        """

        if self.pedge_thresh is None:
            def accessor(cell):
                return cell.area
        else:
            def accessor(cell):
                return cell.edge_score

        # def accessor(group):
        #     return getattr(group, 'a' if self.pedge_thresh else 'b')

        for lower_group, upper_group in zip(self.groups,
                                            self.groups[1:]):
            pairs = [(lower, upper)
                     for lower in lower_group.cells
                     for upper in upper_group.cells
                     if self.contains(lower.mask, upper.mask)]

            for lower, upper in pairs:
                # Check that one of the cells hasn't already been removed in
                # previous pairs
                if lower in lower_group.cells and upper in upper_group.cells:
                    if accessor(lower) < accessor(upper):
                        lower_group.cells.remove(lower)
                    else:
                        upper_group.cells.remove(upper)

    # Todo: rename
    def extract_edges(self, pred, shape, refine_outlines, return_volume):
        masks = [[]]
        if refine_outlines:
            if not self.fit_radial:
                raise BadParam(
                    '"refine_outlines" requires "fit_radial" to have been specified'
                )
            # Refine outlines using edge predictions
            grouped_coords = [[cell.coords for cell in group.cells]
                              for group in self.groups]
            predicted_edges = [group.prediction(pred, 'edge') for group in
                               self.groups]

            if predicted_edges:
                coords = list(itertools.chain.from_iterable(
                    refine_radial_grouped(grouped_coords,
                                          predicted_edges)))
            else:
                coords = tuple()
            edges = [draw_radial(radii, angles, centre, shape)
                     for centre, radii, angles in coords]
            if self.return_masks:
                masks = [ndimage.binary_fill_holes(e) for e in edges]
            if return_volume:
                volumes = [volume(edge, method='conical') for edge in edges]
                return edges, masks, coords, volumes
            return edges, masks, coords
        else:
            # Extract edges, masks and AC coordinates from initial segmentation
            if return_volume:
                outputs = [(cell.edge, cell.mask, cell.coords, cell.volume)
                           for group in self.groups
                           for cell in group.cells]
            else:
                outputs = [(cell.edge, cell.mask, cell.coords)
                           for group in self.groups
                           for cell in group.cells]

            if len(outputs) > 0:
                return zip(*outputs)
            else:
                return 4 * [[]] if return_volume else 3 * [[]]

    def segment(self, pred, refine_outlines=False, return_volume=False):
        """
        Take the output of the neural network and turn it into an instance
        segmentation output.

        :param pred: list of prediction images (ndarray with shape (x, y))
        matching `self.flattener.names()`
        :return: a list of boolean edge images (ndarray shape (x, y)), one for
        each cell identified. If `return_masks` and/or `return_coords` are
        true, the output will be a tuple of edge images, filled masks, and/or
        radial coordinates.
        """
        if len(pred) != len(self.flattener.names()):
            raise BadParam(
                '"pred" arg does not match number of flattener targets')
        shape = np.squeeze(pred[0]).shape
        border_rect = np.pad(
            np.zeros(tuple(x - 2 for x in shape), dtype='bool'),
            pad_width=1, mode='constant',
            constant_values=True)

        for group in self.groups:
            group.segment(pred, border_rect, fit_radial=self.fit_radial)

        # Remove cells that are duplicated in several groups
        self.remove_duplicates()
        # edges, masks, coords, volumes = \
        result = self.extract_edges(pred, shape,
                                    refine_outlines=refine_outlines,
                                    return_volume=return_volume)

        # Todo: return_masks and return_coords seem useless as always set
        #  necessary for brain.segment and tracker
        output = SegmentationOutput(*result)
        # [True,
        #  self.return_masks,
        #  self.return_coords,
        #  self.return_volume]))

        return output
        # if len(output) > 1:
        #     return output
        # else:
        #     return output[0]


class SegmentationOutput(NamedTuple):
    edges: list
    masks: list = []
    coords: list = []
    volume: list = []
