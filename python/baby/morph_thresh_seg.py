import numpy as np
import itertools
from scipy.ndimage import binary_fill_holes

from typing import Union, Iterable, Any, Optional

from .segmentation import mask_containment, iterative_erosion, thresh_seg, \
    binary_edge, single_region_prop, outlines_to_radial, get_edge_scores, \
    refine_radial_grouped, iterative_dilation


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
    def __init__(self, area, coords, edge, mask, predicted_edge):
        self.area = area
        self.coords = coords
        self.edge = edge
        self.mask = mask
        self._edge_score = None
        self.predicted_edge = predicted_edge

    @property
    def edge_score(self):
        if self._edge_score is None:
            self._edge_score = get_edge_scores(self.edge,
                                               self.predicted_edge)[0]
        return self._edge_score



class Target:
    def __init__(self, name, flattener, available_targets=None):
        self.name = name
        self.flattener = flattener
        self._interior = None
        self._overlap = None
        self._edge = None
        self._definition = None

        if available_targets is None:
            available_targets = ['interior', 'filled', 'edge', 'overlap']
        self.available_targets = {target: None for target in available_targets}

    def __getitem__(self, item):
        if item in self.available_targets and \
                self.available_targets[item] is None:
            self._calculate_targets()

        return self.available_targets[item]

    def __contains__(self, item):
        return item in self.available_targets

    def _calculate_targets(self):
        self.available_targets.update(dict(zip(
            self.available_targets.keys(),
            self.flattener.getGroupTargets(
                self.name,
                self.available_targets.keys()))))

        none_targets = [k for k, v in self.available_targets.items()
                        if v is None]
        for none_target in none_targets:
            del self.available_targets[none_target]

        if 'interior' not in self.available_targets \
                and 'filled' in self.available_targets:
            self.available_targets['interior'] = filled
        else:
            raise ValueError('No interior or filled target specified')

    @property
    def definition(self):
        if self._definition is None:
            self._definition = self.flattener.getTargetDef(
                self['interior'])
        return self._definition

    def prediction(self, pred, target_name):
        if target_name in self:
            return pred[self.flattener.names.index(self[target_name])]
        return np.zeros(pred.shape[0], pred.shape[1])


class Group:
    def __init__(self, targets, min_area=10.,
                 use_thresh=False, thresh_expansion=0.,
                 interior_threshold=0.5, n_closing=0, n_opening=0,
                 connectivity=2, edge_sub_dilations=None):
        # Parameter assignment
        self.__connectivity = connectivity
        self.__min_area = min_area
        self.__use_thresh = use_thresh
        self.__thresh_expansion = thresh_expansion
        self.__n_closing = n_closing
        self.__n_opening = n_opening
        self.__interior_threshold = interior_threshold
        self.edge_sub_dilations = edge_sub_dilations

        # Subgroup targets
        self.targets = targets

        # Computed members
        self._n_erode = None

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
            lower = max(lower - expansion, self.min_area)
            upper += expansion
        else:
            lower, upper = self.min_area, float('inf')
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
            return max_n_erode # Todo: add number of dilations?

    @property
    def n_cells(self):
        return len(self.cells)

    def prediction(self, pred, target_name, erode=False):
        predictions = [pred[target.prediction(target_name)]
                       for target in self.targets]
        if erode:
            predictions = self.erode_predictions(predictions)
        return np.dstack(predictions).max(axis=2)

    def erode_predictions(self, predictions: List) -> List:
        if len(predictions) == 1:
            predictions = predictions[0]
        else:
            # Perform morphological erosion on any group members that do not
            # match the maximum (NB: erosion here is non-binary):
            predictions = [iterative_erosion(p, self.max_n_erode -
                                             self.n_erode)
                           for p in predictions]
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

        if len(masks_areas) > 0:
            masks, areas = zip(*masks_areas)
            edges = [binary_edge(m) for m in masks]
            # Todo move to Cell
            if fit_radial:
                rprops = [single_region_prop(m) for m in masks]
                coords, edges = outlines_to_radial(edges, rprops,
                                                   return_outlines=True)
                masks = [binary_fill_holes(o) for o in edges]
            else:
                edges = [e | (border_rect & m) for e, m in zip(edges, masks)]
                coords = [tuple()] * len(masks)
        else:
            masks, areas, edges, coords = [], [], [], []

        self.cells = [Cell(a, e, c, m, p_edge)
                      for a, e, c, m in zip(areas, edges, coords, masks)]


def broadcast_arg(arg: Union[Iterable, Any],
                  argname: Optional[str] = None,
                  n_groups: Optional[int] = 3):
    if argname is None:
        argname = 'Argument'
    try:
        if len(arg) != n_groups:
            raise ValueError(f'"{argname}" is of incorrect length')
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
                 refine_outlines=False,
                 return_masks=False, return_coords=False):

        # Todo: assertions about valid options
        #  (e.g. 0 < interior_threshold < 1)
        # Assign options and parameters
        self.pedge_thresh = pedge_thresh
        self.fit_radial = fit_radial
        self.containment_thresh = containment_thresh
        self.containment_func = containment_func
        self.refine_outlines = refine_outlines
        self.return_masks = return_masks
        self.return_coords = return_coords

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
        edge_sub_dilations = broadcast_arg(edge_sub_dilations,
                                           'edge_substraction_dilations',
                                           n_groups)

        # Initialize the different groups and their targets
        self.groups = []
        for i, target_names in enumerate(self.cell_groups):
            targets = [Target(name, flattener) for name in target_names]
            self.groups.append(Group(targets, min_area=min_area[i],
                                     use_thresh=use_group_thresh,
                                     thresh_expansion=group_thresh_expansion,
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
                return cell.areas
        else:
            def accessor(cell):
                return cell.edge_scores

        # def accessor(group):
        #     return getattr(group, 'a' if self.pedge_thresh else 'b')

        for lower_group, upper_group in zip(self.groups,
                                            self.groups[1:]):
            pairs = [(lower, upper)
                     for lower in lower_group.cells
                     for upper in upper_group.cells
                     if self.contains(lower.masks, upper.masks)]

            for lower, upper in pairs:
                if accessor(lower) < accessor(upper):
                    lower_group.remove(lower)
                else:
                    upper_group.remove(upper)

    # Todo: rename
    def extract_edges(self, shape):
        masks = [[]]
        if self.refine_outlines:
            # Refine outlines using edge predictions
            grouped_coords = [cell.coords for group in self.groups
                              for cell in group.cells]
            predicted_edges = [[target.prediction('edge') for target in
                                group.targets] for group in self.groups]
            coords = list(itertools.chain.from_iterable(
                refine_radial_grouped(grouped_coords, predicted_edges)))
            edges = [draw_radial(radii, angles, centre, shape)
                     for centre, radii, angles in coords]
            if self.return_masks:
                masks = [binary_fill_holes(e) for e in edges]
        else:
            # Extract edges, masks and AC coordinates from initial segmentation
            outputs = [(cell.edge, cell.mask, cell.coords)
                       for group in self.groups
                       for cell in group.cells]
            if len(outputs) > 0:
                edges, masks, coords = zip(*outputs)
            else:
                edges, masks, coords = 3 * [[]]
        return edges, masks, coords

    def segment(self, pred):
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
        if len(pred) == 0:
            raise Exception('there must be at least one prediction image')
        shape = np.squeeze(pred[0]).shape
        border_rect = np.pad(
            np.zeros(tuple(x - 1 for x in shape), dtype='bool'),
            pad_width=1, mode='constant',
            constant_values=True)

        for group in self.groups:
            group.segment(pred, border_rect, fit_radial=self.fit_radial)

        # Remove cells that are duplicated in several groups
        self.remove_duplicates()
        edges, masks, coords = self.extract_edges(shape)

        output = tuple(itertools.compress([edges, masks, coords],
                                          [True, self.return_masks,
                                           self.return_coords]))
        if len(output) > 1:
            return output
        else:
            return output[0]
