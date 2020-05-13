import numpy as np
import itertools
from scipy.ndimage import binary_fill_holes

from typing import Union, Iterable, Any, Optional

from .segmentation import mask_containment, iterative_erosion, thresh_seg, \
    binary_edge, single_region_prop, outlines_to_radial, get_edge_scores, \
    refine_radial_grouped


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
        # if target_name in self:
        return pred[self.flattener.names.index(self[target_name])]
        # return None


class Group:
    def __init__(self, targets, min_area=10.,
                 use_thresh=False, thresh_expansion=0.,
                 interior_threshold=0.5, n_closing=0, n_opening=0,
                 connectivity=2):
        # Parameter assignment
        self.__connectivity = connectivity
        self.__min_area = min_area
        self.__use_thresh = use_thresh
        self.__thresh_expansion = thresh_expansion
        self.__n_closing = n_closing
        self.__n_opening = n_opening
        self.__interior_threshold = interior_threshold
        # TODO with ingroup_edge_segment
        self.n_edge_dilations = None

        # Subgroup targets
        self.targets = targets

        # Computed members
        self._n_erode = None

        # Dunno yet, probably functions
        self.areas = None
        self.coords = None
        self.edges = None
        self.masks = None
        self.edge_scores = None

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
    def max_nerode(self):
        return max(self._n_erode)

    # TODO remove edge stuff
    def erode_predictions(self, pred_interior, pred_edge):
        if len(pred_interior) == 1:
            pred_interior = pred_interior[0]
            pred_edge = pred_edge[0]
        else:
            # Perform morphological erosion on any group members that do not
            # match the maximum (NB: erosion here is non-binary):
            pred_interior = [iterative_erosion(p, self.max_n_erode -
                                               self.n_erode)
                             for p in pred_interior]
            pred_interior = np.dstack(pred_interior).max(axis=2)
            pred_edge = np.dstack(pred_edge).max(axis=2)
        return pred_interior, pred_edge

    def resolve_edge_scores(self, pred_edge=None, pedge_thresh=None):
        # TODO what if no edges?
        if pedge_thresh is None:
            return np.ones(len(self.masks))
        elif pred_edge is not None:
            return get_edge_scores(self.edges, pred_edge)
        else:
            raise ValueError("No edge target found for group")

    def segment(self, pred_interior, border_rect, fit_radial=False,
                refine_outlines=False):
        """
        Obtain the cell masks, areas, edges, and coordiantes from the
        prediction of the interior of the cell group.


        :param fit_radial:
        :param pred_interior: The predicted interior image
        :param border_rect: A boolean array delimiting the border of the
            prediction arrays
        """
        # TODO this is the bit where the use of the overlap informations
        #  from morph_thresh_masks should be added

        masks_areas = [(m, a) for m, a in thresh_seg(
            pred_interior, interior_threshold=self.__interior_threshold or 0.5,
            nclosing=self.__n_closing or 0,
            nopening=self.__n_opening or 0,
            ndilate=self.max_n_erode,
            return_area=True, connectivity=self.__connectivity)
                       if self.lower <= a < self.upper]

        if len(masks_areas) > 0:
            masks, areas = zip(*masks_areas)
        else:
            masks, areas = [], []
        edges = [binary_edge(m) for m in masks]

        if fit_radial:
            rprops = [single_region_prop(m) for m in masks]
            coords, edges = outlines_to_radial(edges, rprops,
                                               return_outlines=True)
            masks = [binary_fill_holes(o) for o in edges]
        else:
            edges = [e | (border_rect & m) for e, m in zip(edges, masks)]
            coords = [tuple()] * len(masks)

        if refine_outlines:
            pass

        self.areas = area
        self.coords = coords
        self.edges = edges
        self.masks = masks


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
                 ingroup_edge_segment=False,
                 containment_thresh=0.8, containment_func=mask_containment,
                 refine_outlines=False,
                 return_masks=False, return_coords=False):

        # Todo: assertions about valid options
        #  (e.g. 0 < interior_threshold < 1)
        # Assign options and parameters
        self.pedge_thresh = pedge_thresh
        self.fit_radial = fit_radial
        self.ingroup_edge_segment = ingroup_edge_segment
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

        # Initialize the targets(?) for the different groups
        self.groups = []
        for i, target_names in enumerate(self.cell_groups):
            targets = [Target(name, flattener) for name in target_names]
            self.groups.append(Group(targets, min_area=min_area[i],
                                     use_thresh=use_group_thresh,
                                     thresh_expansion=group_thresh_expansion,
                                     interior_threshold=interior_threshold[i],
                                     n_closing=n_closing[i],
                                     n_opening=n_opening[i],
                                     connectivity=connectivity[i]))

        self.group_segs = None

    # Todo: This is ideally the form of the input argument
    def contains(self, a, b):
        return self.containment_func(a, b) > self.containment_thresh

    # todo: parameters as arguments, group_segs as member attribute, remove
    #  return
    def remove_duplicates(self):
        """
        Resolve any cells duplicated across adjacent groups:

        :param group_segs:
        :return: The group segmentations with duplicates removed
        """

        if self.pedge_thresh is None:
            def accessor(group):
                return group[3]  # Todo: group class member
        else:
            def accessor(group):
                return group[1]

        # def accessor(group):
        #     return getattr(group, 'a' if self.pedge_thresh else 'b')

        for lower_group, upper_group in zip(self.group_segs,
                                            self.group_segs[1:]):
            pairs = [(lower, upper)
                     for lower in lower_group
                     for upper in upper_group
                     if self.contains(lower[0], upper[0])]

            for lower, upper in pairs:
                if accessor(lower) < accessor(upper):
                    lower_group.remove(lower)
                else:
                    upper_group.remove(upper)

    # Todo move to group class or remove completely
    def format_group_seg(self, group, areas, coords,
                         edges, masks, edge_scores):
        """
        Format the segmentation results for a given group (in `cellgroups`).
        If `self.pedge_thresh` is set, the segmentations are filtered by
        their edge scores.

        :param group: Group defined in the class' `cellgroups`
        :param areas: The areas segmented from this group
        :param coords: The active contouring coordinates (angles and radii) of
        the cells
        :param edges: The edges segmented from this group
        :param masks: The masks segmented from this group
        :param edge_scores: The edge scores, describing how well the edges
        obtained from processing overlap with the predicted edges

        :return: Formatted and filtered list of tuples `(mask, area,
        bounds, edge_score, edge, coordinates)` for the cells in this group
        """
        lower, upper = self.params[group]['bounds']
        group_seg = list(zip(masks, areas, len(masks) * (lower, upper),
                             edge_scores, edges, coords))
        # Only keep cells whose outlines overlap well with the predicted edge
        if self.pedge_thresh is not None:
            group_seg = [val for val in group if val[3] > self.pedge_thresh]
        return group_seg

    def extract_edges(self, p_edges, shape):
        if self.refine_outlines:
            # Refine outlines using edge predictions
            grouped_coords = [group.coords for group in self.groups]
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
            outputs = list(itertools.chain.from_iterable([
                [(edge, mask, coord) for mask, _, _, _, edge, coord in group]
                for group in group_segs]))

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

        # todo:
        #   if self.computed_member is None:
        #       self.compute_member()

        group_segs, p_edges = [], []
        for group in self.groups:

            # FIXME fails if target not available

            pred_interior, pred_edge = zip(*(
                map(lambda x: target.prediction(pred, x), ('interior', 'edge'))
                for target in group.targets))

            # Todo: only eroding the interiors
            pred_interior, pred_edge = group.erode_predictions(pred_interior,
                                                               pred_edge)
            # Save the edge predictions for edge refinement
            p_edges.append(pred_edge)

            # Fixme: this requires the predictions to be numpy arrays
            if self.ingroup_edge_segment:
                pred_interior *= (1 - pred_edge)
                if group.max_n_erode == 0:
                    group.max_n_erode = 1

            # Run instance segmentation on each group
            group.segment(pred_interior, border_rect,
                          fit_radial=self.fit_radial)
            group.resolve_edge_scores(pred_edge, self.pedge_thresh)

        # Remove cells that are duplicated in several groups
        self.remove_duplicates()
        #
        edges, masks, coords = self.extract_edges(group_segs, p_edges, shape)

        output = tuple(itertools.compress([edges, masks, coords],
                                          [True, self.return_masks,
                                           self.return_coords]))
        if len(output) > 1:
            return output
        else:
            return output[0]
