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


Group = list

class Group:
    def __init__(self, name):
        self.name = name
        self.targets = dict()
        self.n_erode = None
        self.max_n_erode = None
        self.lower = None
        self.upper = None
        self.areas = None
        self.coords = None
        self.edges = None
        self.masks = None
        self.edge_scores = None

    def assign_targets(self):
        pass

    def segment_group(self):
        pass


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

        # Todo: assertions about valid options (e.g. 0 < interior_threshold
        #  < 1)
        if cellgroups is None:
            cellgroups = ['large', 'medium', 'small']

        # Initialize the flattener
        # Todo: assign, not initialize
        self.flattener = flattener

        # Initialize parameters and hyperparameters
        self.ngroups = len(cellgroups)
        self.cellgroups = [(g,) if isinstance(g, str) else g for g in
                           cellgroups]
        self.interior_threshold = self.broadcast_arg(
            interior_threshold, 'interior_threshold')
        self.nclosing = self.broadcast_arg(nclosing, 'nclosing')
        self.nopening = self.broadcast_arg(nopening, 'nopening')
        self.min_area = self.broadcast_arg(min_area, 'min_area')
        self.connectivity = self.broadcast_arg(connectivity, 'connectivity')
        self.pedge_thresh = pedge_thresh

        self.tnames = flattener.names()

        # Initialize options
        # Todo: assign, not initialize
        self.fit_radial = fit_radial
        self.ingroup_edge_segment = ingroup_edge_segment
        self.containment_thresh = containment_thresh
        self.containment_func = containment_func
        self.refine_outlines = refine_outlines
        self.return_masks = return_masks
        self.return_coords = return_coords

        # todo: usually initialization is simple.
        # Initialize the targets(?) for the different groups
        params = dict()
        for group, min_area in zip(self.cellgroups, self.min_area):
            targets = dict()
            target_defs = []
            for subgroup in group:
                target_interior, target_filled, target_edge = \
                    flattener.getGroupTargets(
                        subgroup, ['interior', 'filled', 'edge'])
                target_interior = target_interior or target_filled
                assert target_interior is not None, \
                    '"{}" has no "interior" or "filled" target'.format(
                        subgroup)
                target_defs.append(flattener.getTargetDef(target_interior))
                targets[subgroup] = dict(target_interior=target_interior,
                                         target_edge=target_edge)

            if use_group_thresh:
                lower = target_defs[0].get('lower', 1)
                upper = target_defs[-1].get('upper', np.Inf)
                if upper == np.Inf:
                    expansion = group_thresh_expansion * lower
                else:
                    expansion = group_thresh_expansion * (upper - lower)
                lower -= expansion
                lower = max(lower, min_area)
                upper += expansion
            else:
                lower, upper = min_area, np.Inf

            nerode = [d.get('nerode', 0) for d in target_defs]
            max_nerode = np.max(nerode)
            params[group] = dict(targets=targets, bounds=(lower, upper),
                                 nerode=nerode, max_nerode=max_nerode)

        #  class Group:
        #      def __init__(self, targets, bounds, nerode, max_nerode):
        #          ...
        #      def erode_predictions(self):
        #

        self.params = params

    def broadcast_arg(self, arg: Union[Iterable, Any],
                      argname: Optional[str] = None):
        if argname is None:
            argname = 'Argument'
        try:
            if len(arg) != self.ngroups:
                raise ValueError(f'"{argname}" is of incorrect length')
        except TypeError:
            return [arg] * self.ngroups

    # Todo: This is ideally the form of the input argument
    def contains(self, a, b):
        return self.containment_func(a, b) > self.containment_thresh

    # todo: parameters as arguments, group_segs as member attribute, remove
    #  return
    def remove_duplicates(self, group_segs):
        """
        Resolve any cells duplicated across adjacent groups:

        :param group_segs:
        :return: The group segmentations with duplicates removed
        """

        if self.pedge_thresh is None:
            def accessor(group):
                return group[3] # Todo: group class member
        else:
            def accessor(group):
                return group[1]

        # def accessor(group):
        #     return getattr(group, 'a' if self.pedge_thresh else 'b')

        for lower_group, upper_group in zip(group_segs, group_segs[1:]):
            pairs = [(lower, upper)
                     for lower in lower_group
                     for upper in upper_group
                     if self.contains(lower[0], upper[0])]

            for lower, upper in pairs:
                if accessor(lower) < accessor(upper):
                    lower_group.remove(lower)
                else:
                    upper_group.remove(upper)

        return group_segs

    # todo: move to group class.
    def erode_predictions(self, pred_interior, pred_edge, group):
        max_nerode = self.params[group]['max_nerode']
        if len(pred_interior) == 1:
            pred_interior = pred_interior[0]
            pred_edge = pred_edge[0]
        else:
            # Perform morphological erosion on any group members that do not
            # match the maximum (NB: erosion here is non-binary):
            nerode = self.params[group]['nerode']
            pred_interior = [iterative_erosion(p, max_nerode - ne)
                             for p, ne in zip(pred_interior, nerode)]
            pred_interior = np.dstack(pred_interior).max(axis=2)
            pred_edge = np.dstack(pred_edge).max(axis=2)
        return pred_interior, pred_edge

    # Todo: Move to group class
    def segment_group(self, pred_interior, conn, nc, no,
                      thresh, border_rect, bounds, max_nerode):
        """
        Obtain the cell masks, areas, edges, and coordiantes from the
        prediction of the interior of the cell group.


        :param pred_interior: The predicted interior image
        :param thresh: The threshold probability for interior segmentation
        :param no: Number of openings to apply to interior prediction
        :param nc: Number of closings to apply to interior prediction
        :param conn: The connectivity for labelling
        :param max_nerode: The maximum number of erosions
        :param bounds: The size bounds for this group
        :param border_rect: A boolean array delimiting the border of the
            prediction arrays
        :return: areas, coords, edges, masks of the cells
        """
        # TODO this is the bit where the use of the overlap informations
        #  from morph_thresh_masks should be added
        lower, upper = bounds
        masks_areas = [(m, a) for m, a in thresh_seg(
            pred_interior, interior_threshold=thresh or 0.5,
            nclosing=nc or 0, nopening=no or 0, ndilate=max_nerode,
            return_area=True, connectivity=conn)
                       if lower <= a < upper]

        if len(masks_areas) > 0:
            masks, areas = zip(*masks_areas)
        else:
            masks, areas = [], []
        edges = [binary_edge(m) for m in masks]

        if self.fit_radial:
            rprops = [single_region_prop(m) for m in masks]
            coords, edges = outlines_to_radial(edges, rprops,
                                               return_outlines=True)
            masks = [binary_fill_holes(o) for o in edges]
        else:
            edges = [e | (border_rect & m) for e, m in zip(edges, masks)]
            coords = [tuple()] * len(masks)

        return areas, coords, edges, masks

    def resolve_edge_scores(self, target_edge, masks, group):
        if self.pedge_thresh is not None:
            assert target_edge is not None, \
                '"{}" has no "edge target"'.format(group)
            pred_edge = pred[self.tnames.index(target_edge)]
            edge_scores = get_edge_scores(edges, pred_edge)
        else:
            edge_scores = np.ones(len(masks))
        return edge_scores

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

    def extract_edges(self, group_segs, p_edges, shape):
        if self.refine_outlines:
            # Refine outlines using edge predictions
            grouped_coords = [[cell[-1] for cell in group] for group in
                              group_segs]
            coords = list(itertools.chain.from_iterable(
                refine_radial_grouped(grouped_coords, p_edges)))
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
        for group, thresh, nc, no, ma, conn in \
                zip(self.cellgroups, self.interior_threshold, self.nclosing,
                    self.nopening, self.min_area, self.connectivity):

            pred_interior, pred_edge = [], []
            for subgroup in group:
                # FIXME fails if target not available
                pred_interior.append(pred[self.tnames.index(
                    self.params[group][subgroup]['target_interior'])])
                pred_edge.append(pred[self.tnames.index(
                    self.params[group][subgroup]['target_edge'])])

            pred_interior, pred_edge = self.erode_predictions(pred_interior,
                                                              pred_edge, group)
            # Save the edge predictions for edge refinement
            p_edges.append(pred_edge)

            if self.ingroup_edge_segment:
                # todo: use *=
                pred_interior = pred_interior * (1 - pred_edge)
                if max_nerode == 0:
                    max_nerode = 1

            # Todo: Red flag; many return arguments. probably make some of
            #  them (computed) members.
            areas, coords, edges, masks = self.segment_group(
                pred_interior, conn, nc, no, thresh, border_rect,
                self.params[group]['bounds'], max_nerode)

            edge_scores = self.resolve_edge_scores(target_edge, masks, group)

            group_segs.append(self.format_group_seg(group, areas, coords,
                                                    edges, masks, edge_scores))
        group_segs = self.remove_duplicates(group_segs)
        edges, masks, coords = self.extract_edges(group_segs, p_edges, shape)

        output = tuple(itertools.compress([edges, masks, coords],
                                          [True, self.return_masks,
                                           self.return_coords]))
        if len(output) > 1:
            return output
        else:
            return output[0]
