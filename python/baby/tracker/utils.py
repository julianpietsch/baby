import numpy as np

# Calculate barycentre
def calc_barycentre(centres, weights=None, **kwargs):
    '''
    :centres: ndarray containing the (x,y) centres of each cell
    :weights: (optional) list of weights to consider for each cell
    '''
    if weights is None:
        weights = np.ones_like(centres)

    barycentre = np.average(centres, axis=0, weights = weights)
    return(barycentre)

# Calculate distance to center
def calc_barydists(centres, bary, **kwargs):
    '''
    Calculate distances to the barycentre
    :centre: int (2,) tuple. Centre of cell
    :bary: float (2,) tuple. Barycentre of image
    '''
    vec2bary = centres - bary
    dists = np.sqrt(np.sum(vec2bary ** 2, axis=1))

    return dists

# Calculate angle to center
def calc_baryangles(centres, bary, areas=None, **kwargs):
    '''
    Calculate angle using centre of cell and barycentre
    :centre: int (2,) tuple. Centre of cell
    :bary: float (2,) tuple. Barycentre of image
    :anchor_cell: int Cell id to use as angle 0.
    '''

    angles = []
    vec2bary = centres - bary
    angles = np.apply_along_axis(lambda x: np.arctan2(*x), 1, vec2bary)
    if areas is not None:
        anchor_cell = np.argmax(areas)
        angles -= angles[anchor_cell]

    return(angles)

def pick_baryfun(key):
    baryfuns = {'barydist':calc_barydists,
                'baryangle':calc_baryangles}
    return(baryfuns[key])

##  Tracking benchmark utils

def lol_to_adj(lol):
    '''
    Convert a series list of lists with cell ids into a matrix
    representing a graph.

    Note that information is lost in the process, and a matrix can't be
    turned back into a list of list by itself.

    input

    :lol: list of lists with cell ids 

    returns

    :adj_matrix: (n, n) ndarray where n is the number of cells
    
    '''
    n = len([y for x in lol for y in x])
    adj_mat = np.zeros((n,n))

    prev = None
    cur = 0
    for l in lol:
        if not prev:
            prev = l
        else:
            for i, el in enumerate(l):
                prev_idx = prev.index(el) if el in prev else None
                if prev_idx is not None:
                    adj_mat[cur + len(prev) + i, cur + prev_idx] = True
            cur += len(l)

    return adj_mat

def compare_pred_truth_lols(prediction, truth):
    '''
    input

    :prediction: list of lists with predicted cell ids
    :truth: list of lists with real cell ids

    returns

    number of diferences between equivalent truth matrices

    '''
    adj_pred = lol_to_adj(prediction)
    adj_truth = lol_to_adj(truth)

    return(int(((adj_pred - adj_truth) != 0).sum()))
