import numpy as np

# Calculate barycentre
def calc_barycentre(centres, weights=None):
    '''
    :centres: ndarray containing the (x,y) centres of each cell
    :weights: (optional) list of weights to consider for each cell
    '''
    if weights is None:
        weights = np.ones_like(centres)

    if not np.sum(weights):
        weights = None

    barycentre = np.average(centres, axis=0, weights = weights)
    return(barycentre)

# Calculate distance to center
def calc_barydists(centres, bary):
    '''
    Calculate distances to the barycentre
    :centre: int (2,) tuple. Centre of cell
    :bary: float (2,) tuple. Barycentre of image
    '''
    vec2bary = centres - bary
    dists = np.sqrt(np.sum(vec2bary ** 2, axis=1))

    return dists

# Calculate angle to center
def calc_baryangles(centres, bary):
    '''
    Calculate angle using centre of cell and barycentre
    :centre: int (2,) tuple. Centre of cell
    :bary: float (2,) tuple. Barycentre of image
    '''
    
    angles = []
    vec2bary = centres - bary
    angles = np.apply_along_axis(lambda x: np.arctan2(*x), 1, vec2bary)

    return(angles)

def pick_baryfun(key):
    baryfuns = {'barydist':calc_barydists,
                'baryangle':calc_baryangles}
    return(baryfuns[key])

# Fix single-cell case
# Add calculated features to feature vector
# Add models with extended tracking included
