import json
import numpy as np
from imageio import imread
from scipy.ndimage.morphology import binary_fill_holes
from skimage.measure import regionprops
from skimage.draw import polygon


def load_tiled_image(filename):
    tImg = imread(filename)
    info = json.loads(tImg.meta['Description'])
    tw, th = info['tilesize']
    nt = info['ntiles']
    nr, nc = info['layout']
    nc_final_row = np.mod(nt, nc);
    img = np.zeros((tw, th, nt), dtype=tImg.dtype)
    for i in range(nr):
        i_nc = nc_final_row if i+1==nr and nc_final_row>0 else nc
        for j in range(i_nc):
            ind = i*nc+j
            img[:,:,ind] = tImg[i*th:(i+1)*th,j*tw:(j+1)*tw];
    return img, info


def preprocess_brightfield(img):
    """Robust normalisation of intensity to [-1,1]"""
    img = img.copy().astype('float')
    low, mid, high = np.quantile(img, (0.02, 0.5, 0.98))
    return (img - mid)*0.96/(high - low)


def preprocess_segim(edge_stack, info):
    """Returns a stack of images in order:
        edge: edges for all cells flattened into a single layer
        filled: filled area for all cells flattened into a single layer
        interiors: filled area excluding the edges
        overlap: regions occupied by more than one cell
        budring: edges located between mother and bud centres
        bud: buds that are are smaller than a fractional threshold of the mother
    """
    imsize = edge_stack.shape[0:2]
    ncells = edge_stack.shape[2]
    imout = np.zeros(imsize+(6,), dtype='bool')

    edge_stack = edge_stack>0
    edge_flat = np.any(edge_stack, axis=2)
    fill_stack = np.concatenate([
        binary_fill_holes(edge_stack[:,:,i])[..., np.newaxis]
        for i in range(ncells)], axis=2)
    fill_flat = np.any(fill_stack, axis=2)

    imout[:,:,0] = edge_flat # edge
    imout[:,:,1] = fill_flat # filled
    imout[:,:,2] = fill_flat & ~edge_flat # interiors
    imout[:,:,3] = np.sum(fill_stack, axis=2)>1 # overlap

    bud_pairs = [(m, np.nonzero(np.array(info['cellLabels'])==b)[0][0])
                 for m, b in enumerate(info['buds']) if b>0]

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
        imout[:,:,4] |= (edge_stack[:,:,m] | edge_stack[:,:,b]) & r_im

        # Smaller buds
        if cell_info[b].minor_axis_length < 15:
            imout[:,:,5] |= fill_stack[:,:,b]

    return imout

