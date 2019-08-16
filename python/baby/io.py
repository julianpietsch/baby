import json
import numpy as np
from imageio import imread


def load_tiled_image(filename):
    tImg = imread(filename)
    info = json.loads(tImg.meta.get('Description', '{}'))
    tw, th = info.get('tilesize', tImg.shape[0:2])
    nt = info.get('ntiles', 1)
    nr, nc = info.get('layout', (1, 1))
    nc_final_row = np.mod(nt, nc);
    img = np.zeros((tw, th, nt), dtype=tImg.dtype)
    for i in range(nr):
        i_nc = nc_final_row if i+1==nr and nc_final_row>0 else nc
        for j in range(i_nc):
            ind = i*nc+j
            img[:,:,ind] = tImg[i*tw:(i+1)*tw,j*th:(j+1)*th];
    return img, info

