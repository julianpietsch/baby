import json
import re
from itertools import groupby
import numpy as np
from imageio import imread, imwrite
from PIL.PngImagePlugin import PngInfo


class LayoutError(Exception):
    pass


class UnpairedImagesError(Exception):
    pass


def load_tiled_image(filename):
    tImg = imread(filename)
    info = json.loads(tImg.meta.get('Description', '{}'))
    tw, th = info.get('tilesize', tImg.shape[0:2])
    nt = info.get('ntiles', 1)
    nr, nc = info.get('layout', (1, 1))
    nc_final_row = np.mod(nt, nc)
    img = np.zeros((tw, th, nt), dtype=tImg.dtype)
    for i in range(nr):
        i_nc = nc_final_row if i+1==nr and nc_final_row>0 else nc
        for j in range(i_nc):
            ind = i*nc+j
            img[:,:,ind] = tImg[i*tw:(i+1)*tw,j*th:(j+1)*th]
    return img, info


def save_tiled_image(img, filename, info={}, layout=None):
    if layout is not None and len(layout) != 2:
        raise LayoutError('"layout" must a 2-tuple')

    tw, th, nt = img.shape
    if not layout or layout == (None, None):
        nr = int(np.floor(np.sqrt(nt)))
        nc = None
    else:
        nr, nc = layout

    if nr is None:
        nr = int(np.ceil(nt / nc))
    if nc is None:
        nc = int(np.ceil(nt / nr))

    info['tilesize'] = (tw, th)
    info['ntiles'] = nt
    info['layout'] = (nr, nc)

    nc_final_row = np.mod(nt, nc)
    tImg = np.zeros((tw * nr, th * nc), dtype=img.dtype)
    for i in range(nr):
        i_nc = nc_final_row if i+1==nr and nc_final_row>0 else nc
        for j in range(i_nc):
            ind = i*nc+j
            tImg[i*tw:(i+1)*tw,j*th:(j+1)*th] = img[:,:,ind]

    meta = PngInfo()
    meta.add_text('Description', json.dumps(info))
    imwrite(filename, tImg, format='png', pnginfo=meta,
            prefer_uint8=False)


def load_paired_images(filenames, typeA='Brightfield', typeB='segoutlines'):
    re_imlbl = re.compile(r'^(.*)_(' + typeA + r'|' + typeB + r')\.png$')
    # For groupby to work, the list needs to be sorted; also has the side
    # effect of ensuring filenames is no longer a generator
    filenames = sorted(filenames)
    matches = [re_imlbl.match(f.name) for f in filenames]
    valid = filter(lambda m: m[0], zip(matches, filenames))
    grouped = {k: {m.group(2): f for m, f in v}
               for k, v in groupby(valid, key=lambda m: m[0].group(1))} 
    valid = [set(v.keys()).issuperset({typeA, typeB})
             for v in grouped.values()]
    if not all(valid):
        raise UnpairedImagesError
    return {l: {t: load_tiled_image(f) for t, f in g.items()}
            for l, g in grouped.items()}
