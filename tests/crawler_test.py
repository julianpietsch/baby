import pytest
import numpy as np

from baby.crawler import BabyCrawler
from baby.io import save_tiled_image


def test_evolve_crawl(bb_evolve60, imgs_evolve60, tmp_path,
                      save_crawler_output):
    imgstack = np.stack([v['Brightfield'][0] for v in imgs_evolve60.values()])
    crawler = BabyCrawler(bb_evolve60)
    # eqlen_outkeys = {'angles', 'radii', 'cell_label', 'mother_assign',
    #                  'edgemasks'}
    eqlen_outkeys = {'angles', 'radii', 'cell_label', 'edgemasks'}
    info_keys = ('centres', 'angles', 'radii', 'cell_label')

    output0 = crawler.step(imgstack, with_edgemasks=True, assign_mothers=True)
    for o, l in zip(output0, imgs_evolve60.keys()):
        assert all([len(o['centres']) == len(o[k]) for k in eqlen_outkeys])
        edges = o['edgemasks']
        assert edges.shape[1:3] == imgstack.shape[1:3]
        if save_crawler_output and len(o['centres']) > 0:
            save_tiled_image(edges.transpose([1, 2, 0]).astype('uint16'),
                             tmp_path / '{}_t0_segoutlines.png'.format(l),
                             {k: o[k] for k in info_keys},
                             layout=(1, None))

    # Step once to the same image (cells should track exactly)
    output1 = crawler.step(imgstack, with_edgemasks=True, assign_mothers=True)
    for o1, o0 in zip(output1, output0):
        assert all([len(o1['centres']) == len(o1[k]) for k in eqlen_outkeys])
        assert np.all(
            np.array(o0['cell_label']) == np.array(o1['cell_label']))
