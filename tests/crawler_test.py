import pytest
import numpy as np

from baby.crawler import BabyCrawler


def test_evolve_crawl(bb_evolve60, imgs_evolve60):
    imgstack = np.stack([v['Brightfield'][0] for v in imgs_evolve60.values()])
    crawler = BabyCrawler(bb_evolve60)
    # eqlen_outkeys = {'angles', 'radii', 'cell_label', 'mother_assign',
    #                  'edgemasks'}
    eqlen_outkeys = {'angles', 'radii', 'cell_label', 'edgemasks'}

    output0 = crawler.step(imgstack, with_edgemasks=True, assign_mothers=True)
    for o in output0:
        assert all([len(o['centres']) == len(o[k]) for k in eqlen_outkeys])

    # Step once to the same image (cells should track exactly)
    output1 = crawler.step(imgstack, with_edgemasks=True, assign_mothers=True)
    for o1, o0 in zip(output1, output0):
        assert all([len(o1['centres']) == len(o1[k]) for k in eqlen_outkeys])
        assert np.all(np.array(o0['cell_label']) == np.array(o1['cell_label']))

