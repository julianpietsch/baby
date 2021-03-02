# If you publish results that make use of this software or the Birth Annotator
# for Budding Yeast algorithm, please cite:
# Julian M J Pietsch, Alán Muñoz, Diane Adjavon, Ivan B N Clark, Peter S
# Swain, 2021, Birth Annotator for Budding Yeast (in preparation).
# 
# 
# The MIT License (MIT)
# 
# Copyright (c) Julian Pietsch, Alán Muñoz and Diane Adjavon 2021
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to
# deal in the Software without restriction, including without limitation the
# rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
# sell copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# 
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
# IN THE SOFTWARE.
import pytest
import numpy as np

from baby.crawler import BabyCrawler
from baby.io import save_tiled_image


@pytest.fixture(scope='function')
def save_output(tmp_path, save_crawler_output):
    info_keys = ('centres', 'angles', 'radii', 'cell_label')

    def savefn(crawlerout, imnames, suffix=None):
        for o, l in zip(crawlerout, imnames):
            edges = o['edgemasks']
            if len(edges) == 0:
                continue
            filename = '_'.join(([l, suffix] if suffix else [l]) +
                                ['segoutlines.png'])
            save_tiled_image(edges.transpose([1, 2, 0]).astype('float'),
                             tmp_path / filename,
                             {k: o[k] for k in info_keys},
                             layout=(1, None))

    if save_crawler_output:
        return savefn
    else:
        return lambda crawlerout, imnames: None


def test_evolve_crawl(bb_evolve60, imgs_evolve60, save_output):
    imgstack = np.stack([v['Brightfield'][0] for v in imgs_evolve60.values()])
    crawler = BabyCrawler(bb_evolve60)
    # eqlen_outkeys = {'angles', 'radii', 'cell_label', 'mother_assign',
    #                  'edgemasks'}
    eqlen_outkeys = {'angles', 'radii', 'cell_label', 'edgemasks'}

    output0 = crawler.step(imgstack, with_edgemasks=True, assign_mothers=True)
    for o in output0:
        assert all([len(o['centres']) == len(o[k]) for k in eqlen_outkeys])
        assert o['edgemasks'].shape[1:3] == imgstack.shape[1:3]
        if o['edgemasks'].shape[0] > 0:
            assert o['edgemasks'].any()

    save_output(output0, imgs_evolve60.keys())

    # Step once to the same image (cells should track exactly)
    output1 = crawler.step(imgstack, with_edgemasks=True, assign_mothers=True)
    for o1, o0 in zip(output1, output0):
        assert all([len(o1['centres']) == len(o1[k]) for k in eqlen_outkeys])
        assert np.all(
            np.array(o0['cell_label']) == np.array(o1['cell_label']))


def test_evolve_crawl_step(bb_evolve60, imgs_evolve60, save_output):
    eqlen_outkeys = ('angles', 'radii', 'cell_label', 'edgemasks')
    tp3 = imgs_evolve60['evolve_testF_tp3']['Brightfield'][0][None, ...]
    tp4 = imgs_evolve60['evolve_testF_tp4']['Brightfield'][0][None, ...]
    crawler = BabyCrawler(bb_evolve60)
    output3 = crawler.step(tp3, with_edgemasks=True)
    for o in output3:
        assert all([len(o['centres']) == len(o[k]) for k in eqlen_outkeys])
    output4 = crawler.step(tp4, with_edgemasks=True)
    for o in output4:
        assert all([len(o['centres']) == len(o[k]) for k in eqlen_outkeys])
