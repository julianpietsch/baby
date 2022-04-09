from setuptools import setup, find_packages

setup(
    name='baby',
    version='0.24',
    packages=find_packages('python'),
    package_dir={'': 'python'},
    include_package_data=True,
    entry_points={
        'console_scripts': [
            'baby-phone = baby.server:main',
            'baby-race = baby.speed_tests:main',
            'baby-fit-grs = baby.postprocessing:main'
            ]
        },
    url='',
    license='MIT License',
    author='Julian Pietsch',
    author_email='julian.pietsch@ed.ac.uk',
    description='Birth Annotator for Budding Yeast',
    long_description='''
If you publish results that make use of this software or the Birth Annotator
for Budding Yeast algorithm, please cite:
Julian M J Pietsch, Alán F Muñoz, Diane-Yayra A Adjavon, Ivan B N Clark, Peter
S Swain, 2022, A label-free method to track individuals and lineages of
budding cells (in submission).


The MIT License (MIT)

Copyright (c) Julian Pietsch, Alán Muñoz and Diane Adjavon 2022

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
    ''',
    install_requires=['tensorflow>=1.14,<2.4',
                      'scipy',
                      'numpy',
                      'pandas',
                      'scikit-image',
                      'scikit-learn==0.22.2',
                      'tqdm',
                      'imageio',
                      'pillow<9',
                      'matplotlib',
                      'aiohttp',
                      'gaussianprocessderivatives']
)
