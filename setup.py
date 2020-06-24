from setuptools import setup, find_packages

setup(
    name='baby',
    version='0.1',
    packages=find_packages('python'),
    package_dir={'': 'python'},
    include_package_data=True,
    url='',
    license='',
    author='Julian Pietsch',
    author_email='julian.pietsch@ed.ac.uk',
    description='Birth Annotator for Budding Yeast',
    install_requires=['scipy',
                      'numpy',
                      'scikit-image',
                      'tensorflow',
                      'imageio',
                      'pillow',
                      'matplotlib']
)
