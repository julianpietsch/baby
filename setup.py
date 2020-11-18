from setuptools import setup, find_packages

setup(
    name='baby',
    version='0.1',
    packages=find_packages('python'),
    package_dir={'': 'python'},
    include_package_data=True,
    entry_points={
        'console_scripts': [
            'baby-phone = baby.server:main',
            'baby-race = baby.speed_tests:main'
            ]
        },
    url='',
    license='',
    author='Julian Pietsch',
    author_email='julian.pietsch@ed.ac.uk',
    description='Birth Annotator for Budding Yeast',
    install_requires=['scipy',
                      'numpy',
                      'pandas',
                      'scikit-image',
                      'scikit-learn',
                      'tqdm',
                      'tensorflow>=1.14',
                      'imageio',
                      'pillow',
                      'matplotlib']
)
