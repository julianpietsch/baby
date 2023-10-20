# BABY

## Birth Annotation for Budding Yeast

An image processing pipeline for accurate single-cell growth estimation of
budding cells from bright-field stacks. The Birth Annotator for Budding Yeast
(BABY) features:

- segmentation of overlapping cells, e.g., where small buds overlap with their mothers/sisters,
- automated assignment of buds to mothers, and
- estimation of the time of birth (cytokinesis) from growth rate.

The algorithm is described in:

[Julian MJ Pietsch, Alán F Muñoz, Diane-Yayra A Adjavon, Iseabail Farquhar,
Ivan BN Clark, Peter S Swain. (2023). Determining growth rates from
bright-field images of budding cells through identifying overlaps. eLife.
12:e79812.](https://doi.org/10.7554/eLife.79812)


## Installation

We recommend installing BABY and all its requirements in a virtual environment
(e.g., `conda create` if you are using Anaconda, or `python3 -m venv` otherwise).

NB: If you want to install with GPU support see below. 

Otherwise, if you simply wish to use install with the latest compatible versions
of all packages, then simply run the following in your virtual environment:

```bash
> pip install baby-seg
```

NB: You can update by running: `pip install -U baby-seg`.

### Python and TensorFlow version

BABY requires Python 3 and [TensorFlow](https://www.tensorflow.org).
Different versions of TensorFlow have different Python version requirements.
You can find a table of matching versions
[here](https://www.tensorflow.org/install/source#tested_build_configurations).

Our models were trained with TensorFlow version 2.8, but have been tested up
to version 2.14.

By default, BABY will trigger installation of the highest compatible version
of TensorFlow. If you want to use an earlier version, then first install that
version in your virtual environment by running:

```bash
> pip install tensorflow==2.8
```

and then follow the instructions for installing BABY as above.

### Running with GPU

To make use of a GPU you should follow the [TensorFlow set up
instructions](https://www.tensorflow.org/install/gpu) before installing BABY.

BABY can make use of Metal on M1/M2 Macs by following the instructions
[here](https://developer.apple.com/metal/tensorflow-plugin/).


### For developers

You may prefer to install an editable version by first obtaining this repo and
then pip installing the cloned directory:

```bash
> git clone https://git.ecdf.ed.ac.uk/swain-lab/baby.git
> cd baby
> pip install -e .
```

## Quickstart using the Python API

The BABY algorithm makes use of several machine learning models that are
defined as a model set. Various model sets are available, and each has been
optimised for a particular species, microfluidics device, pixel size, channel
and number of input Z sections.

You can get a list of available model sets and the types of input they were
trained for using the `meta` function in the `modelsets` module:

```python
>>> from baby import modelsets
>>> modelsets.meta()
```

You then load your favourite model set as a `BabyBrain` object, which
coordinates all the models and parameters in the set to produce tracked and
segmented outlines from input images. You can get a `BabyBrain` for a given
model set using the `get` function in the `modelsets` module:

```python
>>> brain = modelsets.get('yeast-alcatras-brightfield-EMCCD-60x-5z')
```

For each time course you want to process, instantiate a new `BabyCrawler`. The
crawler keeps track of cells between time steps.

```python
>>> from baby import BabyCrawler
>>> crawler = BabyCrawler(brain)
```

Load an image time series (from the `tests` subdirectory in this repository).
The image should have shape (x, y, z).

```python
>>> from baby.io import load_tiled_image
>>> image_series = [load_tiled_image(
...     'tests/images/evolve_testG_tp{:d}_Brightfield.png'.format(t))
...     for t in range(1,6)]
```

Send images to the crawler in time-order (here a batch of size 1). We 
additionally request that outlines are optimised to edge predictions, and that 
lineage assignments, binary edge-masks and volume estimates (using the conical
method) should be output at each time point.

```python
>>> segmented_series = [crawler.step(
...     img[None, ...], refine_outlines=True, assign_mothers=True,
...     with_edgemasks=True, with_volumes=True)
...     for img, _ in image_series]
```

Finally, save the segmentation outlines, labels, volumes and lineage assignments
as an annotated tiled png:

```python
>>> from baby.io import save_tiled_image
>>> for t, s in enumerate(segmented_series): 
...     save_tiled_image(255 * s[0]['edgemasks'].astype('uint8').transpose((1, 2, 0)), 
...     '../segout_tp{:d}.png'.format(t + 1), 
...     {k: s[0][k] for k in ('cell_label', 'mother_assign', 'volumes')})
```

## Jupyter notebooks

More extensive examples for using BABY can be found in the `python/notebooks`
folder. To maintain the repository in a clean state, it's probably best to
copy these to another directory for routine use. If you want to share a
notebook, you can then specifically add it back to the repository at a useful
checkpoint.

The notebooks include a [complete
example](python/notebooks/example-baby-analysis-fig1.ipynb) for reproducing
Figure 1e in the associated paper.

## Run via a server

Once installed, you should be able to start a server to accept segmentation
requests using:

```bash
> baby-phone
```

Server runs by default on [http://0.0.0.0:5101](). HTTP requests need to be
sent to the correct URL endpoint, but the HTTP API is currently undocumented.
The primary client implementation is in Matlab.

