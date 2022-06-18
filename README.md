# BABY

## Birth Annotation for Budding Yeast

An image processing pipeline for accurate single-cell growth estimation of
budding cells from bright-field stacks. The Birth Annotator for Budding Yeast
(BABY) features:

- segmentation of overlapping cells, e.g., where small buds overlap with their mothers/sisters,
- automated assignment of buds to mothers, and
- estimation of the time of birth (cytokinesis) from growth rate.

The algorithm is described in:

Julian M J Pietsch, Alán F Muñoz, Diane-Yayra A Adjavon, Ivan B N Clark, Peter
S Swain, 2022, A label-free method to track individuals and lineages of
budding cells (in submission).

## Installation

BABY can be used with Python versions 3.6-3.8 (see below for details). If you
wish to use the latest compatible versions of all packages, BABY can simply be
installed by first obtaining this repository (e.g., `git clone
https://git.ecdf.ed.ac.uk/jpietsch/baby.git`), and then running pip on the 
repository directory:

```bash
> cd baby
> pip install .
```

NB: The '.' is important!

If you pull new changes, you need to update by running: `pip install -U .` from
within the repository directory.

*Developers:* You may prefer to install an editable version:

```bash
> pip install -e .
```

This avoids the need to run the update command.

**Requirements for Python and TensorFlow**

BABY requires Python 3 and [TensorFlow](https://www.tensorflow.org). The
models were trained in TensorFlow 1.14.0, but are compatible with versions of
TensorFlow up to 2.3.4. The required version of Python depends on the version
of TensorFlow you choose. We recommend either:

- Python 3.6 and TensorFlow 1.14,
- Python 3.7 and TensorFlow 1.15, or
- Python 3.8 and TensorFlow 2.3.  

In any case, it is recommended that you install TensorFlow and all other
required packages into a virtual environment (i.e., `conda create` if you are
using Anaconda, or `python3 -m venv` otherwise).

By default, BABY will trigger installation of the highest compatible version of
TensorFlow. If you want to use an earlier version as suggested above, then
first install that version in your virtual environment by running:

```bash
> pip install tensorflow==1.14
```

and then follow the instructions for installing BABY as above.

**NB:** To make use of a GPU you should also follow the other [set up
instructions](https://www.tensorflow.org/install/gpu).

**NB:** For `tensorflow==1.14`, you will also need to downgrade the default
version of `h5py`: 

```bash
> pip uninstall h5py
> pip install h5py==2.9.0
```

## Run using the Python API

Create a new `BabyBrain` with one of the model sets. The `brain` contains
all the models and parameters for segmenting and tracking cells.

```python
>>> from baby import BabyBrain, BabyCrawler, modelsets
>>> modelset = modelsets()['evolve_brightfield_60x_5z']
>>> brain = BabyBrain(**modelset)
```

For each time course you want to process, instantiate a new `BabyCrawler`. The
crawler keeps track of cells between time steps.

```python
>>> crawler = BabyCrawler(brain)
```

Load an image time series (from the `tests` subdirectory in this example). The
image should have shape (x, y, z).

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

or on windows:

```
> baby-phone.exe
```

Server runs by default on [http://0.0.0.0:5101](). HTTP requests need to be
sent to the correct URL endpoint, but the HTTP API is currently undocumented.
The primary client implementation is in Matlab.

