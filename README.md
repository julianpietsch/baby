# Baby

## Birth Annotation for Budding Yeast

Neural network code for segmenting buds from brightfield stacks.

## Installation

BABY requires Python 3 and [TensorFlow](https://www.tensorflow.org). For some
versions of TensorFlow, you specifically need Python 3.6.

In any case, it is recommended that you install the package into a virtual
environment (i.e., `conda create` if you are using Anaconda, or `python3 -m
venv` otherwise).

By default, BABY will trigger installation of the latest version of
TensorFlow. Our experience, however, is that performance is best with
TensorFlow version 1.14. If you want to use this version, first install that
in your virtual environment by running:

```bash
> pip install tensorflow==1.14
```

**NB:** To make use of a GPU you should also follow the [set up
instructions](https://www.tensorflow.org/install/gpu#windows_setup) for
installing `tensorflow-gpu`.

Install BABY by first obtaining this repository (e.g., `git clone
https://git.ecdf.ed.ac.uk/jpietsch/baby.git`), and then using pip:

```bash
> pip install baby/
```

NB: If you are upgrading, then you may instead need to run: `pip install -U
baby/`.

*Developers:* You may prefer to install an editable version:

```bash
> pip install -e baby/
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

## Jupyter notebooks

Training scripts are saved in Jupyter notebooks in the `notebooks` folder. To
maintain the repository in a clean state, it's probably best to copy these to
another directory for routine use. If you want to share a notebook, you can
then specifically add it back to the repository at a useful checkpoint.
