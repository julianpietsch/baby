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

## Running the server

Once installed, you should be able to start a server to accept segmentation
requests using:

```bash
> baby-phone
```

or on windows:

```
> baby-phone.exe
```

## Jupyter notebooks

Training scripts are saved in Jupyter notebooks in the `notebooks` folder. To
maintain the repository in a clean state, it's probably best to copy these to
another directory for routine use. If you want to share a notebook, you can
then specifically add it back to the repository at a useful checkpoint.
