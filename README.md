# Baby

## Birth Annotation for Budding Yeast

Neural network code for segmenting buds from brightfield stacks.

## Set up

**NB:** To make use of a GPU, first follow the [set up
instructions](https://www.tensorflow.org/install/gpu#windows_setup) for
installing `tensorflow-gpu`.

**Requirements:**

- Python 3.6 (not 3.7)
- tensorflow == 1.14
- numpy
- scipy
- scikit-image
- scikit-learn

If you install using Anaconda, then you will probably need to `pip install`
the `tensorflow` package to get the correct version.

It is recommended that you install the packages into a virtual environment
(using either `conda create` or `python3 -m venv`).

Once installed, you can add the python folder to your python path by creating
a `.pth` file in the environment's `site-packages` folder. E.g., create a file
`baby.pth` that contains a single line specifying the path to the python
folder of this repository `C:\path\to\repository\baby\python`.

## Jupyter notebooks

Training scripts are saved in Jupyter notebooks in the `notebooks` folder. To
maintain the repository in a clean state, it's probably best to copy these to
another directory for routine use. If you want to share a notebook, you can
then specifically add it back to the repository at a useful checkpoint.
