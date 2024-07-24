"""Dimension Reduction via various methods."""

import logging
import os
import pathlib

import numpy

from .algorithms import Algorithm
from .algorithms import pca
from .algorithms import tsne
from .algorithms import umap
from .algorithms.pca import SvdSolver
from .data_io import Formats

POLUS_LOG_LVL = os.environ.get("POLUS_LOG", logging.INFO)
POLUS_TAB_EXT = os.environ.get("POLUS_TAB_EXT", ".feather")

__version__ = "0.1.0-dev1"


def reduce(
    inp_path: pathlib.Path,
    out_path: pathlib.Path,
    algorithm: Algorithm,
    kwargs: dict,
) -> None:
    """Reduce the dimensionality of the data using the specified algorithm.

    The allowed formats for the input and output data are CSV, Parquet, Feather,
    and NPY.

    The allowed algorithms are PCA, t-SNE, t-SNE with PCA initialization, and UMAP.

    Args:
        inp_path: The path to the input data.
        out_path: The path to write the reduced data.
        algorithm: The algorithm to use for dimensionality reduction.
        kwargs: Additional keyword arguments for the algorithm.
    """
    data = Formats.read(inp_path)
    reduced_data: numpy.ndarray

    if algorithm == Algorithm.PCA:
        reduced_data = pca.reduce(data, **kwargs)
    elif algorithm == Algorithm.TSNE:
        reduced_data = tsne.reduce(data, **kwargs)
    elif algorithm == Algorithm.TSNE_INIT_PCA:
        reduced_data = tsne.reduce_init_pca(data, **kwargs)
    elif algorithm == Algorithm.UMAP:
        reduced_data = umap.reduce(data, **kwargs)
    else:
        allowed_algorithms = ", ".join(Algorithm.__members__.keys())
        msg = (
            f"Unsupported algorithm: {algorithm}. Must be one of: {allowed_algorithms}"
        )
        raise ValueError(msg)

    Formats.write(reduced_data, out_path)


__all__ = [
    "pca",
    "tsne",
    "umap",
    "POLUS_LOG_LVL",
    "POLUS_TAB_EXT",
    "__version__",
    "SvdSolver",
    "Algorithm",
    "Formats",
    "reduce",
]
