"""Tests for the tools."""

import copy
import pytest
import numpy
import sklearn.datasets

from polus.tabular.transforms.dimension_reduction import algorithms


def gen_pca_args(
    n_components: list[int] = [2, 3, 10],
    whiten: list[bool] = [True, False],
    svd_solver: list[algorithms.SvdSolver] = [
        algorithms.SvdSolver.AUTO,
        algorithms.SvdSolver.FULL,
        algorithms.SvdSolver.ARPACK,
        algorithms.SvdSolver.RANDOMIZED,
    ],
    tol: list[float] = [0.0, 0.1, 0.5, 1.0],
) -> list[dict]:
    """Generate arguments for the PCA algorithm."""
    all_kwargs = []
    for n in n_components:
        for w in whiten:
            for s in svd_solver:
                if s == algorithms.SvdSolver.ARPACK:
                    for t in tol:
                        all_kwargs.append(
                            {
                                "n_components": n,
                                "whiten": w,
                                "svd_solver": s,
                                "tol": t,
                            }
                        )
                else:
                    all_kwargs.append(
                        {
                            "n_components": n,
                            "whiten": w,
                            "svd_solver": s,
                            "tol": 0.0,
                        }
                    )
    return all_kwargs


@pytest.mark.parametrize("kwargs", gen_pca_args())
def test_pca(kwargs: dict):
    """Test the PCA algorithm."""

    digits = sklearn.datasets.load_digits()
    data: numpy.ndarray = digits.data

    assert data.shape == (1797, 64)

    reduced = algorithms.pca.reduce(data.astype(numpy.float32), **kwargs)

    assert reduced.shape[0] == 1797
    assert reduced.shape[1] == kwargs["n_components"]
    assert reduced.dtype == numpy.float32


def gen_tsne_args(
    n_components: list[int] = [2, 3],
    perplexity: list[float] = [5.0, 50.0],
    early_exaggeration: list[float] = [5.0, 20.0],
    learning_rate: list[float] = [200.0, "auto"],
    max_iter: list[int] = [250, 1000],
    metric: list[str] = ["euclidean", "cosine"],
) -> list[dict]:
    """Generate arguments for testing the t-SNE algorithm."""
    all_kwargs = []
    for n in n_components:
        for p in perplexity:
            for e in early_exaggeration:
                for l in learning_rate:
                    for m in max_iter:
                        for me in metric:
                            all_kwargs.append(
                                {
                                    "n_components": n,
                                    "perplexity": p,
                                    "early_exaggeration": e,
                                    "learning_rate": l,
                                    "max_iter": m,
                                    "metric": me,
                                }
                            )
    return all_kwargs


@pytest.mark.parametrize("kwargs", gen_tsne_args())
def test_tsne(kwargs: dict):
    """Test the t-SNE algorithm."""

    digits = sklearn.datasets.load_digits()
    data: numpy.ndarray = digits.data

    assert data.shape == (1797, 64)

    reduced = algorithms.tsne.reduce(data.astype(numpy.float32), **kwargs)

    assert reduced.shape[0] == 1797
    assert reduced.shape[1] == kwargs["n_components"]
    assert reduced.dtype == numpy.float32


def gen_tsne_pca_args(
    n_components: list[int] = [2, 3],
    pca_n_components: list[int] = [10, 50],
    perplexity: list[float] = [10.0, 30.0, 50.0],
) -> list[dict]:
    """Generate arguments for testing the t-SNE algorithm with PCA initialization."""
    base_kwargs = {
        "pca_n_components": 2,
        "pca_whiten": False,
        "pca_svd_solver": algorithms.SvdSolver.AUTO,
        "pca_tol": 0.0,
        "early_exaggeration": 12.0,
        "learning_rate": "auto",
        "max_iter": 1000,
        "metric": "euclidean",
    }
    all_kwargs = []
    for n in n_components:
        for p in pca_n_components:
            for pe in perplexity:
                kwargs = copy.deepcopy(base_kwargs)
                kwargs["n_components"] = n
                kwargs["pca_n_components"] = p
                kwargs["perplexity"] = pe
                all_kwargs.append(kwargs)
    return all_kwargs


@pytest.mark.parametrize("kwargs", gen_tsne_pca_args())
def test_tsne_pca(kwargs: dict):
    """Test the t-SNE algorithm with PCA initialization."""

    digits = sklearn.datasets.load_digits()
    data: numpy.ndarray = digits.data

    assert data.shape == (1797, 64)

    reduced = algorithms.tsne.reduce_init_pca(data.astype(numpy.float32), **kwargs)

    assert reduced.shape[0] == 1797
    assert reduced.shape[1] == kwargs["n_components"]
    assert reduced.dtype == numpy.float32


def gen_umap_args(
    n_components: list[int] = [2, 3, 10],
    n_neighbors: list[int] = [5, 15, 50],
    metric: list[str] = ["euclidean", "cosine"],
    n_epochs: list[int] = [None, 100],
    min_dist: list[float] = [0.05, 0.1, 0.2],
    spread: list[float] = [1.0, 2.0],
) -> list[dict]:
    """Generate arguments for the UMAP algorithm."""
    all_kwargs = []
    for n in n_components:
        for nn in n_neighbors:
            for m in metric:
                for ne in n_epochs:
                    for md in min_dist:
                        for s in spread:
                            all_kwargs.append(
                                {
                                    "n_components": n,
                                    "n_neighbors": nn,
                                    "metric": m,
                                    "n_epochs": ne,
                                    "min_dist": md,
                                    "spread": s,
                                }
                            )
    return all_kwargs


@pytest.mark.parametrize("kwargs", gen_umap_args())
def test_umap(kwargs: dict):
    """Test the UMAP algorithm."""

    digits = sklearn.datasets.load_digits()
    data: numpy.ndarray = digits.data

    assert data.shape == (1797, 64)

    reduced = algorithms.umap.reduce(data.astype(numpy.float32), **kwargs)

    assert reduced.shape[0] == 1797
    assert reduced.shape[1] == kwargs["n_components"]
    assert reduced.dtype == numpy.float32
