"""Tests for the tools."""

import pytest
import numpy
import sklearn.datasets

from polus.tabular.transforms.dimension_reduction import algorithms

SVD_SOLVERS = [
    algorithms.SvdSolver.AUTO,
    algorithms.SvdSolver.FULL,
    algorithms.SvdSolver.ARPACK,
    algorithms.SvdSolver.RANDOMIZED,
]


@pytest.mark.parametrize("n_components", [2, 10])
@pytest.mark.parametrize("whiten", [True, False])
@pytest.mark.parametrize("svd_solver", SVD_SOLVERS)
@pytest.mark.parametrize("tol", [0.0, 0.5])
def test_pca(
    n_components: int,
    whiten: bool,
    svd_solver: algorithms.SvdSolver,
    tol: float,
):
    """Test the PCA algorithm."""

    digits = sklearn.datasets.load_digits()
    data: numpy.ndarray = digits.data

    assert data.shape == (1797, 64)

    reduced = algorithms.pca.reduce(
        data.astype(numpy.float32),
        n_components=n_components,
        whiten=whiten,
        svd_solver=svd_solver,
        tol=tol,
    )

    assert reduced.ndim == data.ndim
    assert reduced.shape[0] == data.shape[0]
    assert reduced.shape[1] == n_components
    assert reduced.dtype == numpy.float32


@pytest.mark.parametrize("n_components", [3])
@pytest.mark.parametrize("perplexity", [5.0, 50.0])
@pytest.mark.parametrize("early_exaggeration", [5.0, 20.0])
@pytest.mark.parametrize("learning_rate", [200.0, "auto"])
@pytest.mark.parametrize("max_iter", [250, 1000])
@pytest.mark.parametrize("metric", ["euclidean", "cosine"])
def test_tsne(
    n_components: int,
    perplexity: float,
    early_exaggeration: float,
    learning_rate: float,
    max_iter: int,
    metric: str,
):
    """Test the t-SNE algorithm."""

    digits = sklearn.datasets.load_digits()
    data: numpy.ndarray = digits.data

    assert data.shape == (1797, 64)

    reduced = algorithms.tsne.reduce(
        data.astype(numpy.float32),
        n_components=n_components,
        perplexity=perplexity,
        early_exaggeration=early_exaggeration,
        learning_rate=learning_rate,
        max_iter=max_iter,
        metric=metric,
    )

    assert reduced.ndim == data.ndim
    assert reduced.shape[0] == 1797
    assert reduced.shape[1] == n_components
    assert reduced.dtype == numpy.float32


@pytest.mark.parametrize("n_components", [2, 3])
@pytest.mark.parametrize("pca_n_components", [10, 50])
@pytest.mark.parametrize("perplexity", [5.0, 50.0])
def test_tsne_pca(
    n_components: int,
    pca_n_components: int,
    perplexity: float,
):
    """Test the t-SNE algorithm with PCA initialization."""

    digits = sklearn.datasets.load_digits()
    data: numpy.ndarray = digits.data

    assert data.shape == (1797, 64)

    reduced = algorithms.tsne.reduce_init_pca(
        data.astype(numpy.float32),
        pca_n_components=pca_n_components,
        pca_whiten=False,
        pca_svd_solver=algorithms.SvdSolver.AUTO,
        pca_tol=0.0,
        n_components=n_components,
        perplexity=perplexity,
        early_exaggeration=12.0,
        learning_rate="auto",
        max_iter=1000,
        metric="euclidean",
    )

    assert reduced.ndim == data.ndim
    assert reduced.shape[0] == 1797
    assert reduced.shape[1] == n_components
    assert reduced.dtype == numpy.float32


@pytest.mark.parametrize("n_components", [3, 10])
@pytest.mark.parametrize("n_neighbors", [10, 25])
@pytest.mark.parametrize("metric", ["euclidean", "cosine"])
@pytest.mark.parametrize("n_epochs", [None, 100])
@pytest.mark.parametrize("min_dist", [0.05, 0.2])
@pytest.mark.parametrize("spread", [1.0, 2.0])
def test_umap(
    n_components: int,
    n_neighbors: int,
    metric: str,
    n_epochs: int,
    min_dist: float,
    spread: float,
):
    """Test the UMAP algorithm."""

    digits = sklearn.datasets.load_digits()
    data: numpy.ndarray = digits.data

    assert data.shape == (1797, 64)

    reduced = algorithms.umap.reduce(
        data.astype(numpy.float32),
        n_components=n_components,
        n_neighbors=n_neighbors,
        metric=metric,
        n_epochs=n_epochs,
        min_dist=min_dist,
        spread=spread,
    )

    assert reduced.ndim == data.ndim
    assert reduced.shape[0] == 1797
    assert reduced.shape[1] == n_components
    assert reduced.dtype == numpy.float32
