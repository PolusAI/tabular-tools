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


@pytest.mark.skipif("not config.getoption('slow')")
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
    if all(
        (
            n_components == 2,
            whiten is False,
            svd_solver == algorithms.SvdSolver.AUTO,
            tol == 0.0,
        )
    ):
        # This test has been handled in `test_fast.py`
        return

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


@pytest.mark.skipif("not config.getoption('slow')")
@pytest.mark.parametrize("n_components", [2, 3, 10])
@pytest.mark.parametrize("perplexity", [5.0, 30.0, 50.0])
@pytest.mark.parametrize("early_exaggeration", [5.0, 12.0, 20.0])
@pytest.mark.parametrize("learning_rate", [50.0, 100.0, 200.0, 500.0, 1000.0, "auto"])
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
    if all(
        (
            n_components == 2,
            perplexity == 30.0,
            early_exaggeration == 12.0,
            learning_rate == "auto",
            max_iter == 250,
            metric == "euclidean",
        )
    ):
        # This test has been handled in `test_fast.py`
        return

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


@pytest.mark.skipif("not config.getoption('slow')")
@pytest.mark.parametrize("pca_n_components", [10, 50])
@pytest.mark.parametrize("whiten", [False, True])
@pytest.mark.parametrize("svd_solver", SVD_SOLVERS)
@pytest.mark.parametrize("tol", [0.0, 0.5])
@pytest.mark.parametrize("n_components", [2, 3])
@pytest.mark.parametrize("perplexity", [5.0, 30.0, 50.0])
@pytest.mark.parametrize("early_exaggeration", [5.0, 12.0, 20.0])
@pytest.mark.parametrize("learning_rate", [50.0, 100.0, 200.0, 500.0, 1000.0, "auto"])
@pytest.mark.parametrize("max_iter", [250, 1000])
@pytest.mark.parametrize("metric", ["euclidean", "cosine"])
def test_tsne_init_pca(
    pca_n_components: int,
    whiten: bool,
    svd_solver: algorithms.SvdSolver,
    tol: float,
    n_components: int,
    perplexity: float,
    early_exaggeration: float,
    learning_rate: float,
    max_iter: int,
    metric: str,
):
    """Test the t-SNE algorithm with PCA initialization."""
    if all(
        (
            pca_n_components == 10,
            n_components == 2,
            perplexity == 30.0,
            early_exaggeration == 12.0,
            learning_rate == "auto",
            max_iter == 250,
            metric == "euclidean",
        )
    ):
        # This test has been handled in `test_fast.py`
        return

    digits = sklearn.datasets.load_digits()
    data: numpy.ndarray = digits.data

    assert data.shape == (1797, 64)

    reduced = algorithms.tsne.reduce_init_pca(
        data.astype(numpy.float32),
        pca_n_components=pca_n_components,
        pca_whiten=whiten,
        pca_svd_solver=svd_solver,
        pca_tol=tol,
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


@pytest.mark.skipif("not config.getoption('slow')")
@pytest.mark.parametrize("n_components", [2, 3, 10])
@pytest.mark.parametrize("n_neighbors", [5, 15, 50])
@pytest.mark.parametrize("metric", ["euclidean", "cosine"])
@pytest.mark.parametrize("n_epochs", [None, 200, 500])
@pytest.mark.parametrize("min_dist", [0.05, 0.1, 0.2])
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
    if all(
        (
            n_components == 2,
            n_neighbors == 15,
            metric == "euclidean",
            n_epochs == 200,
            min_dist == 0.1,
            spread == 1.0,
        )
    ):
        # This test has been handled in `test_fast.py`
        return

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
