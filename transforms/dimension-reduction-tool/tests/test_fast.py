"""Fast tests for github actions."""

import copy
import pathlib
import tempfile

import numpy
import pytest
import sklearn.datasets
import typer.testing
from polus.tabular.transforms.dimension_reduction.algorithms import Algorithm
from polus.tabular.transforms.dimension_reduction import algorithms
from polus.tabular.transforms.dimension_reduction.data_io import Formats
from polus.tabular.transforms.dimension_reduction.__main__ import app


FORMATS = ["csv", "feather", "parquet", "npy"]


def create_data(inp_format: str) -> tuple[pathlib.Path, pathlib.Path]:
    """Generate data."""

    data_dir = pathlib.Path(tempfile.mkdtemp(suffix="_data_dir"))

    inp_dir = data_dir.joinpath("inp_dir")
    inp_dir.mkdir()

    out_dir = data_dir.joinpath("out_dir")
    out_dir.mkdir()

    digits = sklearn.datasets.load_digits()
    data: numpy.ndarray = digits.data
    data = data.astype(numpy.float32)
    Formats.write(data, inp_dir.joinpath(f"digits.{inp_format}"))

    return inp_dir, out_dir


@pytest.mark.parametrize("inp_format", FORMATS)
@pytest.mark.parametrize("out_format", FORMATS)
def test_data_io(inp_format: str, out_format: str) -> None:
    """Test data IO."""

    inp_dir, out_dir = create_data(inp_format)
    assert inp_dir.exists()
    assert out_dir.exists()

    inp_files: list[pathlib.Path] = list(inp_dir.iterdir())

    assert len(inp_files) == 1
    assert inp_files[0].name == "digits." + inp_format

    out_path = out_dir.joinpath(inp_files[0].stem + f".{out_format}")
    inp_data = Formats.read(inp_dir.joinpath(inp_files[0]))
    Formats.write(inp_data, out_path)

    out_files: list[pathlib.Path] = list(out_dir.iterdir())
    assert len(out_files) == 1
    assert out_files[0].name == "digits." + out_format

    out_data = Formats.read(out_path)

    assert inp_data.shape == out_data.shape
    assert inp_data.dtype == out_data.dtype
    numpy.testing.assert_allclose(inp_data, out_data)


@pytest.mark.parametrize("n_components", [2])
@pytest.mark.parametrize("whiten", [False])
@pytest.mark.parametrize("svd_solver", [algorithms.SvdSolver.AUTO])
@pytest.mark.parametrize("tol", [0.0])
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


@pytest.mark.parametrize("n_components", [2])
@pytest.mark.parametrize("perplexity", [30.0])
@pytest.mark.parametrize("early_exaggeration", [12.0])
@pytest.mark.parametrize("learning_rate", ["auto"])
@pytest.mark.parametrize("max_iter", [250])
@pytest.mark.parametrize("metric", ["euclidean"])
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


@pytest.mark.parametrize("pca_n_components", [10])
@pytest.mark.parametrize("whiten", [False])
@pytest.mark.parametrize("svd_solver", [algorithms.SvdSolver.AUTO])
@pytest.mark.parametrize("tol", [0.0])
@pytest.mark.parametrize("n_components", [2])
@pytest.mark.parametrize("perplexity", [30.0])
@pytest.mark.parametrize("early_exaggeration", [12.0])
@pytest.mark.parametrize("learning_rate", ["auto"])
@pytest.mark.parametrize("max_iter", [250])
@pytest.mark.parametrize("metric", ["euclidean"])
def test_tsne_pca(
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


@pytest.mark.parametrize("n_components", [2])
@pytest.mark.parametrize("n_neighbors", [15])
@pytest.mark.parametrize("metric", ["euclidean"])
@pytest.mark.parametrize("n_epochs", [200])
@pytest.mark.parametrize("min_dist", [0.1])
@pytest.mark.parametrize("spread", [1.0])
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


def test_cli():
    inp_dir, out_dir = create_data("csv")

    args = [
        "--inpDir",
        str(inp_dir),
        "--nComponents",
        "3",
        "--algorithm",
        "umap",
        "--umapNNeighbors",
        "15",
        "--umapNEpochs",
        "200",
        "--umapMinDist",
        "0.1",
        "--umapSpread",
        "1.0",
        "--umapMetric",
        "euclidean",
        "--outDir",
        str(out_dir),
    ]

    runner = typer.testing.CliRunner()
    result = runner.invoke(app, args)

    assert result.exit_code == 0

    inp_files = list(map(pathlib.Path, inp_dir.iterdir()))
    out_files = list(map(pathlib.Path, out_dir.iterdir()))

    assert len(inp_files) == 1
    assert len(out_files) == 1

    for inp_path in inp_files:
        out_path = out_dir.joinpath(inp_path.stem + ".feather")
        msg = f"Missing {inp_path.stem} from {inp_files} in {out_files}\n{args}"
        assert out_path in out_files, msg

        data = Formats.read(out_path)
        assert data.shape == (1797, 3)
        assert data.dtype == numpy.float32
