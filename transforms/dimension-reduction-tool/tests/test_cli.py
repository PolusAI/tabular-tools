"""Tests for the CLI."""

import copy
import pathlib
import tempfile

from polus.tabular.transforms.dimension_reduction.algorithms import Algorithm

import numpy
import pytest
import sklearn.datasets
import typer.testing
from polus.tabular.transforms.dimension_reduction.data_io import Formats
from polus.tabular.transforms.dimension_reduction.__main__ import app


ALGORITHMS = [Algorithm.TSNE_INIT_PCA, Algorithm.UMAP]
FORMATS = ["csv", "feather"]


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


def gen_pca_args(
    svd_solver: list[str] = ["auto", "arpack"],
    tol: list[float] = [0.0, 0.1, 0.5, 1.0],
) -> list[dict]:
    """Generate arguments for the PCA algorithm."""
    all_kwargs = []
    for s in svd_solver:
        if s == "arpack":
            for t in tol:
                all_kwargs.append(
                    {
                        "pcaSvdSolver": s,
                        "pcaTol": t,
                    }
                )
        else:
            all_kwargs.append(
                {
                    "pcaSvdSolver": s,
                    "pcaTol": 0.0,
                }
            )
    return all_kwargs


def gen_tsne_args(
    perplexity: list[float] = [5.0, 50.0],
    early_exaggeration: list[float] = [4.0, 24.0],
    learning_rate: list[float] = [100.0, 200.0],
    max_iter: list[int] = [250, 1000],
    metric: list[str] = ["euclidean", "cosine"],
) -> list[dict]:
    """Generate arguments for the t-SNE algorithm."""
    all_kwargs = []
    for p in perplexity:
        for e in early_exaggeration:
            for l in learning_rate:
                for m in max_iter:
                    for me in metric:
                        all_kwargs.append(
                            {
                                "tsnePerplexity": p,
                                "tsneEarlyExaggeration": e,
                                "tsneLearningRate": l,
                                "tsneMaxIter": m,
                                "tsneMetric": me,
                            }
                        )
    return all_kwargs


def gen_tsne_pca_args(
    perplexity: list[float] = [5.0, 50.0],
    early_exaggeration: list[float] = [4.0, 24.0],
    learning_rate: list[float] = [100.0, 200.0],
    max_iter: list[int] = [250, 1000],
    metric: list[str] = ["euclidean", "cosine"],
    init_n_components: list[int] = [10, 50],
) -> list[dict]:
    """Generate arguments for the t-SNE algorithm with PCA initialization."""
    tsne_kwargs = gen_tsne_args(
        perplexity, early_exaggeration, learning_rate, max_iter, metric
    )
    all_kwargs = []
    for inp_kwargs in tsne_kwargs:
        for n in init_n_components:
            kwargs = copy.deepcopy(inp_kwargs)
            kwargs["tsneInitNComponents"] = n
            all_kwargs.append(kwargs)
    return all_kwargs


def gen_umap_args(
    n_neighbors: list[int] = [5, 15, 50],
    n_epochs: list[int] = [200, 500],
    min_dist: list[float] = [0.1, 0.5],
    spread: list[float] = [1.0, 2.0],
    metric: list[str] = ["euclidean", "cosine"],
) -> list[dict]:
    """Generate arguments for the UMAP algorithm."""
    all_kwargs = []
    for n in n_neighbors:
        for e in n_epochs:
            for m in min_dist:
                for s in spread:
                    for me in metric:
                        all_kwargs.append(
                            {
                                "umapNNeighbors": n,
                                "umapNEpochs": e,
                                "umapMinDist": m,
                                "umapSpread": s,
                                "umapMetric": me,
                            }
                        )
    return all_kwargs


@pytest.mark.parametrize("inp_format", FORMATS)
@pytest.mark.parametrize("algorithm", ALGORITHMS)
@pytest.mark.parametrize("n_components", [3])
def test_cli(
    inp_format: str,
    algorithm: Algorithm,
    n_components: int,
) -> None:
    """Test the CLI."""

    inp_dir, out_dir = create_data(inp_format)

    base_kwargs = {
        "inpDir": str(inp_dir),
        "outDir": str(out_dir),
        "nComponents": str(n_components),
        "algorithm": algorithm.value,
    }
    all_kwargs: list[dict] = []
    if algorithm == Algorithm.PCA:
        all_kwargs = gen_pca_args()
    elif algorithm == Algorithm.TSNE:
        all_kwargs = gen_tsne_args()
    elif algorithm == Algorithm.TSNE_INIT_PCA:
        all_kwargs = gen_tsne_pca_args()
    elif algorithm == Algorithm.UMAP:
        all_kwargs = gen_umap_args()
    else:
        raise ValueError(f"Unknown algorithm {algorithm}")

    for inp_kwargs in all_kwargs:
        kwargs = copy.deepcopy(base_kwargs)
        kwargs.update(inp_kwargs)

        args = []
        for k, v in kwargs.items():
            args.extend(["--" + k, str(v)])

        runner = typer.testing.CliRunner()
        result = runner.invoke(app, args)

        assert result.exit_code == 0, f"CLI failed with {result.stdout}\n{args}"

        inp_dir = pathlib.Path(kwargs["inpDir"])
        out_dir = pathlib.Path(kwargs["outDir"])
        inp_files: list[pathlib.Path] = [p for p in inp_dir.iterdir()]
        out_files: list[pathlib.Path] = [p for p in out_dir.iterdir()]

        assert len(inp_files) == 1
        assert len(out_files) == 1

        for inp_path in inp_files:
            out_path = out_dir.joinpath(inp_path.stem + ".feather")
            msg = f"Missing {inp_path.stem} from {inp_files} in {out_files}\n{args}"
            assert out_path in out_files, msg

            data = Formats.read(out_path)
            assert data.shape == (1797, n_components)
            assert data.dtype == numpy.float32

        for out_path in out_files:
            out_path.unlink()
