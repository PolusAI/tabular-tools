"""Test Command line Tool."""

from typer.testing import CliRunner
from pathlib import Path
import pytest
from polus.tabular.clustering.pytorch_tabnet.__main__ import app
from .conftest import clean_directories
from typing import Union


def test_cli(
    output_directory: Path,
    create_dataset: Union[str, Path],
    get_params: pytest.FixtureRequest,
) -> None:
    """Test the command line."""

    inp_dir = create_dataset

    runner = CliRunner()

    test_size, optimizer_fn, scheduler_fn, eval_metric, loss_fn, classifier = get_params

    result = runner.invoke(
        app,
        [
            "--inpDir",
            inp_dir,
            "--filePattern",
            ".*.csv",
            "--testSize",
            test_size,
            "--optimizerFn",
            optimizer_fn,
            "--evalMetric",
            eval_metric,
            "--schedulerFn",
            scheduler_fn,
            "--lossFn",
            loss_fn,
            "--targetVar",
            "income",
            "--classifier",
            classifier,
            "--outDir",
            output_directory,
        ],
    )
    assert result.exit_code == 0
    clean_directories()
