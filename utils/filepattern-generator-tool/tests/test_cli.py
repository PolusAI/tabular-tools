"""Test Command line Tool."""

from typer.testing import CliRunner
from pathlib import Path
import pytest
from polus.tabular.utils.filepattern_generator.__main__ import app
from .conftest import clean_directories
import time


def test_cli(
    output_directory: Path, get_params: pytest.FixtureRequest, create_data: Path
) -> None:
    """Test the command line."""
    runner = CliRunner()
    pattern, group_by, chunk_size, _ = get_params
    result = runner.invoke(
        app,
        [
            "--inpDir",
            create_data,
            "--filePattern",
            pattern,
            "--chunkSize",
            chunk_size,
            "--groupBy",
            group_by,
            "--outDir",
            output_directory,
        ],
    )

    assert result.exit_code == 0
    time.sleep(5)
    clean_directories()

    def test_short_cli(
        output_directory: Path, get_params: pytest.FixtureRequest, create_data: Path
    ) -> None:
        """Test the short command line."""
        runner = CliRunner()
        pattern, group_by, chunk_size, _ = get_params
        result = runner.invoke(
            app,
            [
                "--i",
                create_data,
                "--f",
                pattern,
                "-c",
                chunk_size,
                "-g",
                group_by,
                "--o",
                output_directory,
            ],
        )

        assert result.exit_code == 0
        time.sleep(5)
        clean_directories()
