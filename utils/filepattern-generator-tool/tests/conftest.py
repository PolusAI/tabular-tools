"""Test fixtures.

Set up all data used in tests.
"""

import shutil
import tempfile
from itertools import product
from pathlib import Path
from typing import Union

import pytest


def pytest_addoption(parser: pytest.Parser) -> None:
    """Add options to pytest."""
    parser.addoption(
        "--slow",
        action="store_true",
        dest="slow",
        default=False,
        help="run slow tests",
    )


def clean_directories() -> None:
    """Remove all temporary directories."""
    for d in Path(".").cwd().iterdir():
        if d.is_dir() and d.name.startswith("tmp"):
            shutil.rmtree(d)


@pytest.fixture()
def output_directory() -> Union[str, Path]:
    """Create output directory."""
    return Path(tempfile.mkdtemp(dir=Path.cwd()))


@pytest.fixture()
def input_directory() -> Union[str, Path]:
    """Create input directory."""
    return Path(tempfile.mkdtemp(dir=Path.cwd()))


@pytest.fixture(
    params=[
        ("x{x:d+}_y{y:d+}_p{p:d+}_c1.ome.tif", "x", 0, 4),
        ("x{x:d+}_y{y:d+}_p{p:d+}_c1.ome.tif", "y", 0, 8),
        ("x{x:d+}_y{y:d+}_p{p:d+}_c1.ome.tif", "p", 0, 2),
    ],
)
def get_params(request: pytest.FixtureRequest) -> pytest.FixtureRequest:
    """To get the parameter of the fixture."""
    return request.param


@pytest.fixture()
def create_data(input_directory: Path) -> Union[str, Path]:
    """Generate image files."""
    for x, y, p in product(range(4), range(8), range(2)):
        fname = (
            f"x0{x}".zfill(2) + f"_y0{y}".zfill(2) + f"_p0{p}".zfill(2) + "_c1.ome.tif"
        )
        file_name = Path(input_directory).joinpath(fname)
        with Path.open(file_name, "w"):
            pass

    return Path(input_directory)
