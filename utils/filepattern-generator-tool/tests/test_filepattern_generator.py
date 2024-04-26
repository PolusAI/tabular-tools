"""Filepattern Generator Tool"""
from pathlib import Path
from .conftest import clean_directories
import pytest
import json
import polus.tabular.utils.filepattern_generator.filepattern_generator as fg


def test_generate_patterns(
    output_directory: Path, get_params: pytest.FixtureRequest, create_data: Path
) -> None:
    """Test generate filepatterns of image files."""

    pattern, group_by, chunk_size, number = get_params
    fg.generate_patterns(create_data, output_directory, pattern, chunk_size, group_by)

    files = []
    with open(output_directory.joinpath("file_patterns.json"), "r") as read_file:
        data = json.load(read_file)
        file_pattern = data["filePatterns"]
        files.append(file_pattern)

    assert len(files[0]) == number
    clean_directories()
