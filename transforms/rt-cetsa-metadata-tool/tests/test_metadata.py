"""Tests."""
from pathlib import Path
from polus.tabular.transforms.rt_cetsa_metadata import (
    preprocess_metadata,
    preprocess_from_range,
)

RAW_DIR = Path(__file__).parent / "data" / "images"
PREPROCESSED_DIR = Path(__file__).parent / "out"
PREPROCESSED_DIR.mkdir(exist_ok=True)

CAMERA_FILE = Path(__file__).parent / "data" / "CameraData.xlsx"


def test_preprocess_from_range():
    preprocess_from_range(RAW_DIR, PREPROCESSED_DIR, range_t=(37, 90))


def test_preprocess_metadata():
    preprocess_metadata(RAW_DIR, PREPROCESSED_DIR, CAMERA_FILE)
