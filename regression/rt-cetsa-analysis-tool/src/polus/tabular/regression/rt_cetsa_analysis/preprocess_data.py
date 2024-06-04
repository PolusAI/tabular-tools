"""Preprocess Data."""

from pathlib import Path

import openpyxl


def preprocess_platemap(platemap_path: Path, out_dir: Path):
    """Preprocess platemap to normalize inputs for downstream tasks."""
    platemap = openpyxl.load_workbook(platemap_path)
    # printing the sheet names
    print(platemap.sheetnames)
    for name in platemap.sheetnames:
        if "sample" in name.lower():
            sample = platemap[name]
            sample.title = "sample"
        if "conc" in name.lower():
            conc = platemap[name]
            conc.title = "conc"
    print(f"after preprocessing : {platemap.sheetnames}")

    processed_platemap_path = out_dir / platemap_path.name
    platemap.save(processed_platemap_path)
    return processed_platemap_path
