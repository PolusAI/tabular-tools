"""Run R scripts."""

import logging
import os
import subprocess
from pathlib import Path

POLUS_LOG = os.environ.get("POLUS_LOG", logging.INFO)

logger = logging.getLogger("rt_cetsa_analysis")
logger.setLevel(POLUS_LOG)


def run_rscript(
    params_filepath: Path,
    values_filepath: Path,
    platemap_filepath: Path,
    out_dir: Path,
):
    """Run R script."""
    print(
        "run rscript with args: ",
        params_filepath,
        values_filepath,
        platemap_filepath,
        out_dir,
    )

    cmd = [
        "Rscript",
        "./main.R",
        "--params",
        params_filepath.as_posix(),
        "--values",
        values_filepath.as_posix(),
        "--platemap",
        platemap_filepath.as_posix(),
        "--outdir",
        out_dir.as_posix(),
    ]

    subprocess.run(args=cmd, cwd="src/polus/tabular/regression/rt_cetsa_analysis/")
