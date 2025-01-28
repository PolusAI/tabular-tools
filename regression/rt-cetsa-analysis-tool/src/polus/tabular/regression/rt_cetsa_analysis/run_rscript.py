"""Run R scripts."""

import logging
import os
import subprocess
from pathlib import Path

POLUS_LOG = os.environ.get("POLUS_LOG", logging.INFO)
WORKDIR = os.environ.get("WORKDIR", "")

logger = logging.getLogger("rt_cetsa_analysis")
logger.setLevel(POLUS_LOG)


def run_rscript(
    data_filepath: Path,
    out_dir: Path,
):
    """Run R script."""
    cwd = Path(__file__).parent

    if WORKDIR:
        cwd = (
            Path(WORKDIR)
            / "src"
            / "polus"
            / "tabular"
            / "regression"
            / "rt_cetsa_analysis/"
        )

    logger.info(f"current working directory : {cwd.as_posix()}")

    cmd = [
        "Rscript",
        "./main.R",
        data_filepath.as_posix(),
        out_dir.as_posix(),
    ]

    subprocess.run(args=cmd, cwd=cwd)
