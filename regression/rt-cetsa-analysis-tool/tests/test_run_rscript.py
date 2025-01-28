"""Tests."""
from pathlib import Path

import pytest
from polus.tabular.regression.rt_cetsa_analysis.run_rscript import run_rscript


@pytest.mark.skipif("not config.getoption('slow')")
def test_run_rscript():
    """Run R script."""
    inpDir = Path.cwd() / "tests" / "data"
    params = "plate_(1-59)_moltenprot_params.csv"
    values = "plate_(1-59)_moltenprot_curves.csv"
    platemap = Path.cwd() / "tests" / "data" / "platemap.xlsx"
    outDir = Path.cwd() / "tests" / "out"

    params = inpDir / params
    values = inpDir / values

    run_rscript(params, values, platemap, outDir)
