"""Tests."""
from polus.tabular.regression.rt_cetsa_moltenprot import run_moltenprot_fit
from pathlib import Path

import pytest


@pytest.mark.skipif("not config.getoption('slow')")
def test_moltenprot():
    path = Path(__file__).parent / "data" / "plate_(1-58).csv"
    params, values = run_moltenprot_fit(path)
    assert params is not None
    assert values is not None


def test_dummy_test():
    pass
