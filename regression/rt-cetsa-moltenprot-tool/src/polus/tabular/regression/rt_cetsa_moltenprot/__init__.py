"""RT_CETSA Moltprot Regression Tool."""

__version__ = "0.2.0-dev0"

import os
import pathlib
import warnings

import pandas

from . import core

# Suppress FutureWarning messages coming from pandas
warnings.simplefilter(action="ignore", category=FutureWarning)

POLUS_TAB_EXT = os.environ.get("POLUS_TAB_EXT", ".csv")


def fit_data(file_path: pathlib.Path) -> pandas.DataFrame:
    """Fit data to a model using Moltprot."""
    fit = core.MoltenProtFit(
        filename=file_path,
        input_type="csv",
    )

    fit.SetAnalysisOptions(
        model="santoro1988",
        baseline_fit=3,
        baseline_bounds=3,
        dCp=0,
        onset_threshold=0.01,
        savgol=10,
        blanks=[],
        exclude=[],
        invert=False,
        mfilt=None,
        shrink=None,
        trim_max=0,
        trim_min=0,
    )

    fit.PrepareData()
    fit.ProcessData()

    return fit
