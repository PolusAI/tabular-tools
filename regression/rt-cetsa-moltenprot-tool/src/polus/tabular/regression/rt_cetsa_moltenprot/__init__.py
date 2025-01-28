"""RT_CETSA Moltprot Regression Tool."""

__version__ = "0.5.0-dev0"

import os
import pathlib
import warnings

import pandas

from . import core

# Suppress FutureWarning messages coming from pandas
warnings.simplefilter(action="ignore", category=FutureWarning)

POLUS_TAB_EXT = os.environ.get("POLUS_TAB_EXT", ".csv")


def run_moltenprot_fit(
    file_path: pathlib.Path,
    moltenprot_params: dict[str, int],
) -> tuple[pandas.DataFrame, pandas.DataFrame]:
    """Run moltenprot.

    Args:
        file_path : path to intensities file.

    Returns:
        tuple of dataframe containing the fit_params and the curve values.
    """
    fit = fit_data(file_path, moltenprot_params)

    # sort fit_params by row/column
    fit_params = fit.plate_results
    fit_params["_index"] = fit_params.index
    fit_params["letter"] = fit_params.apply(lambda row: row._index[:1], axis=1)
    fit_params["number"] = fit_params.apply(
        lambda row: row._index[1:],
        axis=1,
    ).astype(int)
    fit_params = fit_params.drop(columns="_index")
    fit_params = fit_params.sort_values(["letter", "number"])
    fit_params = fit_params.drop(columns=["letter", "number"])

    # keep only 2 signicant digits for temperature index
    fit_curves = fit.plate_raw_corr
    fit_curves.index = fit_curves.index.map(lambda t: round(t, 2))

    return fit_params, fit_curves


def fit_data(
    file_path: pathlib.Path,
    moltenprot_params: dict[str, int],
) -> core.MoltenProtFit:
    """Fit data to a model using Moltprot."""
    fit = core.MoltenProtFit(
        filename=file_path,
        input_type="csv",
    )

    fit.SetAnalysisOptions(
        model="santoro1988",
        baseline_fit=moltenprot_params["baseline_fit"],
        baseline_bounds=moltenprot_params["baseline_bounds"],
        dCp=0,
        onset_threshold=0.01,
        savgol=moltenprot_params["savgol"],
        blanks=[],
        exclude=[],
        invert=False,
        mfilt=None,
        shrink=None,
        trim_max=moltenprot_params["trim_max"],
        trim_min=moltenprot_params["trim_min"],
    )

    fit.PrepareData()
    fit.ProcessData()

    return fit
