"""Tabular Thresholding."""
import enum
import json
import logging
import os
import pathlib
import warnings
from typing import Union

import numpy as np
import vaex

from .thresholding import custom_fpr
from .thresholding import n_sigma
from .thresholding import otsu

logger = logging.getLogger(__name__)

POLUS_TAB_EXT = os.environ.get("POLUS_TAB_EXT", ".csv")


class Methods(str, enum.Enum):
    """Threshold methods."""

    OTSU = "otsu"
    NSIGMA = "n_sigma"
    FPR = "fpr"
    ALL = "all"
    Default = "all"


def thresholding_func(  # noqa: PLR0915, PLR0912, PLR0913, C901
    neg_control: str,
    pos_control: str,
    var_name: str,
    threshold_type: Methods,
    false_positive_rate: float,
    num_bins: int,
    n: int,
    out_dir: pathlib.Path,
    file: pathlib.Path,
) -> None:
    """Compute variable threshold using negative or negative and positive control data.

    Computes the variable value of each ROI if above or below threshold. The control
    data used for computing threshold depends on the type of thresholding methods
    https://github.com/nishaq503/thresholding.git.

    Args:
        neg_control: Column name containing information of non treated wells.
        pos_control: Column name for the well with known treatment
        var_name: Column name for computing thresholds.
        threshold_type: Name of threshold method.
        false_positive_rate: Tuning parameter.
        num_bins: Number of bins.
        n: Number of standard deviation away from mean value.
        out_dir: Output directory.
        file: Filename.

    """
    if file.suffix == ".csv":
        df = vaex.from_csv(file)
    else:
        df = vaex.open(file, progress=True)

    plate = df["plate"].unique()[0]

    # Check for missing columns based on whether pos_control is provided
    missing_columns = (
        not all(item in df.columns for item in [var_name, neg_control])
        if pos_control is None
        else not all(
            item in df.columns for item in [var_name, neg_control, pos_control]
        )
    )

    if missing_columns:
        missing_msg = (
            f"{file} is missing {var_name} and {neg_control} columns."
            if pos_control is None
            else f"{file} is missing {var_name}, {neg_control}, {pos_control} column."
        )
        logger.error(missing_msg)
        raise ValueError(missing_msg)

    if df.shape == (0, 0):
        msg = f"File {file} is not loaded properly! Please check input files again!"
        logger.error(msg)
        raise ValueError(msg)

    unique_neg = df[neg_control].unique()

    if unique_neg != [0.0, 1.0]:
        msg = (
            f"The {neg_control} column has unique values {unique_neg}, "
            "which are not exactly [0.0, 1.0]. Ensure proper negative controls are set."
        )
        logger.error(msg)
        raise ValueError(msg)

    if pos_control:
        unique_positive = df[pos_control].unique()
        if unique_positive != [0.0, 1.0]:
            msg = (
                f"The {pos_control} column has unique values {unique_positive}, "
                "which are not exactly [0.0, 1.0]. Verify positive controls"
            )
            logger.error(msg)
            raise ValueError(msg)

    if pos_control is None:
        msg = "pos_control is missing. Otsu threshold will not be computed!"
        logger.info(msg)

    threshold_dict: dict[str, Union[float, str]] = {}
    nan_value = np.nan * np.arange(0, len(df[neg_control].values), 1)
    threshold_dict["FPR"] = np.nan
    threshold_dict["OTSU"] = np.nan
    threshold_dict["NSIGMA"] = np.nan
    df["FPR"] = nan_value
    df["OTSU"] = nan_value
    df["NSIGMA"] = nan_value

    if pos_control:
        pos_controls = df[df[pos_control] == 1][var_name].values

    neg_controls = df[df[neg_control] == 1][var_name].values

    if threshold_type == "fpr":
        threshold = custom_fpr.find_threshold(
            neg_controls,
            false_positive_rate=false_positive_rate,
        )
        threshold_dict["FPR"] = threshold
        df["FPR"] = df.func.where(df[var_name] <= threshold, 0, 1)

    elif threshold_type == "otsu":
        if len(pos_controls) == 0:
            msg = f"{pos_control} controls missing. NaN values for Otsu thresholds"
            logger.error(msg)
            threshold_dict["OTSU"] = np.nan
            df["OTSU"] = np.nan * np.arange(0, len(df[var_name].values), 1)
        else:
            combine_array = np.append(neg_controls, pos_controls, axis=0)
            threshold = otsu.find_threshold(
                combine_array,
                num_bins=num_bins,
                normalize_histogram=False,
            )
            threshold_dict["OTSU"] = threshold
            df["OTSU"] = df.func.where(df[var_name] <= threshold, 0, 1)
    elif threshold_type == "nsigma":
        threshold = n_sigma.find_threshold(neg_controls, n=n)
        threshold_dict["NSIGMA"] = threshold
        df["NSIGMA"] = df.func.where(df[var_name] <= threshold, 0, 1)
    elif threshold_type == "all":
        fpr_thr = custom_fpr.find_threshold(
            neg_controls,
            false_positive_rate=false_positive_rate,
        )
        combine_array = np.append(neg_controls, pos_controls, axis=0)

        if len(pos_controls) == 0:
            warnings.warn(  # noqa: B028
                f"{pos_control} missing; NaN values computed for Otsu thresholds",
            )
            threshold_dict["OTSU"] = np.nan
            df["OTSU"] = np.nan * np.arange(0, len(df[var_name].values), 1)
        else:
            otsu_thr = otsu.find_threshold(
                combine_array,
                num_bins=num_bins,
                normalize_histogram=False,
            )
            threshold_dict["OTSU"] = otsu_thr
            df["OTSU"] = df.func.where(df[var_name] <= otsu_thr, 0, 1)

            nsigma_thr = n_sigma.find_threshold(neg_controls, n=n)
            threshold_dict["FPR"] = fpr_thr
            threshold_dict["NSIGMA"] = nsigma_thr
            df["FPR"] = df.func.where(df[var_name] <= fpr_thr, 0, 1)
            df["NSIGMA"] = df.func.where(df[var_name] <= nsigma_thr, 0, 1)

    outjson = out_dir.joinpath(f"{plate}_thresholds.json")
    with pathlib.Path.open(outjson, "w") as outfile:
        json.dump(threshold_dict, outfile)
    logger.info(f"Saving Thresholds in JSON fileformat {outjson}")
    

    out_format = POLUS_TAB_EXT
    outname = out_dir.joinpath(f"{plate}_binary{POLUS_TAB_EXT}")

    if out_format in [".feather", ".arrow", ".parquet", ".hdf5"]:
        df.export(outname)
    else:
        df.export_csv(path=outname, chunk_size=10_000)

    logger.info(f"Saving{plate}_binary{out_format}")
