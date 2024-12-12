"""Tabular Thresholding."""
import enum
import json
import logging
import os
import pathlib
import warnings
from typing import Union

import numpy as np
import pyarrow as pa
import pyarrow.csv as pacsv
import pyarrow.parquet as pq
import pyarrow.feather as pf

from .thresholding import custom_fpr
from .thresholding import n_sigma
from .thresholding import otsu

logger = logging.getLogger(__name__)

POLUS_TAB_EXT = os.environ.get("POLUS_TAB_EXT", ".arrow")


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
        # Read CSV using pyarrow.csv
        table = pacsv.read_csv(file)
    else:
        # For Arrow or Parquet files, load directly as PyArrow table
        if file.suffix == ".arrow":
            table = pa.ipc.open_file(file).read_all()
        elif file.suffix == ".parquet":
            table = pq.read_table(file)
        elif file.suffix == ".feather":
            table = pf.read_feather(file)
        else:
            raise ValueError(f"Unsupported file format: {file.suffix}")

    plate = table["plate"].unique()[0]

    # Check for missing columns based on whether pos_control is provided
    missing_columns = (
        not all(item in table.column_names for item in [var_name, neg_control])
        if pos_control is None
        else not all(
            item in table.column_names for item in [var_name, neg_control, pos_control]
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

    if table.num_rows == 0:
        msg = f"File {file} is not loaded properly! Please check input files again!"
        logger.error(msg)
        raise ValueError(msg)

    # Convert to Pandas for specific operations if needed
    df = table.to_pandas()

    unique_neg = df[neg_control].unique()

    if not np.array_equal(np.sort(unique_neg), [0, 1]):
        msg = (
            f"The {neg_control} column has unique values {unique_neg}, "
            "which are not exactly [0, 1]. Ensure proper negative controls are set."
        )
        logger.error(msg)
        raise ValueError(msg)

    if pos_control:
        unique_positive = df[pos_control].unique()
        if not np.array_equal(np.sort(unique_positive), [0, 1]):
            msg = (
                f"The {pos_control} column has unique values {unique_positive}, "
                "which are not exactly [0, 1]. Verify positive controls"
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
        df["FPR"] = np.where(df[var_name] <= threshold, 0, 1)

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
            df["OTSU"] = np.where(df[var_name] <= threshold, 0, 1)
    elif threshold_type == "nsigma":
        threshold = n_sigma.find_threshold(neg_controls, n=n)
        threshold_dict["NSIGMA"] = threshold
        df["NSIGMA"] = np.where(df[var_name] <= threshold, 0, 1)
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
            df["OTSU"] = np.where(df[var_name] <= otsu_thr, 0, 1)

            nsigma_thr = n_sigma.find_threshold(neg_controls, n=n)
            threshold_dict["FPR"] = fpr_thr
            threshold_dict["NSIGMA"] = nsigma_thr
            df["FPR"] = np.where(df[var_name] <= fpr_thr, 0, 1)
            df["NSIGMA"] = np.where(df[var_name] <= nsigma_thr, 0, 1)

    outjson = out_dir.joinpath(f"{plate}_thresholds.json")
    with pathlib.Path.open(outjson, "w") as outfile:
        json.dump(threshold_dict, outfile)
    logger.info(f"Saving Thresholds in JSON fileformat {outjson}")
    
    out_format = POLUS_TAB_EXT
    outname = out_dir.joinpath(f"{plate}_binary{POLUS_TAB_EXT}")

    if out_format in [".feather", ".arrow", ".parquet"]:
        # Convert back to PyArrow Table if output is .arrow or .parquet
        output_table = pa.Table.from_pandas(df)
        
        if out_format == ".arrow":
            with pa.OSFile(str(outname), "wb") as sink:
                writer = pa.ipc.new_file(sink, output_table.schema)
                writer.write_table(output_table)
                writer.close()
        
        elif out_format == ".parquet":
            pq.write_table(output_table, outname)
        
    elif out_format == ".feather":
        pf.write_feather(output_table, outname)

    else:
        pacsv.write_csv(output_table, outname)

    logger.info(f"Saving {plate}_binary{out_format}")