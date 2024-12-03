"""Tabular Statistics."""
import json
import logging
import math
import os
from pathlib import Path
from typing import Any
from typing import Optional

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from pyarrow import csv
from pyarrow import feather
from pyarrow import ipc
from scipy.stats import kurtosis
from tqdm import tqdm

logger = logging.getLogger(__name__)
logger.setLevel(os.environ.get("POLUS_LOG", logging.INFO))
POLUS_TAB_EXT = os.environ.get("POLUS_TAB_EXT", ".feather")


def mean(values: list[float]) -> float:
    """Calculate the mean of a list of numbers.

    Args:
        values: A list of numerical values.

    Returns:
        float: The mean or average of the input values.

    Raises:
        ValueError: If the input list is empty.
    """
    if len(values) == 0:
        msg = "The input list is empty. Cannot calculate the mean."
        raise ValueError(msg)
    return np.mean(values)


def median(values: list[float]) -> float:
    """Calculate the median of a list of numbers.

    Args:
        values: A list of numerical values.

    Returns:
        float: The median value of the input values.

    Raises:
        ValueError: If the input list is empty.
    """
    if len(values) == 0:
        msg = "The input list is empty. Cannot calculate the median."
        raise ValueError(msg)
    return np.median(values)


def std(values: list[float]) -> float:
    """Calculate the standard deviation of a list of numbers.

    Args:
        values: A list of numerical values.

    Returns:
        float: The standard deviation of the input values.

    Raises:
        ValueError: If the input list is empty.
    """
    if len(values) == 0:
        msg = "The input list is empty. Cannot calculate the standard deviation."
        raise ValueError(
            msg,
        )
    return np.std(values)


def var(values: list[float]) -> float:
    """Calculate the variance of a list of numbers.

    Args:
        values: A list of numerical values.

    Returns:
        float: The variance of the input values.

    Raises:
        ValueError: If the input list is empty.
    """
    if len(values) == 0:
        msg = "The input list is empty. Cannot calculate the variance."
        raise ValueError(msg)
    return np.var(values)


def skew(values: list[float]) -> float:
    """Calculate the skewness of a list of numbers.

    Args:
        values: A list of numerical values.

    Returns:
        float: The skewness of the input values.

    Raises:
        ValueError: If the input list is empty.
    """
    if len(values) == 0:
        msg = "The input list is empty. Cannot calculate the skewness."
        raise ValueError(msg)

    n = len(values)
    mean = sum(values) / n
    std_dev = math.sqrt(sum((x - mean) ** 2 for x in values) / n)

    if std_dev == 0:
        return float("nan")  # If standard deviation is 0, skewness is undefined

    return sum((x - mean) ** 3 for x in values) / (n * std_dev**3)


def kurt(values: list[float]) -> float:
    """Calculate the kurtosis of a list of numbers.

    Args:
        values: A list of numerical values.

    Returns:
        float: The kurtosis of the input values.

    Raises:
        ValueError: If the input list is empty.
    """
    if len(values) == 0:
        msg = "The input list is empty. Cannot calculate the kurtosis."
        raise ValueError(msg)
    return kurtosis(values)


def count(values: list[float]) -> int:
    """Count the number of elements in a list.

    Args:
        values: A list of numerical values.

    Returns:
        int: The number of elements in the input list.

    Raises:
        ValueError: If the input list is empty.
    """
    if len(values) == 0:
        msg = "The input list is empty. Cannot calculate the count."
        raise ValueError(msg)
    return len(values)


def max(values: list[float]) -> float:  # noqa: A001
    """Calculate the maximum value from a list of numbers.

    Args:
        values: A list of numerical values.

    Returns:
        float: The maximum value in the input list.

    Raises:
        ValueError: If the input list is empty.
    """
    if len(values) == 0:
        msg = "The input list is empty. Cannot calculate the maximum."
        raise ValueError(msg)
    return np.max(values)


def min(values: list[float]) -> float:  # noqa: A001
    """Calculate the minimum value from a list of numbers.

    Args:
        values: A list of numerical values.

    Returns:
        float: The minimum value in the input list.

    Raises:
        ValueError: If the input list is empty.
    """
    if len(values) == 0:
        msg = "The input list is empty. Cannot calculate the minimum."
        raise ValueError(msg)
    return np.min(values)


def iqr(values: list[float]) -> float:
    """Calculate the interquartile range (IQR) of a list of numbers.

    Args:
        values: A list of numerical values.

    Returns:
        float: The IQR of the input values.

    Raises:
        ValueError: If the input list is empty.
    """
    if len(values) == 0:
        msg = "The input list is empty. Cannot calculate the IQR."
        raise ValueError(msg)
    return np.percentile(values, 75) - np.percentile(values, 25)


def proportion(values: list[float]) -> float:
    """Calculate the proportion of positive values in a list.

    Args:
        values: A list of numerical values.

    Returns:
        float: The proportion of positive values in the input list.

    Raises:
        ValueError: If the input list is empty.
    """
    if len(values) == 0:
        msg = "The input list is empty. Cannot calculate the proportion."
        raise ValueError(msg)
    return sum(1 for x in values if x > 0) / len(values)


# STATS dictionary with statistical functions
STATS = {
    "mean": mean,
    "median": median,
    "std": std,
    "var": var,
    "skew": skew,
    "kurt": kurt,
    "count": count,
    "max": max,
    "min": min,
    "iqr": iqr,
    "prop": proportion,
}


# Function to apply statistics to each group
def apply_statistics(
    table: pa.Table,
    statistics: str,
    group_by_columns: Optional[str] = None,
) -> pa.Table:
    """Apply statistical functions to each group in a PyArrow Table.

    Args:
        table: Input PyArrow Table.
        group_by_columns: Columns to group by.
        statistics: Comma-separated list or 'all' for all statistics.

    Returns:
        pyarrow.Table: Aggregated statistics for each numeric column.
    """
    # Step 1: Convert PyArrow Table to Pandas DataFrame for easy manipulation
    if statistics == "all":
        metrics = STATS
    else:
        metrics = statistics.split(",")  # type: ignore
        # Check if all keys in statistics are in the STATS dictionary
        missing_keys = [key for key in metrics if key not in STATS]
        if missing_keys:
            msg = f"Invalid statistics: {', '.join(missing_keys)}"
            raise KeyError(msg)

        metrics = {k: v for k, v in STATS.items() if k in metrics}

    df = table.to_pandas()

    numeric_columns = df.select_dtypes(include="number").columns.tolist()

    # # Step 5: Apply the statistics to each group
    aggregated_results = {}

    for stat_name, stat_func in metrics.items():
        for col in numeric_columns:
            col_stat_name = f"{col}_{stat_name}"
            if group_by_columns:
                # Aggregate without grouping
                aggregated_results[col_stat_name] = df.groupby(group_by_columns)[
                    col
                ].apply(stat_func)
            else:
                # Aggregate without grouping
                aggregated_results[col_stat_name] = [stat_func(df[col])]

    if group_by_columns:
        group_table = pd.DataFrame(aggregated_results)
        # Perform an left join on the "group_by_columns" column
        arrow_table = pd.merge(df, group_table, on=group_by_columns, how="left")

        arrow_table.reset_index(inplace=True)
        arrow_table = pa.Table.from_pandas(arrow_table)

    else:
        arrow_table = pa.Table.from_pydict(aggregated_results)

    return arrow_table


def save_outputs(data: pa.Table, out_dir: Path) -> None:
    """Save data to a file in CSV, Feather, Arrow, or Parquet format.

    Args:
        data: The data to save.
        out_dir: Directory to save the file.
        file: Base filename for the saved file.
    """
    # Determine the output file path and extension based on the desired format
    output_file = out_dir.joinpath(f"tabular_statistics_output{POLUS_TAB_EXT}")

    if POLUS_TAB_EXT == ".csv":
        logger.info(f"Saved outputs:{output_file}")
        csv.write_csv(data, output_file)

    elif POLUS_TAB_EXT == ".feather":
        logger.info(f"Saved outputs: {output_file}")
        feather.write_feather(data, output_file)

    elif POLUS_TAB_EXT == ".arrow":
        logger.info(f"Saved outputs:{output_file}")
        record_batch = data.to_batches()
        # Write the data using ipc.new_file
        with pa.OSFile(str(output_file), "wb") as sink:  # noqa:SIM117
            with ipc.new_file(sink, data.schema) as writer:
                for batch in record_batch:
                    writer.write(batch)

    elif POLUS_TAB_EXT == ".parquet":
        logger.info(f"Saved outputs:{output_file}")
        pq.write_table(data, output_file)

    else:
        msg = f"Unsupported extension: {POLUS_TAB_EXT}"
        raise ValueError(
            msg,
        )

    logger.info(f"File saved successfully:{output_file}")


def load_files(flist: list[Path]) -> pa.Table:
    """Load and concatenate multiple files into a PyArrow Table.

    Args:
        flist: List of file paths to load.

    Returns:
        pa.Table: Concatenated table of all loaded data.
    """
    tables = []

    for file in tqdm(flist, desc="Processing files"):
        try:
            if file.suffix == ".csv":
                logger.info(f"Loading CSV file: {file.name}")
                # Read CSV with pandas and convert to PyArrow Table
                df = pd.read_csv(file)
                df[df.select_dtypes(include=["object"]).columns] = df.select_dtypes(
                    include=["object"],
                ).map(str)
                table = pa.Table.from_pandas(df)
            elif file.suffix == ".feather":
                logger.info(f"Loading Feather file: {file.name}")
                # Read Feather file directly using PyArrow
                table = feather.read_table(str(file))
            elif file.suffix == ".arrow":
                logger.info(f"Loading Arrow file: {file.name}")
                # Read Arrow file using PyArrow IPC
                with pa.memory_map(str(file), "r") as source:
                    table = ipc.open_file(source).read_all()
            elif file.suffix == ".parquet":
                logger.info(f"Loading Parquet file: {file.name}")
                # Read Parquet file using PyArrow
                table = pq.read_table(str(file))
            else:
                logger.error(f"Unsupported file format: {file.suffix}")
                continue  # Skip unsupported files

            tables.append(table)
        except (OSError, ValueError) as e:
            logger.error(f"Error processing file {file.name}: {e}")

    # Concatenate all loaded tables into a single table
    return pa.concat_tables(tables)


def preview(out_dir: Path, file_pattern: str) -> None:
    """Create a preview JSON file with output directory and file pattern details.

    Args:
        out_dir: Directory for the outputs.
        file_pattern: The file pattern used for processing.
    """
    out_name = out_dir.joinpath(f"tabular_statistics_output{POLUS_TAB_EXT}")

    out_json: dict[str, Any] = {
        "filepattern": file_pattern,
        "outDir": str(out_name),
    }
    with Path(out_dir, "preview.json").open("w") as jfile:
        json.dump(out_json, jfile, indent=2)
