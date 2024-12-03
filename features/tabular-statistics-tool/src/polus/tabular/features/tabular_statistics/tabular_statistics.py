"""Tabular Statistics."""
import logging
import os
from scipy.stats import kurtosis
from typing import Optional
import numpy as np
import pandas as pd
import math
import pyarrow as pa
import pyarrow.csv as csv
import pyarrow.ipc as ipc
import pyarrow.feather as feather
import pyarrow.parquet as pq
from pathlib import Path
from typing import List
from typing import Any
from tqdm import tqdm
import json

logger = logging.getLogger(__name__)
logger.setLevel(os.environ.get("POLUS_LOG", logging.INFO))
POLUS_TAB_EXT = os.environ.get("POLUS_TAB_EXT", ".csv")



def mean(values: list[float]) -> float:
    """
    Calculate the mean of a list of numbers.

    Args:
        values: A list of numerical values.

    Returns:
        float: The mean or average of the input values.

    Raises:
        ValueError: If the input list is empty.
    """
    if values.empty:
        raise ValueError("The input list is empty. Cannot calculate the mean.")
    return np.mean(values)


def median(values: list[float]) -> float:
    """
    Calculate the median of a list of numbers.

    Args:
        values: A list of numerical values.

    Returns:
        float: The median value of the input values.

    Raises:
        ValueError: If the input list is empty.
    """
    if values.empty:
        raise ValueError("The input list is empty. Cannot calculate the median.")
    return np.median(values)

def std(values: list[float]) -> float:
    """
    Calculate the standard deviation of a list of numbers.

    Args:
        values: A list of numerical values.

    Returns:
        float: The standard deviation of the input values.

    Raises:
        ValueError: If the input list is empty.
    """
    if values.empty:
        raise ValueError("The input list is empty. Cannot calculate the standard deviation.")
    return np.std(values)

def var(values: list[float]) -> float:
    """
    Calculate the variance of a list of numbers.

    Args:
        values: A list of numerical values.

    Returns:
        float: The variance of the input values.

    Raises:
        ValueError: If the input list is empty.
    """
    if values.empty:
        raise ValueError("The input list is empty. Cannot calculate the variance.")
    return np.var(values)


def skew(values: list[float]) -> float:
    """
    Calculate the skewness of a list of numbers.

    Args:
        values: A list of numerical values.

    Returns:
        float: The skewness of the input values.

    Raises:
        ValueError: If the input list is empty.
    """
    if values.empty:
        raise ValueError("The input list is empty. Cannot calculate the skewness.")
    
    n = len(values)
    mean = sum(values) / n
    std_dev = math.sqrt(sum((x - mean) ** 2 for x in values) / n)
    
    if std_dev == 0:
        return float('nan')  # If standard deviation is 0, skewness is undefined
    
    skew_value = sum((x - mean) ** 3 for x in values) / (n * std_dev ** 3)
    return skew_value

def kurt(values: list[float]) -> float:
    """
    Calculate the kurtosis of a list of numbers.

    Args:
        values: A list of numerical values.

    Returns:
        float: The kurtosis of the input values.

    Raises:
        ValueError: If the input list is empty.
    """
    if values.empty:
        raise ValueError("The input list is empty. Cannot calculate the kurtosis.")
    return kurtosis(values)

def count(values: list[float]) -> int:
    """
    Count the number of elements in a list.

    Args:
        values: A list of numerical values.

    Returns:
        int: The number of elements in the input list.

    Raises:
        ValueError: If the input list is empty.
    """
    if values.empty:
        raise ValueError("The input list is empty. Cannot calculate the count.")
    return len(values)



def max(values: list[float]) -> float:
    """
    Calculate the maximum value from a list of numbers.

    Args:
        values: A list of numerical values.

    Returns:
        float: The maximum value in the input list.

    Raises:
        ValueError: If the input list is empty.
    """
    if values.empty:
        raise ValueError("The input list is empty. Cannot calculate the maximum.")
    return np.max(values)

def min(values: list[float]) -> float:
    """
    Calculate the minimum value from a list of numbers.

    Args:
        values: A list of numerical values.

    Returns:
        float: The minimum value in the input list.

    Raises:
        ValueError: If the input list is empty.
    """
    if values.empty:
        raise ValueError("The input list is empty. Cannot calculate the minimum.")
    return np.min(values)

def iqr(values: list[float]) -> float:
    """
    Calculate the interquartile range (IQR) of a list of numbers.

    Args:
        values: A list of numerical values.

    Returns:
        float: The IQR of the input values.

    Raises:
        ValueError: If the input list is empty.
    """
    if values.empty:
        raise ValueError("The input list is empty. Cannot calculate the IQR.")
    return np.percentile(values, 75) - np.percentile(values, 25)

def proportion(values: list[float]) -> float:
    """
    Calculate the proportion of positive values in a list.

    Args:
        values: A list of numerical values.

    Returns:
        float: The proportion of positive values in the input list.

    Raises:
        ValueError: If the input list is empty.
    """
    if values.empty:
        raise ValueError("The input list is empty. Cannot calculate the proportion.")
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
    "prop": proportion
}


 # Function to apply statistics to each group
def apply_statistics(table: pa.Table,  statistics: str, group_by_columns:Optional[str]=None): 
        """
        Apply statistical functions to each group in a PyArrow Table.

        Args:
            table: Input PyArrow Table.
            group_by_columns: Columns to group by.
            statistics: Comma-separated list of statistics or 'all' to apply all available statistics.
        
        Returns:
            pyarrow.Table: Aggregated statistics for each numeric column.
        """
        # Step 1: Convert PyArrow Table to Pandas DataFrame for easy manipulation
        if statistics == "all":
            statistics = STATS
        else:
            statistics = statistics.split(',')
            #Check if all keys in statistics are in the STATS dictionary
            missing_keys = [key for key in statistics if key not in STATS]
            if missing_keys:
                raise KeyError(f"Invalid statistics: {', '.join(missing_keys)}")
            
            statistics = {k: v for k, v in STATS.items() if k in statistics}


        df = table.to_pandas()

        numeric_columns = df.select_dtypes(include='number').columns.tolist()

        # # Step 5: Apply the statistics to each group
        aggregated_results = {}

        for stat_name, stat_func in statistics.items():
            for col in  numeric_columns:
                col_stat_name = f"{col}_{stat_name}"
                if group_by_columns:
                    # Aggregate without grouping
                    aggregated_results[col_stat_name] = df.groupby(group_by_columns)[col].apply(stat_func)
                else:
                    # Aggregate without grouping
                    aggregated_results[col_stat_name] = [stat_func(df[col])]

        if group_by_columns:
            group_table = pd.DataFrame(aggregated_results)
            # Perform an left join on the "group_by_columns" column
            arrow_table = pd.merge(df, group_table, on=group_by_columns, how='left')

            arrow_table.reset_index(inplace=True)
            arrow_table = pa.Table.from_pandas(arrow_table)


        else:
            arrow_table = pa.Table.from_pydict(aggregated_results)

        return arrow_table


def save_outputs(data:pa.Table, out_dir:Path) -> None:
    """
    Save data to a file in CSV, Feather, Arrow, or Parquet format.

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
        with pa.OSFile(str(output_file), 'wb') as arrow_file:
            with ipc.new_file(data,data.schema) as writer:
                writer.write(data)

    elif POLUS_TAB_EXT == ".parquet":
        logger.info(f"Saved outputs:{output_file}")
        pq.write_table(data, output_file)

    else:
        raise ValueError(f"Unsupported extension: {POLUS_TAB_EXT}. Supported: .csv, .feather, .arrow, .parquet.")

    
    logger.info(f"File saved successfully:{output_file}")



def load_files(flist: List[Path]) -> pa.Table:
    """
    Load multiple files (CSV, Feather, Arrow, Parquet) and return a concatenated PyArrow Table.

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
                df[df.select_dtypes(include=['object']).columns] = \
                    df.select_dtypes(include=['object']).map(str)
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
        except Exception as e:
            logger.error(f"Error processing file {file.name}: {e}")

    # Concatenate all loaded tables into a single table
    combined_table = pa.concat_tables(tables)

    return combined_table


def preview(out_dir:Path, file_pattern:str) -> None:
    """
    Create a preview JSON file with output directory and file pattern details.

    Args:
        out_dir: Directory for the outputs.
        file_pattern: The file pattern used for processing.
    """

    out_name= out_dir.joinpath(f"tabular_statistics_output{POLUS_TAB_EXT}")

    out_json: dict[str, Any] = {
        "filepattern": file_pattern,
        "outDir": str(out_name),
    }
    with Path(out_dir, "preview.json").open("w") as jfile:
        json.dump(out_json, jfile, indent=2)

