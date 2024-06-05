"""Preprocess Data."""

from pathlib import Path

import openpyxl
import pandas as pd


def preprocess_data(platemap_path, values_path, params_path, out_dir):
    """Preprocessing all data before running analysis."""
    prepared_platemap = prepare_platemap(platemap_path, out_dir)
    sample = preprocess_platemap_sample(prepared_platemap, out_dir)
    conc = preprocess_platemap_conc(prepared_platemap, out_dir)
    values = preprocess_values(values_path, out_dir)
    params = preprocess_params(params_path, out_dir)
    df = pd.concat([sample, conc, params, values], axis=1)
    df = df.astype({"row": int, "col": int})
    first_column = df.pop("well")
    df.insert(0, "well", first_column)
    data_path = out_dir / "data.csv"
    df.to_csv(data_path, index=True)
    return data_path


def prepare_platemap(platemap_path: Path, out_dir: Path):
    """Preprocess platemap to normalize inputs for downstream tasks."""
    platemap = openpyxl.load_workbook(platemap_path)
    for name in platemap.sheetnames:
        if "sample" in name.lower():
            sample = platemap[name]
            sample.title = "sample"
        if "conc" in name.lower():
            conc = platemap[name]
            conc.title = "conc"

    prepared_platemap_path = out_dir / platemap_path.name
    platemap.save(prepared_platemap_path)
    return prepared_platemap_path


def preprocess_platemap_sample(platemap_path: Path, out_dir: Path):
    """Preprocess platemap sample sheet.

    Returns:
        dataframe with row, col and ncgc_id columns.
    """
    df = pd.read_excel(platemap_path, "sample")
    df.drop(columns=df.columns[0], axis=1, inplace=True)
    df = df.stack()
    df = df.reset_index(level=[0, 1], name="ncgc_id")
    df.rename(columns={"level_0": "row", "level_1": "col"}, inplace=True)
    df.index += 1

    df = df.replace("empty", "vehicle")  # empty are relabed as vehicle

    processed_platemap_path = out_dir / (platemap_path.stem + "_sample.csv")
    df.to_csv(processed_platemap_path, index=True)
    return df


def preprocess_platemap_conc(platemap_path: Path, out_dir: Path):
    """Preprocess platemap conc sheet.

    Returns:
        dataframe with conc column.
    """
    df = pd.read_excel(platemap_path, "conc")
    df.drop(columns=df.columns[0], axis=1, inplace=True)
    df = df.stack()
    df = df.reset_index(level=[0, 1], name="conc")
    df.drop(columns=["level_0", "level_1"], inplace=True)
    df.index += 1
    processed_platemap_path = out_dir / (platemap_path.stem + "_conc.csv")
    df.to_csv(processed_platemap_path, index=True)
    return df


def preprocess_params(params_path: Path, out_dir: Path):
    """Preprocess moltenprot params.

    Returns:
        dataframe subselection of columns.
        All values are converted to celsius.
    """
    df = pd.read_csv(params_path)
    df = df[["dHm_fit", "Tm_fit", "BS_factor", "T_onset", "dG_std"]]
    df["Tm_fit"] = df["Tm_fit"] - 273.15
    df["T_onset"] = df["T_onset"] - 273.15
    df.index += 1
    processed_params_path = out_dir / params_path.name
    df.to_csv(processed_params_path, index=True)
    return df


def preprocess_values(values_path: Path, out_dir: Path):
    """Preprocess moltenprot values.

    Returns:
        dataframe measurement series for each well.
        All temperature are converted to celsius
    """
    df = pd.read_csv(values_path)
    # update temp to plug to the rest of R code.
    df["Temperature"] = df["Temperature"] - 273.15
    df["Temperature"] = df["Temperature"].round(2)
    df["Temperature"] = df["Temperature"].apply(lambda t: "t_" + str(t))
    # the only step that is really necessary
    df = df.transpose()
    df.columns = df.iloc[0]  # make the first row the column names
    df = df.drop(df.index[0])  # drop first row
    df = df.reset_index(names=["well"])  # remove well names and use seq index
    df.index += 1  # should start at 1
    processed_values_path = out_dir / values_path.name
    df.to_csv(processed_values_path, index=True)
    return df


def preprocess_values_R(values_path: Path, out_dir: Path):
    """NOTE: this should be removed."""
    df = pd.read_csv(values_path)
    # update temp to plug to the rest of R code.
    df["Temperature"] = df["Temperature"] - 273.15
    df["Temperature"] = df["Temperature"].round(2)
    df["Temperature"] = df["Temperature"].apply(lambda t: "t_" + str(t))
    # the only step that is really necessary
    df = df.transpose()
    df.columns = df.iloc[0]  # make the first row the column names
    df = df.drop(df.index[0])  # drop first row
    df = df.reset_index(drop=True)  # remove well names and use seq index
    df.index += 1  # should start at 1
    processed_values_path = out_dir / values_path.name
    df.to_csv(processed_values_path, index=True)
    return processed_values_path
