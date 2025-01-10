import pathlib
import string
import numpy as np
import pandas as pd
import pyarrow as pa
from typer.testing import CliRunner
from polus.tabular.features.tabular_statistics.__main__ import app
from polus.tabular.features.tabular_statistics import tabular_statistics as ts


runner = CliRunner()


class Generatedata:
    """Generate tabular data with several different file formats."""

    def __init__(self, file_pattern: str, out_name: str) -> None:
        """Define instance attributes."""
        self.data_dir = pathlib.Path.cwd().parent.joinpath("data")

        self.inp_dir = pathlib.Path(self.data_dir, "input")
        if not self.inp_dir.exists():
            self.inp_dir.mkdir(exist_ok=True, parents=True)

        self.out_dir = pathlib.Path(self.data_dir, "output")
        if not self.out_dir.exists():
            self.out_dir.mkdir(exist_ok=True, parents=True)

        self.file_pattern = file_pattern
        self.out_name = out_name
        self.df = self.create_dataframe()
        self.inpdir = self.get_inp_dir()
        self.outdir = self.get_out_dir()

    def get_inp_dir(self) -> pathlib.Path:
        """Get input directory."""
        return self.inp_dir

    def get_out_dir(self) -> pathlib.Path:
        """Get output directory."""
        return self.out_dir

    def create_dataframe(self) -> pd.DataFrame:
        """Create Pandas dataframe."""
        df_size = 100
        rng = np.random.default_rng()
        letters = list(string.ascii_lowercase)

        diction_1 = {
            "A": [rng.choice(letters) for i in range(df_size)],
            "B": rng.integers(low=1, high=100, size=df_size),
            "C": rng.normal(0.0, 1.0, size=df_size),
        }

        return pd.DataFrame(diction_1)

    def arrow_func(self) -> None:
        """Convert pandas dataframe to Arrow file format."""
        self.df.to_feather(pathlib.Path(self.inp_dir, self.out_name))

    def csv_func(self) -> None:
        """Convert pandas dataframe to csv file format."""
        self.df.to_csv(pathlib.Path(self.inp_dir, self.out_name), index=False)

    def parquet_func(self) -> None:
        """Convert pandas dataframe to parquet file format."""
        self.df.to_parquet(
            pathlib.Path(self.inp_dir, self.out_name),
            engine="auto",
            compression=None,
        )

    def __call__(self) -> None:
        """To make a class callable."""
        data_ext = {
            ".csv": self.csv_func,
            ".parquet": self.parquet_func,
            ".arrow": self.arrow_func,
        }
        return data_ext[self.file_pattern]()


def test_apply_statistics() -> None:
    """Test applying statistics on PyArrow table."""

    for i in [".parquet", ".csv", ".arrow"]:
        d1 = Generatedata(file_pattern=i, out_name=f"data_1{i}")
        d1()
        table = pa.table(d1.df)
        numeric_table = table.drop(
            [col for col in table.column_names if pa.types.is_string(table[col].type)]
        )
        statistics_list = list(ts.STATS.keys())

        # Test applying each statistic in STATS to the table
        for statistic in statistics_list:
            result_table = ts.apply_statistics(numeric_table, statistics=statistic)

            assert isinstance(result_table, pa.Table)

            for col_name in table.column_names:
                col = table[col_name]
                if not pa.types.is_string(col.type):
                    # Check if the new column with the statistic is present in the result table
                    expected_col_name = f"{col_name}_{statistic}"
                    assert (
                        expected_col_name in result_table.column_names
                    ), f"Column {expected_col_name} not found in result table"


def test_all_statistics() -> None:
    """Test applying all statistics in STATS to the table."""

    for i in [".parquet", ".csv", ".arrow"]:
        d1 = Generatedata(file_pattern=i, out_name=f"data_1{i}")
        d1()
        table = pa.table(d1.df)
        numeric_table = table.drop(
            [col for col in table.column_names if pa.types.is_string(table[col].type)]
        )
        statistics = "all"
        result_table = ts.apply_statistics(numeric_table, statistics=statistics)

        # Check that the result is a PyArrow Table
        assert isinstance(result_table, pa.Table)

        # For each numeric column in the table, check if all statistics are applied
        for col_name in table.column_names:
            col = table[col_name]  # Get the actual column data
            if not pa.types.is_string(col.type):  # Skip string columns
                for stat_name in ts.STATS.keys():
                    # Check if the new column with each statistic is present in the result table
                    expected_col_name = f"{col_name}_{stat_name}"
                    assert (
                        expected_col_name in result_table.column_names
                    ), f"Column {expected_col_name} not found in result table"
