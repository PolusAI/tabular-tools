"""Testing Tabular Merger."""
import pathlib
import string
import typing

import filepattern as fp
import numpy as np
import pandas as pd
import pytest
import vaex
from polus.images.transforms.tabular.tabular_merger import tabular_merger as tm


class Generatedata:
    """Generate tabular data with several different file format."""

    def __init__(
        self,
        file_pattern: str,
        out_name: str,
        same_rows: typing.Optional[bool],
        trunc_columns: typing.Optional[bool],
    ) -> None:
        """Define instance attributes."""
        self.data_dir = pathlib.Path.cwd().parent.joinpath("data")

        self.inp_dir = pathlib.Path(self.data_dir, "input")
        if not self.inp_dir.exists():
            self.inp_dir.mkdir(exist_ok=True, parents=True)

        self.out_dir = pathlib.Path(self.data_dir, "output")
        if not self.out_dir.exists():
            self.out_dir.mkdir(exist_ok=True, parents=True)

        self.file_pattern = file_pattern
        self.same_rows = same_rows
        self.trunc_columns = trunc_columns
        self.out_name = out_name
        self.df = self.create_dataframe()

    def get_inp_dir(self) -> pathlib.Path:
        """Get input directory."""
        return self.inp_dir

    def get_out_dir(self) -> pathlib.Path:
        """Get output directory."""
        return self.out_dir

    def create_dataframe(self) -> pd.DataFrame:
        """Create Pandas dataframe."""
        df_size = 100 if self.same_rows else 200
        rng = np.random.default_rng()
        letters = list(string.ascii_lowercase)

        diction_1 = {
            "A": list(range(df_size)),
            "B": [rng.choice(letters) for i in range(df_size)],
            "C": rng.integers(low=1, high=100, size=df_size),
            "D": rng.normal(0.0, 1.0, size=df_size),
        }

        if self.trunc_columns:
            diction_1 = {k: v for k, v in diction_1.items() if k not in ["A", "B"]}

        return pd.DataFrame(diction_1)

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

    def feather_func(self) -> None:
        """Convert pandas dataframe to feather file format."""
        self.df.to_feather(pathlib.Path(self.inp_dir, self.out_name))

    def arrow_func(self) -> None:
        """Convert pandas dataframe to Arrow file format."""
        self.df.to_feather(pathlib.Path(self.inp_dir, self.out_name))

    def hdf_func(self) -> None:
        """Convert pandas dataframe to hdf5 file format."""
        v_df = vaex.from_pandas(self.df, copy_index=False)
        v_df.export(pathlib.Path(self.inp_dir, self.out_name))

    def __call__(self) -> None:
        """To make a class callable."""
        data_ext = {
            ".hdf5": self.hdf_func,
            ".csv": self.csv_func,
            ".parquet": self.parquet_func,
            ".feather": self.feather_func,
            ".arrow": self.arrow_func,
        }

        return data_ext[self.file_pattern]()

    def clean_directories(self) -> None:
        """Remove files."""
        for f in self.get_out_dir().iterdir():
            f.unlink()
        for f in self.get_inp_dir().iterdir():
            f.unlink()


FILE_EXT = [[".hdf5", ".parquet", ".csv", ".feather", ".arrow"]]


@pytest.fixture(params=FILE_EXT)
def poly(request: pytest.FixtureRequest) -> list[str]:
    """To get the parameter of the fixture."""
    return request.param


def test_mergingfiles_row_wise_samerows(poly: list[str]) -> None:
    """Testing of merging of tabular data by rows with equal number of rows."""
    for i in poly:
        d1 = Generatedata(i, out_name=f"data_1{i}", same_rows=True, trunc_columns=False)
        d2 = Generatedata(i, out_name=f"data_2{i}", same_rows=True, trunc_columns=False)
        d3 = Generatedata(i, out_name=f"data_3{i}", same_rows=True, trunc_columns=False)
        d1()
        d2()
        d3()
        pattern = f".*{i}"
        fps = fp.FilePattern(d1.get_inp_dir(), pattern)
        inp_dir_files = [f[1][0] for f in fps()]
        tm.merge_files(
            inp_dir_files,
            strip_extension=True,
            dim="rows",
            same_rows=True,
            same_columns=False,
            map_var="A",
            out_dir=d1.get_out_dir(),
        )

        outfile = [f for f in d1.get_out_dir().iterdir() if f.suffix == ".arrow"][0]
        merged = vaex.open(outfile)
        assert len(merged["file"].unique()) == 3
        d1.clean_directories()


def test_mergingfiles_row_wise_unequalrows(poly: list[str]) -> None:
    """Testing of merging of tabular data by rows with unequal number of rows."""
    for i in poly:
        d1 = Generatedata(i, out_name=f"data_1{i}", same_rows=True, trunc_columns=False)
        d2 = Generatedata(
            i,
            out_name=f"data_2{i}",
            same_rows=False,
            trunc_columns=False,
        )
        d3 = Generatedata(
            i,
            out_name=f"data_3{i}",
            same_rows=False,
            trunc_columns=False,
        )
        d1()
        d2()
        d3()
        pattern = f".*{i}"
        fps = fp.FilePattern(d1.get_inp_dir(), pattern)
        inp_dir_files = [f[1][0] for f in fps()]
        tm.merge_files(
            inp_dir_files,
            strip_extension=True,
            dim="rows",
            same_rows=True,
            same_columns=False,
            map_var="A",
            out_dir=d1.get_out_dir(),
        )
        outfile = [f for f in d1.get_out_dir().iterdir() if f.suffix == ".arrow"][0]
        merged = vaex.open(outfile)
        assert len(merged["file"].unique()) == 3
        assert merged.shape[0] > 300
        d1.clean_directories()


def test_mergingfiles_column_wise_equalrows(poly: list[str]) -> None:
    """Testing of merging of tabular data by columns with equal number of rows."""
    for i in poly:
        d1 = Generatedata(i, out_name=f"data_1{i}", same_rows=True, trunc_columns=False)
        d2 = Generatedata(i, out_name=f"data_2{i}", same_rows=True, trunc_columns=False)
        d3 = Generatedata(i, out_name=f"data_3{i}", same_rows=True, trunc_columns=False)
        d1()
        d2()
        d3()
        pattern = f".*{i}"
        fps = fp.FilePattern(d1.get_inp_dir(), pattern)
        inp_dir_files = [f[1][0] for f in fps()]
        tm.merge_files(
            inp_dir_files,
            strip_extension=True,
            dim="columns",
            same_rows=True,
            same_columns=False,
            map_var="A",
            out_dir=d1.get_out_dir(),
        )
        outfile = [f for f in d1.get_out_dir().iterdir() if f.suffix == ".arrow"][0]
        merged = vaex.open(outfile)
        assert len(merged.get_column_names()) == 12
        assert merged.shape[0] == 100
        d1.clean_directories()


def test_mergingfiles_column_wise_unequalrows(poly: list[str]) -> None:
    """Testing of merging of tabular data by columns with unequal number of rows."""
    for i in poly:
        d1 = Generatedata(i, out_name=f"data_1{i}", same_rows=True, trunc_columns=False)
        d2 = Generatedata(i, out_name=f"data_2{i}", same_rows=True, trunc_columns=False)
        d3 = Generatedata(
            i,
            out_name=f"data_3{i}",
            same_rows=False,
            trunc_columns=False,
        )
        d1()
        d2()
        d3()
        pattern = f".*{i}"
        fps = fp.FilePattern(d1.get_inp_dir(), pattern)
        inp_dir_files = [f[1][0] for f in fps()]
        tm.merge_files(
            inp_dir_files,
            strip_extension=True,
            dim="columns",
            same_rows=False,
            same_columns=False,
            map_var="A",
            out_dir=d1.get_out_dir(),
        )
        outfile = [f for f in d1.get_out_dir().iterdir() if f.suffix == ".arrow"][0]
        merged = vaex.open(outfile)
        assert len(merged.get_column_names()) == 13
        assert "indexcolumn" in merged.get_column_names()
        assert merged.shape[0] == 200
        d1.clean_directories()
