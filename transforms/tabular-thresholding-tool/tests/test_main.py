"""Testing of Tabular Thresholding."""
import pathlib
import random
import shutil
import string
import tempfile

import filepattern as fp
import numpy as np
import pandas as pd
import pytest
import pyarrow as pa
import pyarrow.csv as pv
from polus.tabular.transforms.tabular_thresholding import (
    tabular_thresholding as tt,
)


class Generatedata:
    """Generate tabular data with several different file formats."""

    def __init__(self, file_pattern: str, size: int, outname: str) -> None:
        """Define instance attributes."""
        self.dirpath = pathlib.Path.cwd()
        self.inp_dir = tempfile.mkdtemp(dir=self.dirpath)
        self.out_dir = tempfile.mkdtemp(dir=self.dirpath)
        self.file_pattern = file_pattern
        self.size = size
        self.outname = outname
        self.x = self.create_dataframe()

    def get_inp_dir(self) -> pathlib.Path:
        """Get input directory."""
        return pathlib.Path(self.inp_dir)

    def get_out_dir(self) -> pathlib.Path:
        """Get output directory."""
        return pathlib.Path(self.out_dir)

    def create_dataframe(self) -> pd.core.frame.DataFrame:
        """Create Pandas dataframe."""
        diction_1 = {
            "A": list(range(self.size)),
            "B": [random.choice(string.ascii_letters) for _ in range(self.size)],
            "C": np.random.randint(low=1, high=100, size=self.size),
            "D": np.random.normal(0.0, 1.0, size=self.size),
            "MEAN": np.linspace(1.0, 4000.0, self.size),
            "neg_control": [random.choice("01") for _ in range(self.size)],
            "pos_neutral": [random.choice("01") for _ in range(self.size)],
            "plate": ["CD_SOD1_2_E1023886__1" for _ in range(self.size)],
        }

        df = pd.DataFrame(diction_1)
        df["neg_control"] = df["neg_control"].astype(int)
        df["pos_neutral"] = df["pos_neutral"].astype(int)

        return df

    def csv_func(self) -> None:
        """Convert pandas dataframe to csv file format."""
        self.x.to_csv(pathlib.Path(self.inp_dir, self.outname), index=False)

    def parquet_func(self) -> None:
        """Convert pandas dataframe to parquet file format."""
        self.x.to_parquet(
            pathlib.Path(self.inp_dir, self.outname),
            engine="auto",
            compression=None,
        )

    def feather_func(self) -> None:
        """Convert pandas dataframe to feather file format."""
        self.x.to_feather(pathlib.Path(self.inp_dir, self.outname))

    def arrow_func(self) -> None:
        """Convert pandas dataframe to Arrow IPC file format."""
        table = pa.Table.from_pandas(self.x)
        arrow_path = pathlib.Path(self.inp_dir, self.outname)
        with pa.OSFile(str(arrow_path), "wb") as sink:
            with pa.RecordBatchFileWriter(sink, table.schema) as writer:
                writer.write_table(table)

        # Verify that the file is written correctly
        with pa.memory_map(arrow_path, "r") as source:
            try:
                pa.ipc.RecordBatchFileReader(source).read_all()
            except pa.ArrowInvalid:
                raise ValueError(f"The file {arrow_path} is not a valid Arrow file.")

    def __call__(self) -> None:
        """To make a class callable."""
        data_ext = {
            ".csv": self.csv_func,
            ".parquet": self.parquet_func,
            ".feather": self.feather_func,
            ".arrow": self.arrow_func,
        }

        return data_ext[self.file_pattern]()  # No changes here, this is correct

    def clean_directories(self):
        """Remove files."""
        for d in self.dirpath.iterdir():
            if d.is_dir() and d.name.startswith("tmp"):
                shutil.rmtree(d)


# List of extensions to test
EXT = [[".csv", ".feather", ".arrow", ".parquet"]]

@pytest.fixture(params=EXT)
def poly(request):
    """Fixture to get the file extension parameter for testing."""
    return request.param[0]  # Return the extension, e.g., ".csv", not the list

def test_tabular_thresholding(poly):
    """Test the merging of tabular data by rows with equal number of rows."""

    # Generate data with the specified file extension
    d = Generatedata(poly, outname=f"data_1{poly}", size=1000000)
    d()
    pattern = f".*{poly}"   
    fps = fp.FilePattern(d.get_inp_dir(), pattern)
    for file in fps():
        tt.thresholding_func(
            neg_control="neg_control",
            pos_control="pos_neutral",
            var_name="MEAN",
            threshold_type="all",
            false_positive_rate=0.01,
            num_bins=512,
            n=4,
            out_dir=d.get_out_dir(),
            file=file[1][0], 
        )

        # Find the processed file (excluding JSON files)
        file = [f for f in d.get_out_dir().iterdir() if ".json" not in f.name][0]

        if file.suffix == ".arrow":
            with pa.memory_map(str(file), "r") as source:
                table = pa.ipc.RecordBatchFileReader(source).read_all()
        else:
            table = pv.read_csv(file)

        df = table.to_pandas()

        # List of expected threshold methods
        threshold_methods = ["FPR", "OTSU", "NSIGMA"]
        
        # Check if the expected columns are present in the DataFrame
        assert all(item in list(df.columns) for item in threshold_methods)
        
        # Check if the values in the threshold columns are either 0 or 1
        assert np.allclose(np.unique(df[threshold_methods]), [0, 1])

    # Clean up directories after the test
    d.clean_directories()

