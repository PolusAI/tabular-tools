"""Helpers for reading and writing data."""

import enum
import pathlib

import numpy
import pandas


class Formats(str, enum.Enum):
    """The data formats supported by this tool."""

    CSV = "csv"
    PARQUET = "parquet"
    FEATHER = "feather"
    NPY = "npy"

    @staticmethod
    def read(path: pathlib.Path) -> numpy.ndarray:
        """Read the data from the specified path."""
        # Read the extension of the file
        ext = path.suffix

        data: numpy.ndarray
        if ext == ".csv":
            data = pandas.read_csv(path).to_numpy(dtype=numpy.float32)
        elif ext == ".parquet":
            data = pandas.read_parquet(path).to_numpy(dtype=numpy.float32)
        elif ext == ".feather":
            data = pandas.read_feather(path).to_numpy(dtype=numpy.float32)
        elif ext == ".npy":
            data = numpy.load(path)
            data = data.astype(numpy.float32)
        else:
            allowed_formats = ", ".join(Formats.__members__.keys())
            msg = f"Unsupported file format: {ext}. Must be one of: {allowed_formats}"
            raise ValueError(msg)

    @staticmethod
    def write(data: numpy.ndarray, path: pathlib.Path) -> None:
        """Write the data to the specified path."""
        # Write the extension of the file
        ext = path.suffix

        if ext == ".csv":
            pandas.DataFrame(data).to_csv(path, index=False)
        elif ext == ".parquet":
            pandas.DataFrame(data).to_parquet(path, index=False)
        elif ext == ".feather":
            pandas.DataFrame(data).to_feather(path)
        elif ext == ".npy":
            numpy.save(path, data)
        else:
            allowed_formats = ", ".join(Formats.__members__.keys())
            msg = f"Unsupported file format: {ext}. Must be one of: {allowed_formats}"
            raise ValueError(msg)


__all__ = ["Formats"]
