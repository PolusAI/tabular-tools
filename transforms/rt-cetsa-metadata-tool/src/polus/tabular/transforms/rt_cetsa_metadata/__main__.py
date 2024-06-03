"""CLI for rt-cetsa-metadata-tool."""

import logging
import os
import pathlib

import typer
from polus.tabular.transforms.rt_cetsa_metadata import preprocess_metadata

# Initialize the logger
logging.basicConfig(
    format="%(asctime)s - %(name)-8s - %(levelname)-8s - %(message)s",
    datefmt="%d-%b-%y %H:%M:%S",
)
logger = logging.getLogger(__file__)
logger.setLevel(os.environ.get("POLUS_LOG", logging.INFO))

POLUS_TAB_EXT = os.environ.get("POLUS_TAB_EXT", ".csv")

app = typer.Typer()


@app.command()
def main(
    inp_dir: pathlib.Path = typer.Option(
        ...,
        "--inpDir",
        help="Input directory containing the data files.",
        exists=True,
        dir_okay=True,
        readable=True,
        resolve_path=True,
    ),
    metadata: str = typer.Option(
        "CameraData.xlsx",
        "--metadata",
        help="metadata file for this dataset. (default to CameraData.xlsx)",
    ),
    out_dir: pathlib.Path = typer.Option(
        ...,
        "--outDir",
        help="Output directory to save the results.",
        exists=True,
        dir_okay=True,
        writable=True,
        resolve_path=True,
    ),
    preview: bool = typer.Option(  # noqa ARG001
        False,
        "--preview",
        help="Preview the files that will be processed.",
    ),
) -> None:
    """CLI for rt-cetsa-metadata-tool."""
    logger.info("Starting the CLI for rt-cetsa-metadata-tool.")

    logger.info(f"Input directory: {inp_dir}")
    logger.info(f"Output directory: {out_dir}")

    # NOTE we may eventually deal with other types.
    if POLUS_TAB_EXT != ".csv":
        msg = "this tool can currently only process csv files."
        raise ValueError(msg)

    metadata_file = inp_dir / metadata
    if not metadata_file.exists():
        raise FileNotFoundError(metadata_file)
    logger.info(f"Using metadata file: {metadata_file}")

    preprocess_metadata(metadata_file, inp_dir, out_dir)


if __name__ == "__main__":
    app()
