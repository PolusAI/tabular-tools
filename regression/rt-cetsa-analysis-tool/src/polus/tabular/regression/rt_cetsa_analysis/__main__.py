"""CLI for rt-cetsa-moltprot-tool."""

import logging
import os
from pathlib import Path

import typer
from polus.tabular.regression.rt_cetsa_analysis.run_rscript import run_rscript

# get env
POLUS_LOG = os.environ.get("POLUS_LOG", logging.INFO)
POLUS_TAB_EXT = os.environ.get("POLUS_TAB_EXT", ".csv")

# Initialize the logger
logging.basicConfig(
    format="%(asctime)s - %(name)-8s - %(levelname)-8s - %(message)s",
    datefmt="%d-%b-%y %H:%M:%S",
)
logger = logging.getLogger("rt_cetsa_analysis")
logger.setLevel(POLUS_LOG)

app = typer.Typer()


@app.command()
def main(
    inp_dir: Path = typer.Option(
        ...,
        "--inpDir",
        help="Input directory containing the all data files.",
        exists=True,
        dir_okay=True,
        readable=True,
        resolve_path=True,
    ),
    params_pattern: str = typer.Option(
        ...,
        "--params",
        help="name of the molten fit params csv file in the input directory.",
    ),
    values_pattern: str = typer.Option(
        ...,
        "--values",
        help="name of the baseline corrected values csv file in the input directory.",
    ),
    platemap: Path = typer.Option(
        ...,
        "--platemap",
        help="Path to the platemap file.",
        exists=True,
        readable=True,
        resolve_path=True,
    ),
    preview: bool = typer.Option(
        False,
        "--preview",
        help="Preview the files that will be processed.",
    ),
    out_dir: Path = typer.Option(
        ...,
        "--outDir",
        help="Output directory to save the results.",
        exists=True,
        dir_okay=True,
        writable=True,
        resolve_path=True,
    ),
) -> None:
    """CLI for rt-cetsa-moltprot-tool."""
    # TODO: Add to docs that input csv file should be sorted by `Temperature` column.
    logger.info("Starting the CLI for rt-cetsa-moltenprot-tool.")

    logger.info(f"Input directory: {inp_dir}")
    logger.info(f"params_pattern: {params_pattern}")
    logger.info(f"values_pattern: {values_pattern}")
    logger.info(f"platemap path: {platemap}")
    logger.info(f"Output directory: {out_dir}")

    params = inp_dir / params_pattern
    values = inp_dir / values_pattern

    logger.info(f"{inp_dir}")

    if preview:
        NotImplemented  # noqa:  B018

    if not params.exists():
        raise FileNotFoundError(f"params file not found : {params}")
    if not values.exists():
        raise FileNotFoundError(f"values file not found : {values}")
    if not platemap.exists():
        raise FileNotFoundError(f"platemap file not found : {platemap}")

    run_rscript(params, values, platemap, out_dir)


if __name__ == "__main__":
    app()
