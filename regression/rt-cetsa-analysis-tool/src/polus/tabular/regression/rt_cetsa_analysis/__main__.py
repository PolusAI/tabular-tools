"""CLI for rt-cetsa-analysis tool."""

import json
import logging
import os
from pathlib import Path

import typer
from polus.tabular.regression.rt_cetsa_analysis.preprocess_data import (
    preprocess_platemap,
)
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
    params_filename: str = typer.Option(
        None,
        "--params",
        help="name of the moltenprot fit params csv file in the input directory.",
    ),
    values_filename: str = typer.Option(
        None,
        "--values",
        help="name of the moltenprot baseline corrected values csv file in the input directory.",
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
    """CLI for rt-cetsa-analysis tool."""
    logger.info("Starting the CLI for rt-cetsa-analysis tool.")

    logger.info(f"Input directory: {inp_dir}")
    logger.info(f"params: {params_filename}")
    logger.info(f"values: {values_filename}")
    logger.info(f"platemap path: {platemap}")
    logger.info(f"Output directory: {out_dir}")

    if params_filename:
        params = inp_dir / params_filename
    elif (inp_dir / "params.csv").exists():
        params = inp_dir / "params.csv"
    else:
        raise ValueError(
            f"No 'params.csv' moltenprot parameters file found in {inp_dir}.",
        )

    if values_filename:
        values = inp_dir / values_filename
    elif (inp_dir / "values.csv").exists():
        values = inp_dir / "values.csv"
    else:
        raise ValueError(f"No 'values.csv' moltenprot values file found in {inp_dir}.")

    if not params.exists():
        raise FileNotFoundError(f"params file not found : {params}")
    if not values.exists():
        raise FileNotFoundError(f"values file not found : {values}")

    processed_platemap = preprocess_platemap(platemap, out_dir)

    logger.info(f"params filename: {params}")
    logger.info(f"values filename: {values}")
    logger.info(f"processed platemap path: {processed_platemap}")
    logger.info(f"Output directory: {out_dir}")

    if preview:
        outputs: list[str] = ["signif_df.csv"]
        out_json = {"files": outputs}
        with (out_dir / "preview.json").open("w") as f:
            json.dump(out_json, f, indent=2)
        return

    run_rscript(params, values, processed_platemap, out_dir)


if __name__ == "__main__":
    app()
