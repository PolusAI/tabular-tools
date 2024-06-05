"""CLI for rt-cetsa-analysis tool."""

import json
import logging
import os
from pathlib import Path

import typer
from polus.tabular.regression.rt_cetsa_analysis.preprocess_data import preprocess_data
from polus.tabular.regression.rt_cetsa_analysis.run_new_rscript import run_rscript

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
    platemap_path: Path = typer.Option(
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
    logger.info(f"platemap path: {platemap_path}")
    logger.info(f"Output directory: {out_dir}")

    if params_filename:
        params_path = inp_dir / params_filename
    elif (inp_dir / "params.csv").exists():
        params_path = inp_dir / "params.csv"
    else:
        raise ValueError(
            f"No 'params.csv' moltenprot parameters file found in {inp_dir}.",
        )

    if values_filename:
        values_path = inp_dir / values_filename
    elif (inp_dir / "values.csv").exists():
        values_path = inp_dir / "values.csv"
    else:
        raise ValueError(f"No 'values.csv' moltenprot values file found in {inp_dir}.")

    if not params_path.exists():
        raise FileNotFoundError(f"params file not found : {params_path}")
    if not values_path.exists():
        raise FileNotFoundError(f"values file not found : {values_path}")

    data_path = preprocess_data(platemap_path, values_path, params_path, out_dir)

    logger.info(f"combined data csv file: {data_path}")

    if preview:
        outputs: list[str] = ["signif_df.csv"]
        out_json = {"files": outputs}
        with (out_dir / "preview.json").open("w") as f:
            json.dump(out_json, f, indent=2)
        return

    run_rscript(data_path, out_dir)


if __name__ == "__main__":
    app()
