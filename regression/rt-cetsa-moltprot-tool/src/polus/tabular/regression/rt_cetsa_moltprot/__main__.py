"""CLI for rt-cetsa-moltprot-tool."""

import json
import logging
import os
import pathlib
import typing

import filepattern
import typer
from polus.tabular.regression.rt_cetsa_moltprot import core
from polus.tabular.regression.rt_cetsa_moltprot import fit_data
from polus.tabular.regression.rt_cetsa_moltprot import gen_out_path

# Initialize the logger
logging.basicConfig(
    format="%(asctime)s - %(name)-8s - %(levelname)-8s - %(message)s",
    datefmt="%d-%b-%y %H:%M:%S",
)
logger = logging.getLogger("polus.tabular.transforms.tabular_merger")
logger.setLevel(os.environ.get("POLUS_LOG", logging.INFO))

app = typer.Typer()


@app.command()
def main(
    inp_dir: pathlib.Path = typer.Option(
        ...,
        help="Input directory containing the data files.",
        exists=True,
        dir_okay=True,
        readable=True,
        resolve_path=True,
    ),
    pattern: str = typer.Option(
        ".+",
        help="Pattern to match the files in the input directory.",
    ),
    preview: bool = typer.Option(
        False,
        help="Preview the files that will be processed.",
    ),
    out_dir: pathlib.Path = typer.Option(
        ...,
        help="Output directory to save the results.",
        exists=True,
        dir_okay=True,
        writable=True,
        resolve_path=True,
    ),
    baseline_fit: typing.Optional[float] = typer.Option(
        None,
        help=core.analysis_defaults["baseline_fit_h"],
    ),
    baseline_bounds: typing.Optional[float] = typer.Option(
        None,
        help=core.analysis_defaults["baseline_bounds_h"],
    ),
    dCp: typing.Optional[float] = typer.Option(
        None,
        help=core.analysis_defaults["dCp_h"],
    ),
    onset_threshold: typing.Optional[float] = typer.Option(
        None,
        help=core.analysis_defaults["onset_threshold_h"],
    ),
    savgol: typing.Optional[float] = typer.Option(
        None,
        help=core.analysis_defaults["savgol_h"],
    ),
    trim_max: typing.Optional[float] = typer.Option(
        None,
        help=core.prep_defaults["trim_max_h"],
    ),
    trim_min: typing.Optional[float] = typer.Option(
        None,
        help=core.prep_defaults["trim_min_h"],
    ),
) -> None:
    """CLI for rt-cetsa-moltprot-tool."""
    # TODO: Add to docs that input csv file should be sorted by `Temperature` column.
    logger.info("Starting the CLI for rt-cetsa-moltprot-tool.")

    logger.info(f"Input directory: {inp_dir}")
    logger.info(f"File Pattern: {pattern}")
    logger.info(f"Output directory: {out_dir}")

    fp = filepattern.FilePattern(inp_dir, pattern)
    inp_files = [f[1][0] for f in fp()]

    if preview:
        out_json = {"files": [gen_out_path(f, out_dir) for f in inp_files]}
        with (out_dir / "preview.json").open("w") as f:
            json.dump(out_json, f, indent=2)
        return

    baseline_fit = (
        baseline_fit
        if baseline_fit is not None
        else core.analysis_defaults["baseline_fit"]
    )
    baseline_bounds = (
        baseline_bounds
        if baseline_bounds is not None
        else core.analysis_defaults["baseline_bounds"]
    )
    dCp = dCp if dCp is not None else core.analysis_defaults["dCp"]
    onset_threshold = (
        onset_threshold
        if onset_threshold is not None
        else core.analysis_defaults["onset_threshold"]
    )
    savgol = savgol if savgol is not None else core.analysis_defaults["savgol"]
    trim_max = trim_max if trim_max is not None else core.prep_defaults["trim_max"]
    trim_min = trim_min if trim_min is not None else core.prep_defaults["trim_min"]
    logger.info(f"Baseline Fit: {baseline_fit}")
    logger.info(f"Baseline Bounds: {baseline_bounds}")
    logger.info(f"dCp: {dCp}")
    logger.info(f"Onset Threshold: {onset_threshold}")
    logger.info(f"Savgol: {savgol}")
    logger.info(f"Trim Max: {trim_max}")
    logger.info(f"Trim Min: {trim_min}")
    params = {
        "baseline_fit": baseline_fit,
        "baseline_bounds": baseline_bounds,
        "dCp": dCp,
        "onset_threshold": onset_threshold,
        "savgol": savgol,
        "trim_max": trim_max,
        "trim_min": trim_min,
    }

    for f in inp_files:
        logger.info(f"Processing file: {f}")
        out_path = gen_out_path(f, out_dir)
        df = fit_data(f, params=params)
        df.to_csv(out_path, index=True)


if __name__ == "__main__":
    app()
