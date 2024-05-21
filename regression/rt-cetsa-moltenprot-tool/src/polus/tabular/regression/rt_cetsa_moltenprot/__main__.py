"""CLI for rt-cetsa-moltprot-tool."""

import json
import logging
import os
import pathlib

import filepattern
import typer
from polus.tabular.regression.rt_cetsa_moltenprot import fit_data

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
    pattern: str = typer.Option(
        ".+",
        "--filePattern",
        help="Pattern to match the files in the input directory.",
    ),
    preview: bool = typer.Option(
        False,
        "--preview",
        help="Preview the files that will be processed.",
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
) -> None:
    """CLI for rt-cetsa-moltprot-tool."""
    logger.info("Starting the CLI for rt-cetsa-moltprot-tool.")

    logger.info(f"Input directory: {inp_dir}")
    logger.info(f"File Pattern: {pattern}")
    logger.info(f"Output directory: {out_dir}")

    if POLUS_TAB_EXT != ".csv":
        msg = "this tool can currently only process csv files."
        raise ValueError(msg)

    fp = filepattern.FilePattern(inp_dir, pattern)
    inp_files = [f[1][0] for f in fp()]

    for f in inp_files:
        if not f.suffix == POLUS_TAB_EXT:
            raise ValueError(
                f"this tool can only process {POLUS_TAB_EXT} files. Got {f}",
            )

    if preview:
        outputs: list[str] = []
        for f in inp_files:
            fit_params_path = f.stem + "_moltenprot_params" + POLUS_TAB_EXT
            fit_curves_path = f.stem + "_moltenprot_curves" + POLUS_TAB_EXT
            outputs = [*outputs, fit_params_path, fit_curves_path]
        out_json = {"files": outputs}
        with (out_dir / "preview.json").open("w") as f:
            json.dump(out_json, f, indent=2)
        return

    for f in inp_files:
        logger.info(f"Processing plate timeserie: {f}")
        fit = fit_data(f)
        fit_params_path = out_dir / (f.stem + "_moltenprot_params" + POLUS_TAB_EXT)
        fit_curves_path = out_dir / (f.stem + "_moltenprot_curves" + POLUS_TAB_EXT)

        # sort fit_params by row/column
        fit_params = fit.plate_results
        fit_params["_index"] = fit_params.index
        fit_params["letter"] = fit_params.apply(lambda row: row._index[:1], axis=1)
        fit_params["number"] = fit_params.apply(
            lambda row: row._index[1:],
            axis=1,
        ).astype(int)
        fit_params = fit_params.drop(columns="_index")
        fit_params = fit_params.sort_values(["letter", "number"])
        fit_params = fit_params.drop(columns=["letter", "number"])
        fit_params.to_csv(fit_params_path, index=True)

        # keep only 2 signicant digits for temperature index
        fit_curves = fit.plate_raw_corr
        fit_curves.index = fit_curves.index.map(lambda t: round(t, 2))
        fit_curves.to_csv(fit_curves_path, index=True)


if __name__ == "__main__":
    app()
