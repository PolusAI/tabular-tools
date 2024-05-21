"""CLI for rt-cetsa-moltprot-tool."""

import json
import logging
import os
import pathlib

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
    int_filename: str = typer.Option(
        None,
        "--intensities",
        help="name of the intensities file (optional).",
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
    preview: bool = typer.Option(
        False,
        "--preview",
        help="Preview the files that will be processed.",
    ),
) -> None:
    """CLI for rt-cetsa-moltprot-tool."""
    logger.info("Starting the CLI for rt-cetsa-moltprot-tool.")

    logger.info(f"Input directory: {inp_dir}")
    logger.info(f"Output directory: {out_dir}")

    # NOTE we may eventually deal with other types.
    if POLUS_TAB_EXT != ".csv":
        msg = "this tool can currently only process csv files."
        raise ValueError(msg)

    if int_filename is not None:
        intensities_file = inp_dir / int_filename
        if not intensities_file.exists():
            raise FileNotFoundError(intensities_file)
    else:
        if len(list(inp_dir.iterdir())) != 1:
            raise FileExistsError(
                f"There should be a single intensities file in {inp_dir}",
            )
        intensities_file = next(inp_dir.iterdir())
    logger.info(f"Using intensities file: {intensities_file}")

    if preview:
        outputs = ["params" + POLUS_TAB_EXT, "values" + POLUS_TAB_EXT]
        out_json = {"files": outputs}
        with (out_dir / "preview.json").open("w") as f:
            json.dump(out_json, f, indent=2)
        return

    fit = fit_data(intensities_file)

    fit_params_path = out_dir / ("params" + POLUS_TAB_EXT)
    fit_curves_path = out_dir / ("values" + POLUS_TAB_EXT)

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