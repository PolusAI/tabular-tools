"""CLI for rt-cetsa-moltprot-tool."""

import json
import logging
import os
import pathlib

import filepattern
import typer
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

    for f in inp_files:
        logger.info(f"Processing file: {f}")
        out_path = gen_out_path(f, out_dir)
        df = fit_data(f)
        df.to_csv(out_path, index=True)


if __name__ == "__main__":
    app()
