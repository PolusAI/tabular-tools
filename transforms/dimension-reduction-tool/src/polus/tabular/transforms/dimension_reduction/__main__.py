"""CLI for the Dimension Reduction tool."""

import logging
import pathlib

import typer
from polus.tabular.transforms.dimension_reduction import POLUS_LOG_LVL

# Initialize the logger
logging.basicConfig(
    format="%(asctime)s - %(name)-8s - %(levelname)-8s - %(message)s",
    datefmt="%d-%b-%y %H:%M:%S",
)
logger = logging.getLogger("polus.tabular.transforms.dimension_reduction")
logger.setLevel(POLUS_LOG_LVL)

app = typer.Typer()


@app.command()
def main(
    inp_dir: pathlib.Path = typer.Option(
        ...,
        "--inpDir",
        help="Input data that needs to be reduced in dimensionality.",
        exists=True,
        file_okay=False,
        dir_okay=True,
        readable=True,
        resolve_path=True,
    ),
    file_pattern: str = typer.Option(
        ".*",
        "--filePattern",
        help="pattern to parse tabular files",
    ),
    out_dir: pathlib.Path = typer.Option(
        ...,
        "--outDir",
        help="Output collection",
        exists=True,
        file_okay=False,
        dir_okay=True,
        writable=True,
        resolve_path=True,
    ),
    preview: bool = typer.Option(
        False,
        "--preview",
        help="Output a JSON preview of outputs produced by this tool",
    ),
) -> None:
    """CLI for the Dimension Reduction tool."""
    logger.info(f"inpDir = {inp_dir}")
    logger.info(f"filePattern = {file_pattern}")
    logger.info(f"outDir = {out_dir}")
    logger.info(f"preview = {preview}")

    pass


if __name__ == "__main__":
    app()
