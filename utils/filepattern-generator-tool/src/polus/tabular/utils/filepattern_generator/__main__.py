"""Filepattern Generator Tool."""

import logging
from pathlib import Path
from typing import Optional

import polus.tabular.utils.filepattern_generator.filepattern_generator as fg
import typer

app = typer.Typer()

# Initialize the logger
logging.basicConfig(
    format="%(asctime)s - %(name)-8s - %(levelname)-8s - %(message)s",
    datefmt="%d-%b-%y %H:%M:%S",
)
logger = logging.getLogger("polus.tabular.utils.filepattern_generator")


@app.command()
def main(  # noqa: PLR0913
    inp_dir: Path = typer.Option(
        ...,
        "--inpDir",
        "-i",
        help="Input image collection to be processed by this plugin",
    ),
    file_pattern: Optional[str] = typer.Option(
        None,
        "--filePattern",
        "-f",
        help="Filepattern regex used to parse image files",
    ),
    chunk_size: Optional[int] = typer.Option(
        0,
        "--chunkSize",
        "-c",
        help="Select chunksize for generating filepatterns from collective image set",
    ),
    group_by: Optional[str] = typer.Option(
        None,
        "--groupBy",
        "-g",
        help="Select a parameter to generate Filepatterns in specific order",
    ),
    out_dir: Path = typer.Option(
        ...,
        "--outDir",
        "-o",
        help="Path to download XML files",
    ),
    preview: Optional[bool] = typer.Option(
        False,
        "--preview",
        help="Output a JSON preview of files",
    ),
) -> None:
    """Scaled Nyxus plugin allows to extract features from labelled images."""
    logger.info(f"--inpDir = {inp_dir}")
    logger.info(f"--filePattern = {file_pattern}")
    logger.info(f"--chunkSize = {chunk_size}")
    logger.info(f"--groupBy = {group_by}")
    logger.info(f"--outDir = {out_dir}")

    out_dir = out_dir.resolve()

    if not out_dir.exists():
        out_dir.mkdir(exist_ok=True)

    if not inp_dir.exists():
        msg = f"inpDir '{inp_dir}' does not exist."
        raise FileNotFoundError(msg)

    if not preview:
        fg.generate_patterns(inp_dir, out_dir, file_pattern, chunk_size, group_by)

    if preview:
        fg.generate_preview(out_dir)
        logger.info(f"generating preview data in {out_dir}")


if __name__ == "__main__":
    app()
