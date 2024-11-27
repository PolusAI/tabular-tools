"""Tabular Statistics."""
import json
import logging
import os
import pathlib
import time
import pyarrow as pa
import pyarrow.ipc as ipc
import pyarrow.csv as csv

from typing import Any
from typing import Optional
import filepattern as fp
import typer
# import tabular_statistics as ts

from polus.tabular.features.tabular_statistics import tabular_statistics as ts

app = typer.Typer()

# Initialize the logger
logging.basicConfig(
    format="%(asctime)s - %(name)-8s - %(levelname)-8s - %(message)s",
    datefmt="%d-%b-%y %H:%M:%S",
)
logger = logging.getLogger("polus.tabular.features.tabular_statistics")
logger.setLevel(os.environ.get("POLUS_LOG", logging.INFO))


@app.command()
def main(  # noqa: PLR0913
    inp_dir: pathlib.Path = typer.Option(
        ...,
        "--inpDir",
        help="Input generic data collection to be processed by this plugin",
    ),
    file_pattern: str = typer.Option(".+", "--filePattern", help="file_pattern"),
   
    statistics: str = typer.Option(
        ...,
        "--statistics",
        help="Only merge files with the same number of rows?",
    ),
    group_by: str = typer.Option(
        None,
        "--groupBy",
        help="Merge files with common header",
    ),
    out_dir: pathlib.Path = typer.Option(..., "--outDir", help="Output collection"),
    preview: Optional[bool] = typer.Option(
        False,
        "--preview",
        help="Output a JSON preview of files",
    ),
) -> None:
    """CLI for the tool."""
    logger.info(f"inpDir = {inp_dir}")
    logger.info(f"outDir = {out_dir}")
    logger.info(f"filePattern = {file_pattern}")
    logger.info(f"statistics = {statistics}")
    logger.info(f"groupBy= {group_by}")

    start_time = time.time()

    inp_dir = pathlib.Path(inp_dir).resolve()
    out_dir = pathlib.Path(out_dir).resolve()

    assert inp_dir.exists(), f"{inp_dir} doesnot exists!! Please check input path again"
    assert (
        out_dir.exists()
    ), f"{out_dir} doesnot exists!! Please check output path again"

    # By default it ingests all input files if not file_pattern is defined

    EXTS = [".arrow", ".feather", ".csv", ".parquet"]

    file_pattern = ".*" + file_pattern

    fps = fp.FilePattern(inp_dir, file_pattern)

    if preview:
        ts.preview(out_dir, file_pattern)

    else:
        flist = [f for f in pathlib.Path(inp_dir).iterdir() if f.suffix in  EXTS]

        table = ts.load_files(flist)

        columns = table.column_names
        file_column = "intensity_image"

        # Validate required column
        if file_column not in columns:
            raise ValueError(f"Column '{file_column}' not found")
        
        logger.info(f"Merging files into a single table")

        if group_by:
            logger.info(f"Applying statistics by grouping data on: {group_by}")
            group_by = [col.strip() for col in group_by.split(",") if col.strip()] 
            # Apply statistics to the grouped data
            aggregated_df = ts.apply_statistics(table, statistics, group_by)
        else:
            logger.info(f"Applying statistics on data")
            aggregated_df = ts.apply_statistics_to_group(table, statistics)


        ts.save_outputs(aggregated_df, out_dir)
        

    exec_time = time.time() - start_time
    logger.info(f"Execution time: {time.strftime('%H:%M:%S', time.gmtime(exec_time))}")
    logger.info("Finished processing of files!")


if __name__ == "__main__":
    app()