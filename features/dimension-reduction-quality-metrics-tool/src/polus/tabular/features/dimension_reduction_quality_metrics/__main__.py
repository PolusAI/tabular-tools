"""CLI for the Dimension Reduction tool."""

import json
import logging
import pathlib

import filepattern
import tqdm
import typer
from polus.tabular.features.dimension_reduction_quality_metrics import measure_quality
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
    original_dir: pathlib.Path = typer.Option(
        ...,
        "--originalDir",
        help="Directory containing the original data",
        exists=True,
        file_okay=False,
        dir_okay=True,
        readable=True,
        resolve_path=True,
    ),
    original_pattern: str = typer.Option(
        ".*",
        "--originalPattern",
        help="pattern to parse tabular files for the original data",
    ),
    embedded_dir: pathlib.Path = typer.Option(
        ...,
        "--embeddedDir",
        help="Directory containing the embedded data",
        exists=True,
        file_okay=False,
        dir_okay=True,
        readable=True,
        resolve_path=True,
    ),
    embedded_pattern: str = typer.Option(
        ".*",
        "--embeddedPattern",
        help="pattern to parse tabular files for the embedded data",
    ),
    num_queries: int = typer.Option(
        1000,
        "--numQueries",
        help="Number of queries to use for the quality metrics",
    ),
    ks: str = typer.Option(
        "10,100",
        "--ks",
        help="Comma-separated list of numbers of nearest neighbors to consider",
    ),
    distance_metrics: str = typer.Option(
        "euclidean,cosine",
        "--distanceMetrics",
        help="Comma-separated list of distance metrics to use",
    ),
    quality_metrics: str = typer.Option(
        "fnn",
        "--qualityMetrics",
        help="Comma-separated list of quality metrics to compute",
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
    logger.info(f"originalDir = {original_dir}")
    logger.info(f"originalPattern = {original_pattern}")
    logger.info(f"embeddedDir = {embedded_dir}")
    logger.info(f"embeddedPattern = {embedded_pattern}")
    logger.info(f"numQueries = {num_queries}")
    logger.info(f"ks = {ks}")
    logger.info(f"distanceMetrics = {distance_metrics}")
    logger.info(f"qualityMetrics = {quality_metrics}")
    logger.info(f"outDir = {out_dir}")
    logger.info(f"preview = {preview}")

    original_fp = filepattern.FilePattern(original_dir, original_pattern)
    original_files = [pathlib.Path(p) for _, [p] in original_fp()]
    original_dict = {f.stem: f for f in original_files}

    embedded_fp = filepattern.FilePattern(embedded_dir, embedded_pattern)
    embedded_files = [pathlib.Path(p) for _, [p] in embedded_fp()]
    embedded_dict = {f.stem: f for f in embedded_files}

    data_pairs: dict[str, tuple[pathlib.Path, pathlib.Path]] = {}
    for stem in original_dict:
        if stem in embedded_dict:
            data_pairs[stem] = (original_dict[stem], embedded_dict[stem])
        else:
            logger.warning(f"No matching embedded file found for {stem}")
    for stem in embedded_dict:
        if stem not in original_dict:
            logger.warning(f"No matching original file found for {stem}")

    if preview:
        logger.info(f"Previewing {len(data_pairs)} pairs of data")
        msg = "Not implemented yet"
        raise NotImplementedError(msg)

    for original_path, embedded_path in tqdm.tqdm(
        data_pairs.values(),
        total=len(data_pairs),
    ):
        out_path = out_dir / f"{original_path.stem}.json"
        quality_metrics = measure_quality(
            original_path=original_path,
            embedded_path=embedded_path,
            num_queries=num_queries,
            ks=ks,
            distance_metrics=distance_metrics,
            quality_metrics=quality_metrics,
        )
        with out_path.open("w") as f:
            json.dump(quality_metrics, f)


if __name__ == "__main__":
    app()
