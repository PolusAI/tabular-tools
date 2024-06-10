"""Filepattern Generator Tool."""
import json
import logging
import os
import shutil
import time
from itertools import combinations
from pathlib import Path
from typing import Optional

import filepattern as fp

logger = logging.getLogger(__name__)
logger.setLevel(os.environ.get("POLUS_LOG", logging.INFO))


def generate_preview(
    path: Path,
) -> None:
    """Generate preview of the plugin outputs."""
    source_path = Path(__file__).parents[5].joinpath("example")
    shutil.copytree(source_path, path, dirs_exist_ok=True)


def get_grouping(  # noqa: C901 PLR0912
    inp_dir: Path,
    file_pattern: Optional[str] = None,
    group_by: Optional[str] = None,
    chunk_size: Optional[int] = None,
) -> tuple[str, int]:
    """This function finds the best variable combination for a given chunk size.

    Args:
        inp_dir: Path to Image files
        file_pattern: Regex to parse image files
        group_by: Specify variable to group image filenames
        chunk_size : Number of images to generate collective filepattern
    Returns:
        variables for grouping image filenames, count.
    """
    fps = fp.FilePattern(inp_dir, file_pattern)

    # # Get the number of unique values for each variable
    counts = {k: len(v) for k, v in fps.get_unique_values().items()}

    # Check to see if groupBy already gives a sufficient chunkSize
    best_count = 0
    if group_by is None:
        for k, v in counts.items():
            if v <= chunk_size and v < best_count:  # noqa :SIM114
                best_group, best_count = k, v
            elif best_count == 0:
                best_group, best_count = k, v
        group_by = best_group

    count = 1
    for v in group_by:
        count *= counts[v]
    if count >= chunk_size:
        return group_by, count

    best_group, best_count = group_by, count

    # Search for a combination of `variables` that give a value close to the chunk_size
    variables = [v for v in fps.get_variables() if v not in group_by]
    for i in range(len(variables)):
        groups = {best_group: best_count}
        for p in combinations(variables, i):
            group = group_by + "".join("".join(c) for c in p)
            count = 1
            for v in group:
                count *= counts[v]
            groups[group] = count

        # If all groups are over the chunk_size, then return just return the best_group
        if all(v > chunk_size for _, v in groups.items()):
            return best_group, best_count

        # Find the best_group
        for k, v in groups.items():
            if v > chunk_size:
                continue
            if v > best_count:
                best_group, best_count = k, v
    return best_group, best_count


def save_generator_outputs(x: dict[str, int], out_dir: Path) -> None:
    """Convert filepattern dictionary to JSON file.

    Args:
        x: Dictionary of filepatterns with corresponding parseable image files
        out_dir: Path to save the outputs
    Returns:
        json file with array of file patterns.
    """
    data = json.loads('{"filePatterns": []}')
    with Path.open(Path(out_dir, "file_patterns.json"), "w") as cwlout:
        for key, _ in x.items():
            data["filePatterns"].append(key)
        json.dump(data, cwlout, indent=2)


def generate_patterns(
    inp_dir: Path,
    out_dir: Path,
    file_pattern: Optional[str] = None,
    chunk_size: Optional[int] = None,
    group_by: Optional[str] = None,
) -> None:
    """Generate filepatterns for number of image files.

    Args:
        inp_dir: Path to Image files
        file_pattern: Regex to parse image files
        group_by: Specify variable to group image filenames
        chunk_size : Number of images to generate collective filepattern
        out_dir : Path to output directory.
    """
    starttime = time.time()

    # If the pattern isn't given, try to infer one
    if file_pattern is None:
        try:
            file_pattern = fp.infer_pattern(inp_dir)
        except ValueError:
            logger.error(
                "Could not infer a filepattern from the input files, "
                + "and no filepattern was provided.",
            )
            raise

    logger.info("Finding best grouping...")

    if group_by is None or chunk_size > 0:
        group_by, _ = get_grouping(inp_dir, file_pattern, group_by, chunk_size)

    fps = fp.FilePattern(inp_dir, file_pattern)

    file_count = sum([len(f) for _, f in fps(group_by=group_by)])
    fpss, counts = [], []
    for _, f1 in fps(group_by=group_by):
        images = []
        for _, f2 in f1:
            images.append(str(f2[0]))
        pattern = fp.infer_pattern(files=images)
        fpss.append(pattern)
        counts.append(len(images))

    if sum(counts) != file_count:
        msg = f"Filepattern count do not match the number of files in the {inp_dir} "
        raise ValueError(msg)

    save_generator_outputs(dict(zip(fpss, counts)), out_dir)

    endtime = (time.time() - starttime) / 60
    logger.info(f"Total time taken to process all images: {endtime}")
