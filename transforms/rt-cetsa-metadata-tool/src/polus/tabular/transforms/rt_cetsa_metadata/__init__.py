"""RT_CETSA Metadata Preprocessing Tool."""

__version__ = "0.5.0-dev0"

import logging
import os
import shutil
import warnings
from pathlib import Path
from pathlib import PureWindowsPath

import pandas as pd

# Suppress FutureWarning messages coming from pandas
warnings.simplefilter(action="ignore", category=FutureWarning)

logger = logging.getLogger(__file__)
logger.setLevel(os.environ.get("POLUS_LOG", logging.INFO))
POLUS_TAB_EXT = os.environ.get("POLUS_TAB_EXT", ".csv")


def preprocess_metadata(inp_dir: Path, out_dir: Path, metadata_file: Path):
    """Preprocess images and metadata."""
    df = pd.read_excel(metadata_file)
    temp_col_name = "Current Temp (°C)"

    # create a images subdirectory
    img_out_dir = Path(out_dir / "images")
    img_out_dir.mkdir(parents=False, exist_ok=True)

    new_files: list[str] = []
    temps: list[float] = []
    previous_temp = 0
    for index, row in df.iterrows():
        old_file = PureWindowsPath(Path(row["Filename"])).name
        current_temp = row[temp_col_name]
        if current_temp < previous_temp:
            logger.warning(
                f"{current_temp} < {temps[-1]}. Check your metadata file. Ignore row : {index}.",
            )
            continue
        temps.append(current_temp)
        previous_temp = current_temp
        # copy and renamed the images
        new_file = str(index + 1) + "_" + str(current_temp) + ".tif"
        shutil.copyfile(inp_dir / old_file, img_out_dir / new_file)
        new_files.append(new_file)

    metadata_df = pd.DataFrame({"Temperature": temps, "FileName": new_files})
    metadata_df.to_csv(img_out_dir / "metadata.csv", index=True)

    return img_out_dir


def preprocess_from_range(inp_dir: Path, out_dir: Path, range_t: tuple[float, float]):
    """Preprocess images and metadata."""
    # create a images subdirectory
    img_out_dir = Path(out_dir / "images")
    img_out_dir.mkdir(parents=False, exist_ok=True)

    new_files: list[str] = []
    temps: list[float] = []

    # get tif images
    images = [
        file
        for file in inp_dir.iterdir()
        if file.is_file() and (file.suffix == ".tif" or file.suffix == ".tiff")
    ]

    # if image names are integers, use numerical index
    numerical_indexing = False
    try:
        images = sorted(images, key=lambda img: int(img.stem))
        numerical_indexing = True
    except ValueError:
        images = sorted(images)

    logger.debug(f"sort base on numerical index : {numerical_indexing}")

    for image in images:
        logger.debug(f"{image}")

    # determine temperature metadata for each image.
    image_count = len(images)
    if image_count == 0:
        raise ValueError(f"no tif images in {inp_dir}")
    if image_count == 1:
        raise ValueError(
            f"only one image found in {inp_dir}. Unable to interpolate on range {range_t}.",
        )
    min_t, max_t = range_t
    temps = [
        round(min_t + (index / (image_count - 1)) * (max_t - min_t), 2)
        for index in range(image_count)
    ]

    # save images
    for index, (image, temp) in enumerate(zip(images, temps)):
        new_file = str(index + 1) + "_" + str(temp) + ".tif"
        shutil.copyfile(image, img_out_dir / new_file)
        new_files.append(new_file)

    # save corresponding metadata file
    metadata_df = pd.DataFrame({"Temperature": temps, "FileName": new_files})
    metadata_df.to_csv(img_out_dir / "metadata.csv", index=True)

    return img_out_dir
