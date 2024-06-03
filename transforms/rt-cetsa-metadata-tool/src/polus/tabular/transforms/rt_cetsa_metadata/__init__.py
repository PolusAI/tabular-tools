"""RT_CETSA Metadata Preprocessing Tool."""

__version__ = "0.3.0-dev0"

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


def preprocess_metadata(metadata_file: Path, inp_dir: Path, out_dir: Path):
    """Preprocess images and metadata."""
    df = pd.read_excel(metadata_file)
    temp_col_name = "Current Temp (°C)"

    # create a images subdirectory
    images_path = Path(out_dir / "images")
    images_path.mkdir(parents=False, exist_ok=True)

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
        new_file = str(index + 1) + ".tif"
        shutil.copyfile(inp_dir / old_file, images_path / new_file)
        new_files.append(new_file)

    metadata_df = pd.DataFrame({"Temperature": temps, "FileName": new_files})
    metadata_df.to_csv(out_dir / "metadata.csv", index=True)
