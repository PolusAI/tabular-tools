"""Dimension Reduction via various methods."""

import logging
import os

POLUS_LOG_LVL = os.environ.get("POLUS_LOG", logging.INFO)
POLUS_TAB_EXT = os.environ.get("POLUS_TAB_EXT", ".arrow")

__version__ = "0.1.0-dev1"
