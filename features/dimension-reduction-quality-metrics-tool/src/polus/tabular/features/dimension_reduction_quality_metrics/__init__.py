"""Dimension Reduction Quality Metrics."""

import logging
import os
import pathlib

import numpy
from polus.tabular.transforms.dimension_reduction import Formats

from . import metrics

POLUS_LOG_LVL = os.environ.get("POLUS_LOG", logging.INFO)
POLUS_TAB_EXT = os.environ.get("POLUS_TAB_EXT", ".feather")

__version__ = "0.1.0-dev1"


def measure_quality(
    original_path: pathlib.Path,
    embedded_path: pathlib.Path,
    num_queries: int,
    ks: list[int],
    distance_metrics: list[str],
    quality_metrics: list[str],
) -> dict[int, dict[str, dict[str, float]]]:
    """Measure the quality of the dimension reduction using different metrics.

    Args:
        original_path: The path to the original data.
        embedded_path: The path to the embedded data.
        num_queries: The number of queries to use.
        ks: The numbers of nearest neighbors to consider.
        distance_metrics: The distance metrics to use.
        quality_metrics: The quality metrics to compute.

    Returns:
        A dictionary containing the computed metrics. The format is:
        {
            k_1: {
                distance_metric_1: {
                    quality_metric_1: value,
                    quality_metric_2: value,
                },
                distance_metric_2: {
                    quality_metric_1: value,
                    quality_metric_2: value,
                },
            },
            k_2: {
                distance_metric_1: {
                    quality_metric_1: value,
                    quality_metric_2: value,
                },
                distance_metric_2: {
                    quality_metric_1: value,
                    quality_metric_2: value,
                },
            },
        }
    """
    original_data = Formats.read(original_path)
    embedded_data = Formats.read(embedded_path)

    rng = numpy.random.default_rng()
    query_indices = rng.choice(
        original_data.shape[0],
        size=num_queries,
        replace=False,
    )

    quality: dict[int, dict[str, dict[str, float]]] = {}
    for k in ks:
        quality[k] = {}
        for distance_metric in distance_metrics:
            quality[k][distance_metric] = {}
            for quality_metric in quality_metrics:
                metric_func = getattr(metrics, quality_metric)
                quality[k][distance_metric][quality_metric] = metric_func(
                    original_data=original_data,
                    embedded_data=embedded_data,
                    query_indices=query_indices,
                    n_neighbors=k,
                    distance_metric=distance_metric,
                )

    return quality


__all__ = [
    "measure_quality",
    "POLUS_LOG_LVL",
    "POLUS_TAB_EXT",
    "__version__",
]
