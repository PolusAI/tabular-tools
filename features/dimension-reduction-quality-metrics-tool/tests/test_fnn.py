"""Tests for the knn-search module."""

import numpy
import pytest
import sklearn.datasets

from polus.tabular.features.dimension_reduction_quality_metrics.metrics.fnn import fnn
from polus.tabular.features.dimension_reduction_quality_metrics.metrics.fnn import (
    knn_search,
)
from polus.tabular.transforms.dimension_reduction.algorithms import umap


def test_knn_search():
    """Tests for knn-search."""

    data = numpy.asarray(
        [[i, i, i] for i in range(10)],
        dtype=numpy.float32,
    )
    queries = data[:2, :]

    assert data.shape[1] == queries.shape[1]

    k = 2
    metric = "euclidean"
    dists, indices = knn_search(data, queries, k, metric)

    assert dists.shape == (queries.shape[0], k)
    assert indices.shape == (queries.shape[0], k)

    expected_dists = numpy.sqrt(
        numpy.asarray(
            [[0.0, 3.0], [0.0, 3.0]],
            dtype=numpy.float32,
        )
    )
    numpy.testing.assert_allclose(dists, expected_dists)

    expected_indices = numpy.asarray(
        [[0, 1], [1, 0]],
        dtype=numpy.int32,
    )
    numpy.testing.assert_array_equal(indices, expected_indices)


def gen_data(metric: str) -> tuple[numpy.ndarray, numpy.ndarray]:
    digits = sklearn.datasets.load_digits()
    original_data: numpy.ndarray = digits.data
    embedded_data = umap.reduce(
        data=original_data,
        n_components=3,
        n_neighbors=15,
        metric=metric,
    )
    return original_data, embedded_data


@pytest.mark.parametrize("metric", ["euclidean", "cosine"])
def test_fnn(metric: str):
    """Tests for False Nearest Neighbors (FNN)."""

    original_data, embedded_data = gen_data(metric)
    for num_queries in [10, 100, 200]:
        rng = numpy.random.default_rng()
        query_indices = rng.choice(
            original_data.shape[0],
            size=num_queries,
            replace=False,
        )
        for k in [10, 100]:
            fnn_metric = fnn(
                original_data=original_data,
                embedded_data=embedded_data,
                query_indices=query_indices,
                n_neighbors=k,
                distance_metric=metric,
            )

            msg = f"metric: {metric}, k: {k}, num_queries: {num_queries}"
            assert 0.0 <= fnn_metric <= 1.0, f"FNN: {fnn_metric:.6f}, {msg}"
            expected_fnn = expected_failure_threshold(
                num_queries=num_queries,
                k=k,
                metric=metric,
            )
            assert (
                fnn_metric >= expected_fnn
            ), f"FNN: {fnn_metric:.6f} < {expected_fnn:.6f}, {msg}"


def expected_failure_threshold(
    num_queries: int,
    k: int,
    metric: str,
) -> float:
    threshold = None

    # These thresholds are based on the averages of several measurements
    if metric == "euclidean":
        if k == 10:
            if num_queries == 10:
                threshold = 0.49
            elif num_queries == 100:
                threshold = 0.60
            elif num_queries == 200:
                threshold = 0.59
        elif k == 100:
            if num_queries == 10:
                threshold = 0.58
            elif num_queries == 100:
                threshold = 0.65
            elif num_queries == 200:
                threshold = 0.67
    elif metric == "cosine":
        if k == 10:
            if num_queries == 10:
                threshold = 0.44
            elif num_queries == 100:
                threshold = 0.45
            elif num_queries == 200:
                threshold = 0.50
        elif k == 100:
            if num_queries == 10:
                threshold = 0.56
            elif num_queries == 100:
                threshold = 0.65
            elif num_queries == 200:
                threshold = 0.65

    if threshold is None:
        threshold = 0.0  # If the parameters are not in the table, return 0.0
    else:
        threshold -= 0.1  # This gives us more leeway to pass the tests

    return threshold
