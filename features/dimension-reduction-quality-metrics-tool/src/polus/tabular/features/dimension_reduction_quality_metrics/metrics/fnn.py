"""False Nearest Neighbors (FNN) metric.

Consider a query in the original space and some of its nearest neighbors. For
this query, find the nearest neighbors in the embedded space. FNN is the mean
recall of the nearest neighbors in the embedded space for a large enough number
of queries.

Intuitively, if the embedding is good, the nearest neighbors in the original
space should also be the nearest neighbors in the embedded space.
"""

import numpy
import scipy.spatial.distance


def fnn(
    original_data: numpy.ndarray,
    embedded_data: numpy.ndarray,
    query_indices: numpy.ndarray,
    n_neighbors: int,
    distance_metric: str,
) -> float:
    """Compute the False Nearest Neighbors (FNN) metric.

    Args:
        original_data: The original data.
        embedded_data: The embedded data.
        query_indices: The indices of the queries in the original space.
        n_neighbors: The number of nearest neighbors to consider.
        distance_metric: The distance metric to use.

    Returns:
        The FNN metric.
    """
    original_knn = knn_search(
        data=original_data,
        queries=original_data[query_indices],
        k=n_neighbors,
        metric=distance_metric,
    )

    embedded_knn = knn_search(
        data=embedded_data,
        queries=embedded_data[query_indices],
        k=n_neighbors,
        metric=distance_metric,
    )

    recalls = []
    for i, _ in enumerate(query_indices):
        original_neighbors = original_knn[i]
        embedded_neighbors = embedded_knn[i]
        recall = len(set(original_neighbors) & set(embedded_neighbors)) / n_neighbors
        recalls.append(recall)

    return numpy.mean(recalls)


def knn_search(
    data: numpy.ndarray,
    queries: numpy.ndarray,
    k: int,
    metric: str,
) -> numpy.ndarray:
    """Find the nearest neighbors of the queries in the data.

    Args:
        data: The data.
        queries: The queries.
        k: The number of nearest neighbors to find.
        metric: The distance metric to use.

    Returns:
        The indices of the nearest neighbors.
    """
    distances = scipy.spatial.distance.cdist(queries, data, metric)
    sorted_indices = numpy.argsort(distances, axis=1)
    return sorted_indices[:, :k]
