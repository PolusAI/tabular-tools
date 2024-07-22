"""Dimension reduction by Uniform Manifold Approximation and Projection (UMAP)."""

import typing

import numpy
import umap


def reduce(
    data: numpy.ndarray,
    *,
    n_components: int,
    n_neighbors: int = 15,
    metric: str = "euclidean",
    n_epochs: typing.Optional[int] = None,
    min_dist: float = 0.1,
    spread: float = 1.0,
) -> numpy.ndarray:
    """Reduce the dimensionality of the data using UMAP.

    Args:
        data: The data to reduce.

        n_components: The number of components to reduce to.

        n_neighbors: The size of local neighborhood (in terms of number of
        neighboring sample points) used for manifold approximation. Larger
        values result in more global views of the manifold, while smaller
        values result in more local data being preserved. In general, values
        should be in the range 2 to 100.

        metric: The metric to use when calculating distance between instances in
        the high dimensional space. It must be one of the options allowed by
        scipy.spatial.distance.pdist for its metric parameter.

        n_epochs: The number of training epochs to be used in optimizing the
        low dimensional embedding. Larger values result in more accurate
        embeddings. If None, the value will be set automatically based on the
        size of the input dataset (200 for large datasets, 500 for small).

        min_dist: The effective minimum distance between embedded points.
        Smaller values will result in a more clustered/clumped embedding where
        nearby points on the manifold are drawn closer together, while larger
        values will result in a more even dispersal of points. The value should
        be set relative to the spread value, which determines the scale at
        which embedded points will be spread out.

        spread: The effective scale of embedded points. In combination with
        ``min_dist`` this determines how clustered/clumped the embedded points
        are.

    Returns:
        The reduced data.
    """
    reducer = umap.UMAP(
        n_components=n_components,
        n_neighbors=n_neighbors,
        metric=metric,
        n_epochs=n_epochs,
        min_dist=min_dist,
        spread=spread,
    )
    return reducer.fit_transform(data)
