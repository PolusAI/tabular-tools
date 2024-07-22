"""Dimension reduction by t-distributed Stochastic Neighbor Embedding (t-SNE)."""

import typing

import numpy
import sklearn.manifold

from . import pca


def reduce(
    data: numpy.ndarray,
    *,
    n_components: int,
    perplexity: float = 30.0,
    early_exaggeration: float = 12.0,
    learning_rate: typing.Union[float, typing.Literal["auto"]] = "auto",
    max_iter: int = 1000,
    metric: str = "euclidean",
) -> numpy.ndarray:
    """Reduce the dimensionality of the data using t-SNE.

    Args:
        data: The data to reduce.

        n_components: The number of components to reduce to.

        perplexity: The perplexity is related to the number of nearest neighbors
        that is used in other manifold learning algorithms. Larger datasets
        usually require a larger perplexity. The perplexity must be less
        than the number of samples.

        early_exaggeration: Controls how tight natural clusters in the original
        space are in the embedded space and how much space will be between them.
        For larger values, the space between natural clusters will be larger in
        the embedded space.

        learning_rate: The learning rate for t-SNE is usually in the range
        [10.0, 1000.0]. If the learning rate is too high, the data may look like
        a 'ball' with any point approximately equidistant from its nearest
        neighbours. If the learning rate is too low, most points may look
        compressed in a dense cloud with few outliers. If the cost function gets
        stuck in a bad local minimum increasing the learning rate may help.

        max_iter: Maximum number of iterations for the optimization. Should be
        at least 250.

        metric: The metric to use when calculating distance between instances in
        a feature array. It must be one of the options allowed by
        scipy.spatial.distance.pdist for its metric parameter, or a metric
        listed in pairwise.PAIRWISE_DISTANCE_FUNCTIONS.

    Returns:
        The reduced data.
    """
    tsne = sklearn.manifold.TSNE(
        n_components=n_components,
        perplexity=perplexity,
        early_exaggeration=early_exaggeration,
        learning_rate=learning_rate,
        max_iter=max_iter,
        metric=metric,
    )
    return tsne.fit_transform(data)


def reduce_init_pca(
    data: numpy.ndarray,
    *,
    pca_n_components: int,
    pca_whiten: bool = False,
    pca_svd_solver: pca.SvdSolver = pca.SvdSolver.AUTO,
    pca_tol: float = 0.0,
    n_components: int,
    perplexity: float = 30.0,
    early_exaggeration: float = 12.0,
    learning_rate: typing.Union[float, typing.Literal["auto"]] = "auto",
    max_iter: int = 1000,
    metric: str = "euclidean",
) -> numpy.ndarray:
    """Reduce the dimensionality of the data using PCA followed by t-SNE.

    This is useful when the data has a high number of dimensions and t-SNE
    would be too slow to run directly.

    For the parameter documentation, see the `pca.reduce` and `tsne.reduce`
    functions.
    """
    pca_data = pca.reduce(
        data,
        n_components=pca_n_components,
        whiten=pca_whiten,
        svd_solver=pca_svd_solver,
        tol=pca_tol,
    )
    return reduce(
        pca_data,
        n_components=n_components,
        perplexity=perplexity,
        early_exaggeration=early_exaggeration,
        learning_rate=learning_rate,
        max_iter=max_iter,
        metric=metric,
    )
