"""Dimension reduction by Principal Component Analysis (PCA)."""

import enum

import numpy
import sklearn.decomposition


class SvdSolver(str, enum.Enum):
    """The singular value decomposition solver to use."""

    AUTO = "auto"
    FULL = "full"
    ARPACK = "arpack"
    RANDOMIZED = "randomized"


def reduce(
    data: numpy.ndarray,
    *,
    n_components: int,
    whiten: bool = False,
    svd_solver: SvdSolver = SvdSolver.AUTO,
    tol: float = 0.0,
) -> numpy.ndarray:
    """Reduce the dimensionality of the data using PCA.

    Args:
        data: The data to reduce.
        n_components: The number of components to reduce to.
        whiten: Whether to whiten the data. Defaults to False.
        svd_solver: The singular value decomposition solver to use. Defaults to
        "auto".
        tol: Tolerance for singular values computed by svd_solver == "arpack".
        Must be of range [0.0, infinity).

    Returns:
        The reduced data.
    """
    pca = sklearn.decomposition.PCA(
        n_components=n_components,
        whiten=whiten,
        svd_solver=svd_solver.value,
        tol=tol,
    )
    return pca.fit_transform(data)
