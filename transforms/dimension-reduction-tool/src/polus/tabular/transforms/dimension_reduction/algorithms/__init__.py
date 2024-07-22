"""Dimension Reduction algorithms supported by this tool."""

import enum
import typing

from . import pca
from . import tsne
from . import umap
from .pca import SvdSolver


class Algorithm(str, enum.Enum):
    """The dimension reduction algorithms supported by this tool."""

    PCA = "pca"
    TSNE = "tsne"
    TSNE_INIT_PCA = "tsne_init_pca"
    UMAP = "umap"

    def parse_kwargs(self, inp_kwargs: dict) -> dict:  # noqa: PLR0915, PLR0912, C901
        """Converts the inputs from the typer CLI to be used by the algorithms."""
        out_kwargs = {}

        if "n_components" in inp_kwargs:
            out_kwargs["n_components"] = inp_kwargs["n_components"]
        else:
            msg = "n_components is a required argument."
            raise ValueError(msg)

        if self == Algorithm.PCA:
            expected_keys = ["whiten", "svd_solver", "tol"]
            for key in expected_keys:
                pca_key = f"pca_{key}"
                if pca_key in inp_kwargs:
                    out_kwargs[key] = inp_kwargs[pca_key]
                else:
                    msg = f"{pca_key} is a required argument for PCA."
                    raise ValueError(msg)
        elif self == Algorithm.TSNE:
            expected_keys = [
                "perplexity",
                "early_exaggeration",
                "learning_rate",
                "max_iter",
                "metric",
            ]
            for key in expected_keys:
                tsne_key = f"tsne_{key}"
                if tsne_key in inp_kwargs:
                    out_kwargs[key] = inp_kwargs[tsne_key]
                else:
                    msg = f"{tsne_key} is a required argument for t-SNE."
                    raise ValueError(msg)
        elif self == Algorithm.TSNE_INIT_PCA:
            if "tsne_init_n_components" in inp_kwargs:
                out_kwargs["pca_n_components"] = inp_kwargs["tsne_init_n_components"]
            else:
                msg = (
                    "tsne_init_n_components is a required argument for t-SNE "
                    "with PCA initialization."
                )
                raise ValueError(msg)

            pca_keys = ["whiten", "svd_solver", "tol"]
            for key in pca_keys:
                pca_key = f"pca_{key}"
                if pca_key in inp_kwargs:
                    out_kwargs[pca_key] = inp_kwargs[pca_key]
                else:
                    msg = f"{pca_key} is a required argument for PCA."
                    raise ValueError(msg)

            tsne_keys = [
                "perplexity",
                "early_exaggeration",
                "learning_rate",
                "max_iter",
                "metric",
            ]
            for key in tsne_keys:
                tsne_key = f"tsne_{key}"
                if tsne_key in inp_kwargs:
                    out_kwargs[key] = inp_kwargs[tsne_key]
                else:
                    msg = f"{tsne_key} is a required argument for t-SNE."
                    raise ValueError(msg)
        elif self == Algorithm.UMAP:
            expected_keys = ["n_neighbors", "n_epochs", "min_dist", "spread"]
            for key in expected_keys:
                umap_key = f"umap_{key}"
                if umap_key in inp_kwargs:
                    out_kwargs[key] = inp_kwargs[umap_key]
                else:
                    msg = f"{umap_key} is a required argument for UMAP."
                    raise ValueError(msg)
        else:
            allowed_algorithms = ", ".join(Algorithm.__members__.keys())
            msg = f"Unsupported algorithm: {self}. Must be one of: {allowed_algorithms}"
            raise ValueError(msg)

        return out_kwargs


__all__ = ["pca", "tsne", "umap", "SvdSolver", "Algorithm"]
