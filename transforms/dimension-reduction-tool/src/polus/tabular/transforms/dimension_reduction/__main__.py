"""CLI for the Dimension Reduction tool."""

import json
import logging
import pathlib

import filepattern
import tqdm
import typer
from polus.tabular.transforms.dimension_reduction import POLUS_LOG_LVL
from polus.tabular.transforms.dimension_reduction import POLUS_TAB_EXT
from polus.tabular.transforms.dimension_reduction import Algorithm
from polus.tabular.transforms.dimension_reduction import SvdSolver
from polus.tabular.transforms.dimension_reduction import reduce

# Initialize the logger
logging.basicConfig(
    format="%(asctime)s - %(name)-8s - %(levelname)-8s - %(message)s",
    datefmt="%d-%b-%y %H:%M:%S",
)
logger = logging.getLogger("polus.tabular.transforms.dimension_reduction")
logger.setLevel(POLUS_LOG_LVL)

app = typer.Typer()


@app.command()
def main(
    inp_dir: pathlib.Path = typer.Option(
        ...,
        "--inpDir",
        help="Input data that needs to be reduced in dimensionality.",
        exists=True,
        file_okay=False,
        dir_okay=True,
        readable=True,
        resolve_path=True,
    ),
    file_pattern: str = typer.Option(
        ".*",
        "--filePattern",
        help="pattern to parse tabular files",
    ),
    algorithm: Algorithm = typer.Option(
        Algorithm.UMAP,
        "--algorithm",
        help="The algorithm to use for dimensionality reduction",
    ),
    n_components: int = typer.Option(
        ...,
        "--nComponents",
        help="The dimensionality to reduce the data to",
    ),
    pca_whiten: bool = typer.Option(
        False,
        "--pcaWhiten",
        help="PCA: Whether to whiten the data",
    ),
    pca_svd_solver: SvdSolver = typer.Option(
        SvdSolver.AUTO,
        "--pcaSvdSolver",
        help="PCA: The singular value decomposition solver to use",
    ),
    pca_tol: float = typer.Option(
        0.0,
        "--pcaTol",
        help='PCA: Tolerance for singular values computed by svd_solver == "arpack"',
    ),
    tsne_perplexity: float = typer.Option(
        30.0,
        "--tsnePerplexity",
        help="t-SNE: The perplexity is related to the number of nearest neighbors "
        "that is used in other manifold learning algorithms. Larger datasets "
        "usually require a larger perplexity. Consider selecting a value between "
        "5 and 50.",
    ),
    tsne_early_exaggeration: float = typer.Option(
        12.0,
        "--tsneEarlyExaggeration",
        help="t-SNE: Controls how tight natural clusters in the original space are in "
        "the embedded space and how much space will be between them. For larger "
        "values, the space between natural clusters will be larger in the embedded "
        "space.",
    ),
    tsne_learning_rate: float = typer.Option(
        200.0,
        "--tsneLearningRate",
        help="The learning rate for t-SNE is usually in the range [10.0, 1000.0]. If "
        "the learning rate is too high, the data may look like a 'ball' with any "
        "point approximately equidistant from its nearest neighbours. If the learning "
        "rate is too low, most points may look compressed in a dense cloud with few "
        "outliers. If the cost function gets stuck in a bad local minimum increasing "
        "the learning rate may help.",
    ),
    tsne_max_iter: int = typer.Option(
        1000,
        "--tsneMaxIter",
        help="t-SNE: Maximum number of iterations for the optimization. Should be at "
        "least 250.",
    ),
    tsne_metric: str = typer.Option(
        "euclidean",
        "--tsneMetric",
        help="t-SNE: The metric to use when calculating distance between "
        "instances in a feature array. It must be one of the options allowed by "
        "scipy.spatial.distance.pdist for its metric parameter",
    ),
    tsne_init_n_components: int = typer.Option(
        50,
        "--tsneInitNComponents",
        help="t-SNE: The number of components to reduce to with PCA before running "
        "t-SNE.",
    ),
    umap_n_neighbors: int = typer.Option(
        15,
        "--umapNNeighbors",
        help="UMAP: The size of local neighborhood (in terms of number of neighboring "
        "sample points) used for manifold approximation. Larger values result in more "
        "global views of the manifold, while smaller values result in more local data "
        "being preserved. In general, values should be in the range 2 to 100.",
    ),
    umap_n_epochs: int = typer.Option(
        None,
        "--umapNEpochs",
        help="UMAP: The number of training epochs to be used in optimizing the low "
        "dimensional embedding. Larger values result in more accurate embeddings. If "
        "None, the value will be set automatically based on the size of the input "
        "dataset (200 for large datasets, 500 for small).",
    ),
    umap_min_dist: float = typer.Option(
        0.1,
        "--umapMinDist",
        help="UMAP: The effective minimum distance between embedded points. Smaller "
        "values will result in a more clustered/clumped embedding where nearby points "
        "on the manifold are drawn closer together, while larger values will result "
        "in a more even dispersal of points. The value should be set relative to the "
        "spread value, which determines the scale at which embedded points will be "
        "spread out.",
    ),
    umap_spread: float = typer.Option(
        1.0,
        "--umapSpread",
        help="UMAP: The effective scale of embedded points. In combination with "
        "min_dist this determines how clustered/clumped the embedded points are.",
    ),
    umap_metric: str = typer.Option(
        "euclidean",
        "--umapMetric",
        help="UMAP: The metric to use when calculating distance between "
        "instances in a feature array. It must be one of the options allowed by "
        "scipy.spatial.distance.pdist for its metric parameter",
    ),
    out_dir: pathlib.Path = typer.Option(
        ...,
        "--outDir",
        help="Output collection",
        exists=True,
        file_okay=False,
        dir_okay=True,
        writable=True,
        resolve_path=True,
    ),
    preview: bool = typer.Option(
        False,
        "--preview",
        help="Output a JSON preview of outputs produced by this tool",
    ),
) -> None:
    """CLI for the Dimension Reduction tool."""
    logger.info(f"inpDir = {inp_dir}")
    logger.info(f"filePattern = {file_pattern}")
    logger.info(f"algorithm = {algorithm.value}")
    logger.info(f"nComponents = {n_components}")
    logger.info(f"pcaWhiten = {pca_whiten}")
    logger.info(f"pcaSvdSolver = {pca_svd_solver.value}")
    logger.info(f"pcaTol = {pca_tol}")
    logger.info(f"tsnePerplexity = {tsne_perplexity}")
    logger.info(f"tsneEarlyExaggeration = {tsne_early_exaggeration}")
    logger.info(f"tsneLearningRate = {tsne_learning_rate}")
    logger.info(f"tsneMaxIter = {tsne_max_iter}")
    logger.info(f"tsneMetric = {tsne_metric}")
    logger.info(f"tsneInitNComponents = {tsne_init_n_components}")
    logger.info(f"umapNNeighbors = {umap_n_neighbors}")
    logger.info(f"umapNEpochs = {umap_n_epochs}")
    logger.info(f"umapMinDist = {umap_min_dist}")
    logger.info(f"umapSpread = {umap_spread}")
    logger.info(f"umapMetric = {umap_metric}")
    logger.info(f"outDir = {out_dir}")
    logger.info(f"preview = {preview}")

    kwargs = {
        "n_components": n_components,
        "pca_whiten": pca_whiten,
        "pca_svd_solver": pca_svd_solver,
        "pca_tol": pca_tol,
        "tsne_perplexity": tsne_perplexity,
        "tsne_early_exaggeration": tsne_early_exaggeration,
        "tsne_learning_rate": tsne_learning_rate,
        "tsne_max_iter": tsne_max_iter,
        "tsne_metric": tsne_metric,
        "tsne_init_n_components": tsne_init_n_components,
        "umap_n_neighbors": umap_n_neighbors,
        "umap_n_epochs": umap_n_epochs,
        "umap_min_dist": umap_min_dist,
        "umap_spread": umap_spread,
        "umap_metric": umap_metric,
    }
    kwargs = algorithm.parse_kwargs(kwargs)

    fp = filepattern.FilePattern(path=inp_dir, pattern=file_pattern)
    files = [p for _, [p] in fp()]

    logger.info(f"Found {len(files)} files to process.")

    path: pathlib.Path

    if preview:
        out_dict: dict[str, list[str]] = {"files": []}
        for path in files:
            out_dict["files"].append(str(out_dir / (path.stem + POLUS_TAB_EXT)))
        with (out_dir / "preview.json").open("w") as f:
            json.dump(out_dict, f, indent=2)
    else:
        for path in tqdm.tqdm(files):
            reduce(
                inp_path=path,
                out_path=out_dir / (path.stem + POLUS_TAB_EXT),
                algorithm=algorithm,
                kwargs=kwargs,
            )


if __name__ == "__main__":
    app()
