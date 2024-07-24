# Dimension Reduction (v0.1.0-dev0)

This tool is used to reduce the dimensionality of the input data.
It provides the following methods for dimensionality reduction:

1. Principal Component Analysis (PCA)
2. t-Distributed Stochastic Neighbor Embedding (t-SNE)
3. t-SNE with PCA initialization.
4. Uniform Manifold Approximation and Projection (UMAP)

The input data should be in the form of a tabular file (`.csv`, `.feather`, `parquet` or `npy`).
This tool takes tabular data as input and outputs a reduced dimensionality version of the input data.
Each method has its own set of parameters that can be tuned to get the desired output.

The CLI parameters required for all methods are:

1. `--inpDir`: Directory containing input tabular data.
2. `--filePattern`: Pattern to parse tabular files.
3. `--algorithm`: Dimensionality reduction algorithm to use. Options are `pca`, `tsne`, `tsne_init_pca`, and `umap`.
4. `--nComponents`: Number of dimensions to reduce to.
5. `--outDir`: Output directory.

You can also use the `--preview` flag to generate a JSON file indicating what the outputs would be without running the tool.

For PCA, the required parameters are:

- `--pcaWhiten`: Boolean flag to indicate whether to whiten the data.
- `--pcaSvdSolver`: Solver to use for PCA. Options are `auto`, `full`, `arpack`, and `randomized`.
- `--pcaTol`: Tolerance for PCA with the `arpack` solver.

For more details in each parameter, see [the documentation here](https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html).

For t-SNE, the required parameters are:

- `--tsnePerplexity`: Perplexity parameter for t-SNE.
- `--tsneEarlyExaggeration`: Early exaggeration factor for t-SNE.
- `--tsneLearningRate`: Learning rate for t-SNE.
- `--tsneMaxIter`: Maximum number of iterations for t-SNE.
- `--tsneMetric`: The distance metric to use for t-SNE.

for more details in each parameter, see [the documentation here](https://scikit-learn.org/stable/modules/generated/sklearn.manifold.TSNE.html).

For t-SNE with PCA initialization, the required parameters are:

- All parameters required for t-SNE.
- `--tsneInitNComponents`: Number of components to use for PCA initialization.
- All parameters required for PCA.

For UMAP, the required parameters are:

- `--umapNNeighbors`: Number of neighbors to use for UMAP.
- `--umapNEpochs`: Number of epochs for UMAP.
- `--umapMinDist`: Minimum distance between points in UMAP.
- `--umapSpread`: Spread of UMAP.
- `--umapMetric`: The distance metric to use for UMAP.

For more details in each parameter, see [the documentation here](https://umap-learn.readthedocs.io/en/latest/parameters.html).

## Docker Container

To build the Docker image for the conversion plugin, run `./build-docker.sh`.

## Install WIPP Plugin

If WIPP is running, navigate to the plugins page and add a new plugin.
Paste the contents of `plugin.json` into the pop-up window and submit.
For more information on WIPP, visit the [official WIPP page](https://isg.nist.gov/deepzoomweb/software/wipp).

## Options

This plugin takes seven input arguments and one output argument:

| Name                      | Description                                               | I/O    | Type        | Default   |
| ------------------------- | --------------------------------------------------------- | ------ | ----------- | --------- |
| `--inpDir`                | Directory containing input tabular data.                  | Input  | genericData | N/A       |
| `--filePattern`           | Pattern to parse tabular files.                           | Input  | string      | ".*"      |
| `--preview`               | Generate JSON file with outputs without running the tool. | Input  | boolean     | False     |
| `--outDir`                | Output directory.                                         | Output | genericData | N/A       |
| `--algorithm`             | Dimensionality reduction algorithm to use.                | Input  | enum        | umap      |
| `--nComponents`           | Number of dimensions to reduce to.                        | Input  | int         |           |
| `--pcaWhiten`             | Boolean flag to indicate whether to whiten the data.      | Input  | boolean     | False     |
| `--pcaSvdSolver`          | Solver to use for PCA.                                    | Input  | enum        | auto      |
| `--pcaTol`                | Tolerance for PCA with the `arpack` solver.               | Input  | float       | 0.0       |
| `--tsnePerplexity`        | Perplexity parameter for t-SNE.                           | Input  | float       | 30.0      |
| `--tsneEarlyExaggeration` | Early exaggeration factor for t-SNE.                      | Input  | float       | 12.0      |
| `--tsneLearningRate`      | Learning rate for t-SNE.                                  | Input  | float       | 200.0     |
| `--tsneMaxIter`           | Maximum number of iterations for t-SNE.                   | Input  | int         | 1000      |
| `--tsneMetric`            | The distance metric to use for t-SNE.                     | Input  | string      | euclidean |
| `--tsneInitNComponents`   | Number of components to use for PCA initialization.       | Input  | int         | 50        |
| `--umapNNeighbors`        | Number of neighbors to use for UMAP.                      | Input  | int         | 15        |
| `--umapNEpochs`           | Number of epochs for UMAP.                                | Input  | int         | 500       |
| `--umapMinDist`           | Minimum distance between points in UMAP.                  | Input  | float       | 0.1       |
| `--umapSpread`            | Spread of UMAP.                                           | Input  | float       | 1.0       |
| `--umapMetric`            | The distance metric to use for UMAP.                      | Input  | string      | euclidean |
