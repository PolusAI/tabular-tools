# Dimension Reduction (v0.1.0-dev0)

This tool is used to reduce the dimensionality of the input data.
It provides the following methods for dimensionality reduction:

1. Principal Component Analysis (PCA)
2. t-Distributed Stochastic Neighbor Embedding (t-SNE)
3. Uniform Manifold Approximation and Projection (UMAP)

The input data should be in the form of a tabular file (`.csv` or `.arrow`).
This tool takes tabular data as input and outputs a reduced dimensionality version of the input data.
Each method has its own set of parameters that can be tuned to get the desired output.

The CLI parameters are:

1. `--inpDir`: Directory containing input tabular data.
2. `--filePattern`: Pattern to parse tabular files.
3. `--preview`: Generate JSON file with outputs without running the tool.
4. `--outDir`: Output directory.

## Docker Container

To build the Docker image for the conversion plugin, run `./build-docker.sh`.

## Install WIPP Plugin

If WIPP is running, navigate to the plugins page and add a new plugin.
Paste the contents of `plugin.json` into the pop-up window and submit.
For more information on WIPP, visit the [official WIPP page](https://isg.nist.gov/deepzoomweb/software/wipp).

## Options

This plugin takes seven input arguments and one output argument:

| Name             | Description                                                                  | I/O    | Type          | Default |
| ---------------- | ---------------------------------------------------------------------------- | ------ | ------------- | ------- |
| `--inpDir`       | Directory containing input tabular data.                                     | Input  | genericData   | N/A     |
| `--filePattern`  | Pattern to parse tabular files.                                              | Input  | string        | ".*"    |
| `--preview`      | Generate JSON file with outputs without running the tool.                    | Input  | boolean       | False   |
| `--outDir`       | Output directory.                                                            | Output | genericData   | N/A     |
