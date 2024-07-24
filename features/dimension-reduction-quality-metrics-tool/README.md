# Dimension Reduction Quality Metrics (v0.1.0-dev0)

This tool is used to measure the quality of dimensionality reductions.
It provides the following methods for dimensionality reduction:

1. False Nearest Neighbors (FNN).

## FNN

Consider a query in the original space and some of its nearest neighbors.
Find the nearest neighbors of the query in the reduced space.
If the nearest neighbors in the reduced space are not the same as the nearest neighbors in the original space, then the reduced space is not a good representation of the original space.
FNN is the mean recall of the nearest neighbors in the reduced space over a large number of queries.

## Parameters

This tool takes the following parameters:

1. `--originalDir`: Directory containing the original data.
2. `--originalPattern`: Pattern to parse original files.
3. `--embeddedDir`: Directory containing the reduced data.
4. `--embeddedPattern`: Pattern to parse reduced files.
5. `--numQueries`: Number of queries to use.
6. `--ks`: Comma separated list of numbers of nearest neighbors to consider.
7. `--distanceMetrics`: Comma separated list of distance metrics to use.
8. `--qualityMetrics`: Comma separated list of quality metrics to use.
9. `--outDir`: Output directory.
10. `--preview`: Generate JSON file with outputs without running the tool.

## Docker Container

To build the Docker image for the conversion plugin, run `./build-docker.sh`.

## Install WIPP Plugin

If WIPP is running, navigate to the plugins page and add a new plugin.
Paste the contents of `plugin.json` into the pop-up window and submit.
For more information on WIPP, visit the [official WIPP page](https://isg.nist.gov/deepzoomweb/software/wipp).

## Options

This plugin takes seven input arguments and one output argument:

| Name                | Description                                               | I/O    | Type        | Default            |
| ------------------- | --------------------------------------------------------- | ------ | ----------- | ------------------ |
| `--originalDir`     | Directory containing the original data.                   | Input  | genericData | N/A                |
| `--originalPattern` | Pattern to parse original files.                          | Input  | string      | ".*"               |
| `--embeddedDir`     | Directory containing the reduced data.                    | Input  | genericData | N/A                |
| `--embeddedPattern` | Pattern to parse reduced files.                           | Input  | string      | ".*"               |
| `--numQueries`      | Number of queries to use.                                 | Input  | int         | 1000               |
| `--ks`              | Comma separated list of numbers of nearest neighbors.     | Input  | string      | "10,100"           |
| `--distanceMetrics` | Comma separated list of distance metrics to use.          | Input  | string      | "euclidean,cosine" |
| `--qualityMetrics`  | Comma separated list of quality metrics to use.           | Input  | string      | "fnn"              |
| `--outDir`          | Output directory.                                         | Output | genericData | N/A                |
| `--preview`         | Generate JSON file with outputs without running the tool. | Input  | boolean     | False              |
