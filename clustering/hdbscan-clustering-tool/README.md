﻿# Hierarchical Density-Based Spatial Clustering of Applications with Noise(HDBSCAN) Clustering (v0.4.8-dev1)

The HDBSCAN Clustering plugin clusters the data using [HDBSCAN clustering](https://pypi.org/project/hdbscan/) library. The input and output for this plugin is a CSV file. Each observation (row) in the input CSV file is assigned to one of the clusters. The output CSV file contains the column `cluster` that identifies the cluster to which each observation belongs. A user can supply a regular expression with capture groups if they wish to cluster each group independently, or if they wish to average the numerical features across each group and treat them as a single observation.

## Inputs:

### Input directory:
This plugin supports the all [vaex](https://vaex.readthedocs.io/en/latest/guides/io.html) supported file formats.

### Filename pattern:
This plugin uses [filepattern](https://filepattern2.readthedocs.io/en/latest/Home.html) python library to parse file names of tabular files to be processed by this plugin.

### Grouping pattern:
The input for this parameter is a regular expression with capture group. This input splits the data into groups based on the matched pattern. A new column `group` is created in the output file that has the group based on the given pattern. Unless `averageGroups` is set to `true`, providing a grouping pattern will cluster each group independently.

### Average groups:
`groupingPattern` to average the numerical features and produce a single row per group which is then clustered. The resulting cluster is assigned to all observations belonging in that group.

### Label column:
This is the name of the column containing the labels to be used with `groupingPattern`.

### Minimum cluster size:
This parameter defines the smallest number of points that should be considered as cluster. This is a required parameter. The input should be an integer and the value should be greater than 1.

### Increment outlier ID:
This parameter sets the ID of the outlier cluster to `1`, otherwise it will be 0. This is useful for visualization purposes if the resulting cluster IDs are turned into image annotations.

## Output:
The output is a tabular file containing the clustered data.

## Building
To build the Docker image for the conversion plugin, run
`./build-docker.sh`.

## Install WIPP Plugin
If WIPP is running, navigate to the plugins page and add a new plugin. Paste the contents of `plugin.json` into the pop-up window and submit.
For more information on WIPP, visit the [official WIPP page](https://isg.nist.gov/deepzoomweb/software/wipp).

## Options

This plugin takes four input arguments and one output argument:

| Name                   | Description                                                                                    | I/O    | Type          |
| ---------------------- | ---------------------------------------------------------------------------------------------- | ------ | ------------- |
| `--inpDir`             | Input tabular data files.                                                                      | Input  | genericData   |
| `--groupingPattern`    | Regular expression to group rows. Clustering will be applied across capture groups by default. | Input  | string        |
| `--averageGroups`      | Average data across groups. Requires capture groups                                            | Input  | boolean       |
| `--labelCol`           | Name of the column containing labels for grouping pattern.                                     | Input  | string        |
| `--minClusterSize`     | Minimum cluster size.                                                                          | Input  | number        |
| `--incrementOutlierId` | Increments outlier ID to 1.                                                                    | Input  | boolean       |
| `--outDir`             | Output collection                                                                              | Output | genericData   |
| `--preview`            | Generate a JSON file with outputs                                                              | Output | JSON       |
