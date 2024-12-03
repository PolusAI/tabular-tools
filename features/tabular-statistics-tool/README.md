# Tabular Statistics

This plugin computes statistical measures on numeric and floating point data columns in tabular files. The supported input file formats are `CSV`, `Feather`, `Arrow`, and `Parquet`, leveraging the  [PyArrow](https://arrow.apache.org/) library for efficient processing. If no columns have the `file` header, then this plugin throws and error.

## Available Statistics:

1. [mean (arithmetic mean)](https://en.wikipedia.org/wiki/Mean#Arithmetic_mean_(AM))
2. [median](https://en.wikipedia.org/wiki/Median#The_sample_median)
3. [std (standard deviation)](https://en.wikipedia.org/wiki/Standard_deviation)
4. [var (variance)](https://en.wikipedia.org/wiki/Variance)
5. [skew (Fisher-Pearson skewness)](https://www.itl.nist.gov/div898/handbook/eda/section3/eda35b.htm)
6. [kurt (excess kurtosis)](https://www.itl.nist.gov/div898/handbook/eda/section3/eda35b.htm)
7. count (number of rows sampled)
8. [iqr (Interquartile_range)](https://en.wikipedia.org/wiki/Interquartile_range)
9. prop: counts the numbers greater than 0 and divides this count by the total number of elements in the sequence
10. min: Minimum value in the dataset.
11. max: Maximum value in the dataset.


## Usage:
- A directory containing one or more tabular files in the supported formats
- Each file must include numeric and floating point data columns
- If a `groupBy` column is specified in the input arguments, it must be present in the data

For more information on WIPP, visit the [official WIPP page](https://isg.nist.gov/deepzoomweb/software/wipp).

## Building

To build the Docker image for the conversion plugin, run
`./build-docker.sh`.

## Install WIPP Plugin

If WIPP is running, navigate to the plugins page and add a new plugin. Paste the contents of `plugin.json` into the pop-up window and submit.

## Options

This plugin takes two input argument and one output argument:

| Name            | Description                                         | I/O    | Type          |
| --------------- | --------------------------------------------------- | ------ | ------------- |
| `--statistics`  | Types of statistics to calculate                    | Input  | array         |
| `--inpDir`      | Input csv collection to be processed by this plugin | Input  | genericData   |
| `--filePattern` | The filePattern of the images in represented in csv | Input  | string        |
| `--groupBy`     | The variable(s) of how the images should be grouped | Input  | string        |
| `--outDir`      | Output collection                                   | Output | genericData   |
