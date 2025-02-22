# RT_CETSA Analysis Tool (v0.5.0-dev0)

This WIPP plugin runs statistical analysis for the RT-CETSA pipeline.

## Building

To build the Docker image for the conversion plugin, run
`./build-docker.sh`.

## Install WIPP Plugin

If WIPP is running, navigate to the plugins page and add a new plugin. Paste the contents of `plugin.json` into the pop-up window and submit.

## Options

This plugin takes eight input argument and one output argument:

| Name            | Description                                        | I/O    | Type        |
|-----------------|----------------------------------------------------|--------|-------------|
| `--inpDir` | Input directory containing the all data files | Input  | genericData |
| `--params` | name of the moltenprot fit params csv file in the input directory | Input  | string      |
| `--values` | name of the moltenprot baseline corrected values csv file in the input directory
| `--platemap` | Path to the platemap file | Input | genericData |
| `--outDir`      | Output file                                        | Output | genericData |
| `--preview`     | Generate JSON file with outputs                    | Output | JSON        |

## Build options

By default `./build-docker` will build the image using `Dockerfile`, which install R with conda.
In regression are noticed, this file can be swapped with `Dockerfile-original` which is the original version
of the dockerfile that is using apt to install all dependencies and is known to work correctly.
