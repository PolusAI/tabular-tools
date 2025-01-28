# RT_CETSA metadata (v0.4.0-dev0)

This WIPP plugin preprocess data for the RT-CETSA pipeline.
It generates temperature metadata and generate new image files which
names encode the metadata (1.tif will become 1_37.0.tif).
This metadata can be generated from a range provided as parameters or
extracted from a camera data file.

## Building

To build the Docker image for the conversion plugin, run
`./build-docker.sh`.

## Install WIPP Plugin

If WIPP is running, navigate to the plugins page and add a new plugin. Paste the contents of `plugin.json` into the pop-up window and submit.

## Options

This plugin takes eight input argument and one output argument:

| Name            | Description                                        | I/O    | Type        |
|-----------------|----------------------------------------------------|--------|-------------|
| `--inpDir`      | Input data collection to be processed by this tool | Input  | genericData |
| `--outDir`      | Output file                                         | Output | genericData |
| `--preview`     | Generate JSON file with outputs                     | Output | JSON        |
