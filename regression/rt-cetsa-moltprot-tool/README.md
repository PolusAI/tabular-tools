# RT_CETSA MoltenProt (v0.1.0)

This WIPP plugin runs moltenprot regression for the RT-CETSA pipeline.

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
| `--filePattern` | File Pattern to parse input files                  | Input  | string      |
| `--outDir`      | Output file                                        | Output | genericData |
| `--preview`     | Generate JSON file with outputs                    | Output | JSON        |
