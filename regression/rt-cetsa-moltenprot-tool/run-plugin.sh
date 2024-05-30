#!/bin/bash
version=$(<VERSION)
datapath=$(readlink -f data)

# Inputs
inpDir=/data/input
pattern=".*"

# Output paths
outDir=/data/output

# Log level, must be one of ERROR, CRITICAL, WARNING, INFO, DEBUG
LOGLEVEL=INFO

docker run --mount type=bind,source=${datapath},target=/data/  \
            --env POLUS_LOG=${LOGLEVEL} \
            polusai/rt-cetsa-moltenproten-tool:${version} \
            --inpDir ${inpDir} \
            --filePattern ${pattern} \
            --outDir ${outDir}
