#!/bin/bash

version=$(<VERSION)
datapath=$(readlink --canonicalize data)

# Inputs
inp_dir="./data/input"

# Output paths
outDir="./data/output"

# Additional args
filePattern=".arrow"

# Show the help options
# docker run polusai/k-means-clustering-plugin:${version}

docker run --mount type=bind,source=${datapath},target=/data/ \
            polusai/dimension-reduction-tool:${version} \
            --inpDir ${inpDir} \
            --filePattern ${filePattern} \
            --outDir ${outDir}
