#!/bin/bash

version=$(<VERSION)
datapath=$(readlink --canonicalize ../data)

# Inputs
inpDir=/data/path_to_images
outDir=/data/path_to_output
filePattern='p0{r:d+}_x{x:d+}_y{y:d+}_wx{t:d+}_wy{p:d+}_c{c:d}.ome.tif'
chunkSize=50
groupBy='c'



# Log level, must be one of ERROR, CRITICAL, WARNING, INFO, DEBUG
LOGLEVEL=INFO

docker run --mount type=bind,source=${datapath},target=/data/ \
            --user $(id -u):$(id -g) \
            --env POLUS_LOG=${LOGLEVEL} \
            polusai/filepattern-generator-tool:${version} \
            --inpDir ${inpDir} \
            --outDir ${outDir} \
            --filePattern ${filePattern} \
            --chunkSize ${chunkSize} \
            --groupBy ${groupBy} \
