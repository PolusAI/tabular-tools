#!/bin/bash

version=$(<VERSION)
datapath=$(readlink --canonicalize data)

# Inputs
inp_dir=/data/input
file_pattern="{r:c+}_{c:d+}_c{p:d}.ome.tif"
outformat=".arrow"
group_by=r,c
statistics=mean,median

# Output paths
out_dir=/data/output

# Run the plugin
docker run -e POLUS_TAB_EXT=${outformat} --mount type=bind,source=${datapath},target=/data/ \
            polusai/tabular-statistics-tool:${version} \
            --inpDir ${inp_dir} \
            --outDir ${out_dir} \
            --filePattern ${file_pattern} \
            --groupBy ${group_by} \
            --statistics ${statistics} \
         