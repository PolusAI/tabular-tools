FROM polusai/bfio:2.4.5

# environment variables defined in polusai/bfio
ENV EXEC_DIR="/opt/executables"
ENV POLUS_IMG_EXT=".ome.tif"
ENV POLUS_TAB_EXT=".arrow"
ENV POLUS_LOG="INFO"

# Work directory defined in the base container
WORKDIR ${EXEC_DIR}

# TODO: Change the tool_dir to the tool directory
ENV TOOL_DIR="transforms/tabular-thresholding-tool"

# Copy the repository into the container
RUN mkdir tabular-tools
COPY . ${EXEC_DIR}/tabular-tools

# Install the tool
RUN pip3 install "${EXEC_DIR}/tabular-tools/${TOOL_DIR}" --no-cache-dir

# TODO: Change the entrypoint to the tool entrypoint
ENTRYPOINT ["python3", "-m", "polus.tabular.transforms.tabular_thresholding"]
CMD ["--help"]
