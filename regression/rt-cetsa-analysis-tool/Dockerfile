FROM polusai/bfio:2.1.9

# environment variables defined in polusai/bfio
ENV EXEC_DIR="/opt/executables"
ENV POLUS_LOG="INFO"

COPY env-linux.yml ${EXEC_DIR}/env-linux.yml
# Work directory defined in the base container
WORKDIR ${EXEC_DIR}

RUN apt-get update && apt-get install -y wget
RUN mkdir -p miniconda3 && wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda3/miniconda.sh && \
bash miniconda3/miniconda.sh -b -u -p miniconda3 && rm miniconda3/miniconda.sh
RUN /opt/executables/miniconda3/bin/conda env update -f env-linux.yml
