FROM r-base:4.4.0


ARG EXEC_DIR="/opt/executables"

RUN apt-get update && apt-get install -y python3

#Create folders
RUN mkdir -p ${EXEC_DIR}

# Install R packages external dependencies
RUN apt-get install -y libssl-dev \
    && apt-get install -y libcurl4-openssl-dev libfontconfig1-dev \
    && apt-get install -y libharfbuzz-dev libfribidi-dev \
    && apt-get install -y libxml2-dev \
    && apt-get install -y libfreetype6-dev libpng-dev libtiff5-dev libjpeg-dev \
    && apt-get install -y cmake

# Install required R packages
ADD requirements/Requirements_1.R ${EXEC_DIR}/Requirements_1.R
RUN Rscript ${EXEC_DIR}/Requirements_1.R

ADD requirements/Requirements_2.R ${EXEC_DIR}/Requirements_2.R
RUN Rscript ${EXEC_DIR}/Requirements_2.R

ADD requirements/Requirements_3.R ${EXEC_DIR}/Requirements_3.R
RUN Rscript ${EXEC_DIR}/Requirements_3.R

RUN apt-get install -y --allow-downgrades libxcb-render0=1.15-1 libxcb-shm0=1.15-1 libxcb1=1.15-1
RUN apt-get install -y  libcairo2-dev

ADD requirements/Requirements_4.R ${EXEC_DIR}/Requirements_4.R
RUN Rscript ${EXEC_DIR}/Requirements_4.R

RUN apt-get install -y libgit2-dev

ADD requirements/Requirements_6.R ${EXEC_DIR}/Requirements_6.R
RUN Rscript ${EXEC_DIR}/Requirements_6.R

ADD requirements/Requirements_5.R ${EXEC_DIR}/Requirements_5.R
RUN Rscript ${EXEC_DIR}/Requirements_5.R

COPY pyproject.toml ${EXEC_DIR}
COPY README.md ${EXEC_DIR}
RUN apt-get update && apt-get install -y python3
RUN apt-get -y install python3-pip python3-venv

COPY src/ ${EXEC_DIR}/src

RUN python3 -m venv ${EXEC_DIR}/.venv
# RUN chmod u+x ${EXEC_DIR}/.venv/bin/activate
# RUN ${EXEC_DIR}/.venv/bin/activate
RUN ${EXEC_DIR}/.venv/bin/pip3 install ${EXEC_DIR} --no-cache-dir


WORKDIR ${EXEC_DIR}

# CMD ["Rscript", "main.R"]
ENTRYPOINT ["/opt/executables/.venv/bin/python3", "-m", "polus.tabular.regression.rt_cetsa_analysis"]
CMD ["--help"]
