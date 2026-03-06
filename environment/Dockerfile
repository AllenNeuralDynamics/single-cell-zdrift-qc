# hash:sha256:ceaac4e158e1ec4000763fa4a7125cf9f33991b0951ec4acfd2468ee28546255
ARG REGISTRY_HOST
FROM $REGISTRY_HOST/codeocean/c1-vscode:4.20.0-pylint2023.10.1-copilot1.143.601

ARG DEBIAN_FRONTEND=noninteractive
ARG GIT_ASKPASS
COPY git-askpass /

RUN conda install -y \
        ipykernel==6.29.5 \
    && conda clean -ya

RUN pip install -U --no-cache-dir \
    aind-session \
    aind-data-schema==1.2.0 \
    h5py \
    matplotlib \
    numpy==1.26.4 \
    pandas \
    seaborn \
    codeocean \
    dask \
    distributed \
    tables \
    pyarrow \
    ray \
    ipywidgets \
    scikit-learn \
    aind-codeocean-utils \
    aind-data-access-api \
    aind-ophys-utils \
    aind_session

COPY postInstall /
RUN --mount=type=secret,id=secrets . /run/secrets/secrets \
    && /postInstall
