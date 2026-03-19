# hash:sha256:ceaac4e158e1ec4000763fa4a7125cf9f33991b0951ec4acfd2468ee28546255
ARG REGISTRY_HOST
FROM $REGISTRY_HOST/codeocean/c1-vscode:4.20.0-pylint2023.10.1-copilot1.143.601

ARG DEBIAN_FRONTEND=noninteractive
ARG GIT_ASKPASS
COPY git-ask-pass /

RUN mamba install -y -c conda-forge \
    python==3.11 \
    ipykernel==6.29.5 \
    hdbscan \
    && conda clean -ya

# install libraries for opencv-python
RUN apt-get update && apt-get install -y --no-install-recommends \
    libxcb1 libgl1 libglib2.0-0
RUN apt-get clean && rm -rf /var/lib/apt/lists/*

RUN pip install -U --no-cache-dir \
    suite2p \
    aind-session \
    aind-codeocean-utils \
    aind-data-access-api \
    aind-ophys-utils \
    aind-data-schema==1.2.0 \
    numpy==1.26.4 \
    codeocean==0.14.0 \
    ipywidgets
    

COPY postInstall /
RUN --mount=type=secret,id=secrets . /run/secrets/secrets \
    && /postInstall
