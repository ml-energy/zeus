# Build instructions
#   If you're building this image locally, make sure you specify `TARGETARCH`.
#   Currently, this image supports `amd64` and `arm64`. For instance:
#     docker build -t mlenergy/zeus:master --build-arg TARGETARCH=amd64 .

FROM nvidia/cuda:11.8.0-base-ubuntu22.04

# Basic installs
ARG DEBIAN_FRONTEND=noninteractive
ENV TZ='America/Detroit'
RUN apt-get update -qq \
    && apt-get -y --no-install-recommends install \
       build-essential software-properties-common wget git tar rsync cmake \
    && apt-get clean all \
    && rm -r /var/lib/apt/lists/*

# Install Miniconda3 23.3.1
ENV PATH="/root/.local/miniconda3/bin:$PATH"
ARG TARGETARCH
RUN if [ "$TARGETARCH" = "amd64" ]; then \
      export CONDA_INSTALLER_PATH="Miniconda3-py39_23.3.1-0-Linux-x86_64.sh"; \
    elif [ "$TARGETARCH" = "arm64" ]; then \
      export CONDA_INSTALLER_PATH="Miniconda3-py39_23.3.1-0-Linux-aarch64.sh"; \
    else \
      echo "Unsupported architecture ${TARGETARCH}" && exit 1; \
    fi \
    && mkdir -p /root/.local \
    && wget "https://repo.anaconda.com/miniconda/$CONDA_INSTALLER_PATH" \
    && mkdir /root/.conda \
    && bash "$CONDA_INSTALLER_PATH" -b -p /root/.local/miniconda3 \
    && rm -f "$CONDA_INSTALLER_PATH" \
    && ln -sf /root/.local/miniconda3/etc/profile.d/conda.sh /etc/profile.d/conda.sh

# Install PyTorch and CUDA Toolkit
RUN pip install --no-cache-dir torch==2.0.1 torchvision==0.15.2 --index-url https://download.pytorch.org/whl/cu118

# Place stuff under /workspace
WORKDIR /workspace

# Snapshot of Zeus
ADD . /workspace/zeus

# When an outside zeus directory is mounted, have it apply immediately.
RUN cd /workspace/zeus && pip install --no-cache-dir -e .
