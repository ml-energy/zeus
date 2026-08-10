# Build instructions
#   This image supports amd64 and arm64. For instance:
#     docker build -t mlenergy/zeus:master -f docker/Dockerfile .

FROM nvidia/cuda:12.9.2-base-ubuntu24.04

# Basic installs
ARG DEBIAN_FRONTEND=noninteractive
ENV TZ='America/Detroit'
RUN apt-get update -qq \
    && apt-get -y --no-install-recommends install \
       build-essential software-properties-common wget git tar rsync cmake \
    && apt-get clean all \
    && rm -r /var/lib/apt/lists/*

# Install uv and Python
COPY --from=ghcr.io/astral-sh/uv:0.11 /uv /uvx /bin/
ENV UV_PYTHON_INSTALL_DIR=/root/.local/python
RUN uv python install 3.13

# Install PyTorch with CUDA 12.9
# The venv lives outside /workspace/zeus so that mounting does not shadow it.
ENV VIRTUAL_ENV=/opt/venv
ENV PATH="/opt/venv/bin:$PATH"
RUN uv venv "$VIRTUAL_ENV" --python 3.13 \
    && uv pip install --no-cache torch torchvision --index-url https://download.pytorch.org/whl/cu129

# Place stuff under /workspace/zeus
WORKDIR /workspace/zeus

# Snapshot of Zeus
ADD . /workspace/zeus

# When an outside zeus directory is mounted, have it apply immediately.
RUN uv pip install --no-cache -e .
