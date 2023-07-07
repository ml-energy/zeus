# Copyright (C) 2022 Jae-Won Chung <jwnchung@umich.edu>
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
#     http://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

FROM nvidia/cuda:11.8.0-devel-ubuntu22.04

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
    && ln -sf /root/.local/miniconda3/etc/profile.d/conda.sh /etc/profile.d/conda.sh \
    && unset CONDA_INSTALLER_PATH

# Install PyTorch and CUDA Toolkit
RUN pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# Place stuff under /workspace
WORKDIR /workspace

# Snapshot of Zeus
ADD . /workspace/zeus

# When an outside zeus directory is mounted, have it apply immediately.
RUN pip install -e zeus

# Build and bake in the Zeus monitor.
RUN cd /workspace/zeus/zeus_monitor && cmake . && make && cp zeus_monitor /usr/local/bin/ && cd /workspace
