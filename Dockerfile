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

FROM nvidia/cuda:11.3.1-devel-ubuntu20.04

# Basic installs
ARG DEBIAN_FRONTEND=noninteractive
ENV TZ='America/Detroit'
RUN apt-get update -qq \
    && apt-get -y --no-install-recommends install \
       build-essential software-properties-common wget git tar rsync \
    && apt-get clean all \
    && rm -r /var/lib/apt/lists/*

# Install cmake 3.22.0
RUN wget https://github.com/Kitware/CMake/releases/download/v3.22.0/cmake-3.22.0-linux-x86_64.tar.gz \
    && tar xzf cmake-* \
    && rsync -a cmake-*/bin /usr/local \
    && rsync -a cmake-*/share /usr/local \
    && rm -r cmake-*

# Install Miniconda3 4.12.0
ENV PATH="/root/.local/miniconda3/bin:$PATH"
RUN mkdir -p /root/.local \
    && wget https://repo.anaconda.com/miniconda/Miniconda3-py39_4.12.0-Linux-x86_64.sh \
    && mkdir /root/.conda \
    && bash Miniconda3-py39_4.12.0-Linux-x86_64.sh -b -p /root/.local/miniconda3 \
    && rm -f Miniconda3-py39_4.12.0-Linux-x86_64.sh \
    && ln -sf /root/.local/miniconda3/etc/profile.d/conda.sh /etc/profile.d/conda.sh

# Install PyTorch and CUDA Toolkit
RUN conda install -y -c pytorch pytorch==1.10.1 torchvision==0.11.2 cudatoolkit==11.3.1

# Place stuff under /workspace
WORKDIR /workspace

# Snapshot of Zeus
ADD . /workspace/zeus

# When an outside zeus directory is mounted, have it apply immediately.
RUN pip install -e zeus

# Build and bake in the Zeus monitor.
RUN cd /workspace/zeus/zeus_monitor && cmake . && make && cp zeus_monitor /usr/local/bin/ && cd /workspace
