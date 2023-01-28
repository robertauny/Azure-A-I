############################################################################
# Begin license text.
# Copyright Feb. 27, 2020 Robert A. Murphy

# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

# End license text.
############################################################################

############################################################################
##
## File:      Dockerfile
##
## Purpose:   Recipe for the mtl.py docker file image creation
##
## Parameter: N/A
##
## Creator:   Robert A. Murphy
##
## Date:      Nov. 22, 2021
##
############################################################################

################################## BEGIN CUDA
FROM ubuntu:22.04 as base

FROM base as base-amd64

ENV NVARCH x86_64

ENV NVIDIA_REQUIRE_CUDA "cuda>=12.0 brand=tesla,driver>=450,driver<451 brand=tesla,driver>=470,driver<471 brand=unknown,driver>=470,driver<471 brand=nvidia,driver>=470,driver<471 brand=nvidiartx,driver>=470,driver<471 brand=geforce,driver>=470,driver<471 brand=geforcertx,driver>=470,driver<471 brand=quadro,driver>=470,driver<471 brand=quadrortx,driver>=470,driver<471 brand=titan,driver>=470,driver<471 brand=titanrtx,driver>=470,driver<471 brand=tesla,driver>=510,driver<511 brand=unknown,driver>=510,driver<511 brand=nvidia,driver>=510,driver<511 brand=nvidiartx,driver>=510,driver<511 brand=geforce,driver>=510,driver<511 brand=geforcertx,driver>=510,driver<511 brand=quadro,driver>=510,driver<511 brand=quadrortx,driver>=510,driver<511 brand=titan,driver>=510,driver<511 brand=titanrtx,driver>=510,driver<511 brand=tesla,driver>=515,driver<516 brand=unknown,driver>=515,driver<516 brand=nvidia,driver>=515,driver<516 brand=nvidiartx,driver>=515,driver<516 brand=geforce,driver>=515,driver<516 brand=geforcertx,driver>=515,driver<516 brand=quadro,driver>=515,driver<516 brand=quadrortx,driver>=515,driver<516 brand=titan,driver>=515,driver<516 brand=titanrtx,driver>=515,driver<516 brand=tesla,driver>=520,driver<521 brand=unknown,driver>=520,driver<521 brand=nvidia,driver>=520,driver<521 brand=nvidiartx,driver>=520,driver<521 brand=geforce,driver>=520,driver<521 brand=geforcertx,driver>=520,driver<521 brand=quadro,driver>=520,driver<521 brand=quadrortx,driver>=520,driver<521 brand=titan,driver>=520,driver<521 brand=titanrtx,driver>=520,driver<521"
ENV NV_CUDA_CUDART_VERSION 12.0.107-1
ENV NV_CUDA_COMPAT_PACKAGE cuda-compat-12-0

FROM base as base-arm64

ENV NVARCH sbsa
ENV NVIDIA_REQUIRE_CUDA "cuda>=12.0"
ENV NV_CUDA_CUDART_VERSION 12.0.107-1

FROM base-${TARGETARCH}

ARG TARGETARCH

LABEL maintainer "NVIDIA CORPORATION <cudatools@nvidia.com>"

RUN apt-get update && apt-get install -y --no-install-recommends \
    gnupg2 curl ca-certificates && \
    curl -fsSLO https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/${NVARCH}/cuda-keyring_1.0-1_all.deb && \
    dpkg -i cuda-keyring_1.0-1_all.deb && \
    apt-get purge --autoremove -y curl \
    && rm -rf /var/lib/apt/lists/*

ENV CUDA_VERSION 12.0.0

# For libraries in the cuda-compat-* package: https://docs.nvidia.com/cuda/eula/index.html#attachment-a
RUN apt-get update && apt-get install -y --no-install-recommends \
    cuda-cudart-12-0=${NV_CUDA_CUDART_VERSION} \
    ${NV_CUDA_COMPAT_PACKAGE} \
    && rm -rf /var/lib/apt/lists/*

# Required for nvidia-docker v1
RUN echo "/usr/local/nvidia/lib" >> /etc/ld.so.conf.d/nvidia.conf \
    && echo "/usr/local/nvidia/lib64" >> /etc/ld.so.conf.d/nvidia.conf

ENV PATH /usr/local/nvidia/bin:/usr/local/cuda/bin:${PATH}
ENV LD_LIBRARY_PATH /usr/local/nvidia/lib:/usr/local/nvidia/lib64

COPY NGC-DL-CONTAINER-LICENSE /

# nvidia-container-runtime
ENV NVIDIA_VISIBLE_DEVICES all
ENV NVIDIA_DRIVER_CAPABILITIES compute,utility

################################## END CUDA

# other requirements that need to be done
#
# create your own account <accountname> and repo <reponame> on docker hub
#
# install either docker desktop for windows or ubuntu
#
# docker run --name repo alpine/git clone https://github.com/robertauny:Azure-A-I.git
#
# docker cp repo:/git/Azure-A-I/ .
#
# cd Azure-A-I
#
# docker build -t <accountname>/<reponame>:<tag> . e.g. docker build -t robertauny/python-mtl:latest .
#
# docker run python-mtl

# version of python to use
FROM python:3.9 
# working directory where the python modules can be found
WORKDIR /Services
# make some needed directories
RUN mkdir -p ./models ./data/images ./data/csv
# copy the application files from the host to the docker working directory
COPY Services/*.py Services/*.json .
# copy the python module requirements from the host to the docker working directory
COPY pip_requirements.txt .
# install python modules
RUN pip install -r pip_requirements.txt
# execution of the python module in mtl.py
CMD python ./dataset.py
# copy the data file from the host to the docker working directory
COPY Services/data/csv/clicks.csv ./data/csv
# execution of the python module in mtl.py
CMD python ./mtl.py
