# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

ARG BASEIMAGE=nvidia/cuda:12.2.2-cudnn8-devel-ubi8
FROM $BASEIMAGE

ENV PATH="/opt/rh/gcc-toolset-12/root/usr/bin:${PATH}"
ENV LANG=en_US.UTF-8
ENV LC_ALL=en_US.UTF-8

# Set up CUDA and CUDNN environment variables
ENV CUDA_HOME=/usr/local/cuda
ENV CUDNN_HOME=/usr/lib/x86_64-linux-gnu/
ENV PATH="/usr/local/cuda/bin:${PATH}"
ENV LD_LIBRARY_PATH="/usr/local/cuda/lib64:${LD_LIBRARY_PATH}"
ENV CUDAHOSTCXX=/opt/rh/gcc-toolset-12/root/usr/bin/g++

# Install dependencies
ADD scripts /tmp/scripts
RUN cd /tmp/scripts && /tmp/scripts/install_centos.sh && /tmp/scripts/install_deps.sh && rm -rf /tmp/scripts

ARG BUILD_UID=1001
ARG BUILD_USER=onnxruntimedev
RUN adduser --uid $BUILD_UID $BUILD_USER
WORKDIR /home/$BUILD_USER
USER $BUILD_USER
