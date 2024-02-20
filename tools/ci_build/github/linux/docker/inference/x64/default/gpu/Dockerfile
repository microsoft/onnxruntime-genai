# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

# This file is used by Zip-Nuget Packaging NoContribOps Pipeline,Zip-Nuget-Java Packaging Pipeline
ARG BASEIMAGE=nvidia/cuda:11.8.0-cudnn8-devel-ubi8
FROM $BASEIMAGE

ENV PATH="/usr/local/cuda-11.8/bin:/opt/rh/gcc-toolset-12/root/usr/bin:${PATH}"
ENV LANG=en_US.UTF-8
ENV LC_ALL=en_US.UTF-8
ENV CUDA_HOME=/usr/local/cuda-11.8
ENV CUDA_PATH=/usr/local/cuda-11.8
ENV CUDNN_HOME=/usr/local/cuda-11.8
ENV LD_LIBRARY_PATH="/usr/local/cuda-11.8/lib64:${LD_LIBRARY_PATH}"
ENV CUDAHOSTCXX=/opt/rh/gcc-toolset-12/root/usr/bin/g++
ADD scripts /tmp/scripts
RUN cd /tmp/scripts && /tmp/scripts/install_centos.sh && /tmp/scripts/install_deps.sh && rm -rf /tmp/scripts

ARG BUILD_UID=1001
ARG BUILD_USER=onnxruntimedev
RUN adduser --uid $BUILD_UID $BUILD_USER
WORKDIR /home/$BUILD_USER
USER $BUILD_USER
