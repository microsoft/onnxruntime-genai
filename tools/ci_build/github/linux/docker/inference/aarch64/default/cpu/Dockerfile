# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

ARG BASEIMAGE=arm64v8/almalinux:8
FROM $BASEIMAGE

ENV PATH="/opt/rh/gcc-toolset-12/root/usr/bin:${PATH}"
ENV LANG=en_US.UTF-8
ENV LC_ALL=en_US.UTF-8

# Setup vcpkg environment variables
ENV VCPKG_DEFAULT_TRIPLET=arm64-linux
ENV VCPKG_FORCE_SYSTEM_BINARIES=1
ENV VCPKG_ROOT=/onnxruntime_src/extern/vcpkg
ENV PATH="${VCPKG_ROOT}:${PATH}"
ENV CMAKE_C_COMPILER=/opt/rh/gcc-toolset-12/root/usr/bin/gcc
ENV CMAKE_CXX_COMPILER=/opt/rh/gcc-toolset-12/root/usr/bin/g++

# Install dependencies
ADD scripts /tmp/scripts
RUN cd /tmp/scripts && /tmp/scripts/install_centos.sh && /tmp/scripts/install_deps.sh && rm -rf /tmp/scripts

ARG BUILD_UID=1001
ARG BUILD_USER=onnxruntimedev
RUN adduser --uid $BUILD_UID $BUILD_USER
WORKDIR /home/$BUILD_USER
USER $BUILD_USER

#RUN echo "alias aarch64-linux-gnu-gcc=gcc" >> /home/${BUILD_USER}/.bashrc
#RUN echo "alias aarch64-linux-gnu-g++=g++" >> /home/${BUILD_USER}/.bashrc
#RUN source /home/{$BUILD_USER}/.bashrc