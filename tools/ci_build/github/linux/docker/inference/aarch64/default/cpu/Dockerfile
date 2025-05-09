# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

FROM onnxruntimebuildcache.azurecr.io/internal/azureml/onnxruntime/build/cpu_aarch64_ubi8_gcc14_dotnet:20250124.1

ADD scripts /tmp/scripts
RUN cd /tmp/scripts && /tmp/scripts/install_centos.sh && /tmp/scripts/install_deps.sh && rm -rf /tmp/scripts
RUN yum update -y && yum install -y pkgconfig openssl-devel

ENV PATH="/usr/.cargo/bin:$PATH"
ENV RUSTUP_HOME="/usr/.rustup"
ENV CARGO_HOME="/usr/.cargo"
ARG BUILD_UID=1001
ARG BUILD_USER=onnxruntimedev
RUN adduser --uid $BUILD_UID $BUILD_USER
WORKDIR /home/$BUILD_USER
USER $BUILD_USER
