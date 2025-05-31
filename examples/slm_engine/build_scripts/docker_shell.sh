#!/bin/sh

# This script builds the slm_engine for Android using docker.
# It uses the Dockerfile in the current directory to build a docker image
# that contains all the necessary dependencies for building the slm_engine.
# The script then runs the docker image to build the slm_engine.
# The script assumes that the Dockerfile is in the same directory as this script.
# The script also assumes that the android-sdk and android-ndk are installed
# in the /opt/android-sdk directory.
# 

# Check the architecture
if [ "$(uname -m)" != "x86_64" ]; then
    echo "This script is intended to run on x86_64 architecture only."
    exit 1
fi

set -e
set -x
set -u

# Build the docker image 
docker build -t slm-engine-builder -f Dockerfile .

# Define base build_deps command
BUILD_DEPS_CMD="python3 build_deps.py \
    --build_ort_from_source \
    --android_sdk_path /opt/android-sdk/ \
    --android_ndk_path /opt/android-sdk/ndk/27.2.12479018/"

# Docker volume mount options
VOLUME_MOUNTS="-v `pwd`/../../../:`pwd`/../../../"

# Check if USE_ORT_VERSION is defined
if [ ! -z "${USE_ORT_VERSION:-}" ]; then
    BUILD_DEPS_CMD="$BUILD_DEPS_CMD --ort_version_to_use $USE_ORT_VERSION"
    echo "Using ONNX Runtime version: $USE_ORT_VERSION"
fi

# Check if QNN_SDK_HOME is defined
if [ ! -z "${QNN_SDK_HOME:-}" ]; then
    # Create Docker mount point for QNN SDK
    QNN_SDK_DOCKER_PATH="/opt/qnn_sdk"
    
    # Add mount for QNN SDK
    VOLUME_MOUNTS="$VOLUME_MOUNTS -v $QNN_SDK_HOME:$QNN_SDK_DOCKER_PATH"
    
    # Use the Docker path in build command
    BUILD_DEPS_CMD="$BUILD_DEPS_CMD --qnn_sdk_path $QNN_SDK_DOCKER_PATH"
    
    echo "QNN SDK path detected, building with QNN support"
    echo "Mounting $QNN_SDK_HOME to $QNN_SDK_DOCKER_PATH in container"
fi

# Run the docker to build dependencies
docker run --rm $VOLUME_MOUNTS \
    -u $(id -u):$(id -g) -w `pwd` \
    -w $HOME \
    -it slm-engine-builder bash
