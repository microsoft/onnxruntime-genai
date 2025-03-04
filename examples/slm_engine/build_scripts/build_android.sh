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

# Run the docker to build dependencies
docker run --rm -v \
    `pwd`/../../../:`pwd`/../../../  \
    -u $(id -u):$(id -g) -w `pwd`  \
    slm-engine-builder python3 build_deps.py \
    --build_ort_from_source \
    --android_sdk_path /opt/android-sdk/ \
    --android_ndk_path /opt/android-sdk/ndk/27.2.12479018/ 

# Next build the slm_engine
docker run --rm -v \
    `pwd`/../../../:`pwd`/../../../  \
    -u $(id -u):$(id -g) -w `pwd` \
    slm-engine-builder python3 build.py \
    --android_ndk_path /opt/android-sdk/ndk/27.2.12479018/ \
