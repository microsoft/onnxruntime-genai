#!/bin/sh

# This script builds the slm_engine for Linux using docker.
# It uses the Dockerfile in the current directory to build a docker image   
# that contains all the necessary dependencies for building the slm_engine.
# The script then runs the docker image to build the slm_engine.
# The script assumes that the Dockerfile is in the same directory as this script.
# The script also assumes that the docker is installed and running on the host machine.

set -e
set -x
set -u

# Build the docker image 
docker build -t slm-engine-builder -f Dockerfile .

# Run the docker to build dependencies
docker run --rm -v \
    `pwd`/../../../:`pwd`/../../../  \
    -u $(id -u):$(id -g) -w `pwd`  \
    slm-engine-builder python3 build_deps.py

# Next build the slm_engine
docker run --rm -v \
    `pwd`/../../../:`pwd`/../../../  \
    -u $(id -u):$(id -g) -w `pwd` \
    slm-engine-builder python3 build.py