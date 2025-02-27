#!/bin/sh
set -e
set -x
set -u

# Build the docker image 
docker build -t slm-engine-builder -f Dockerfile.Linux .

# Run the docker to build dependencies
docker run --rm -v `pwd`/../../../:`pwd`/../../../  -u $(id -u):$(id -g) -w `pwd`  slm-engine-builder python3 build_deps.py

# Next build the slm_engine
docker run --rm -v `pwd`/../../../:`pwd`/../../../  -u $(id -u):$(id -g) -w `pwd`  slm-engine-builder python3 build.py