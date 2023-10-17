#!/bin/bash

git submodule update --init --recursive

cuda_version="cuda"
cuda_arch=70

export CUDA_HOME="/usr/local/${cuda_version}"
export CUDNN_HOME="/usr/local/${cuda_version}"
cuda_compiler="/usr/local/${cuda_version}/bin/nvcc"

cd src
cmake -S . -B build -DCMAKE_CUDA_ARCHITECTURES=$cuda_arch -DCMAKE_CUDA_COMPILER=$cuda_compiler -DCMAKE_POSITION_INDEPENDENT_CODE=ON
cd build
make
