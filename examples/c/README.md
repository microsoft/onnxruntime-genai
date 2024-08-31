# ONNX Runtime generate() API C Example

## Setup

Clone this repo and change into the examples/c folder.

```bash
git clone https://github.com/microsoft/onnxruntime-genai.git
cd onnxruntime-genai/examples/c
```

## Download a model

This example uses the [Phi-3 mini model](https://huggingface.co/microsoft/Phi-3-mini-4k-instruct) and the [Phi-3 vision model](https://huggingface.co/microsoft/Phi-3-mini-4k-instruct) optimized to run on CPU. You can clone this entire model repository or download individual model variants. To download individual variants, you need to install the HuggingFace CLI. For example:

```bash
huggingface-cli download microsoft/Phi-3-mini-4k-instruct-onnx --include cpu_and_mobile/cpu-int4-rtn-block-32-acc-level-4/* --local-dir .
```

## Install the onnxruntime and onnxruntime-genai binaries

### Windows

```
curl -L https://github.com/microsoft/onnxruntime/releases/download/v1.19.0/onnxruntime-win-x64-1.19.0.zip -o onnxruntime-win-x64-1.19.0.zip
tar xvf onnxruntime-win-x64-1.19.0.zip
copy onnxruntime-win-x64-1.19.0\include\* include
copy onnxruntime-win-x64-1.19.0\lib\* lib
curl -L https://github.com/microsoft/onnxruntime-genai/releases/download/v0.4.0/onnxruntime-genai-win-cpu-x64-capi.zip -o onnxruntime-genai-win-cpu-x64-capi.zip
tar xvf onnxruntime-genai-win-cpu-x64-capi.zip
cd onnxruntime-genai-win-cpu-x64-capi
tar xvf onnxruntime-genai-0.4.0-win-x64.zip
copy onnxruntime-genai-0.4.0-win-x64\include\* ..\include
copy onnxruntime-genai-0.4.0-win-x64\lib\* ..\lib
cd ..
``` 

### Linux

```
curl -L https://github.com/microsoft/onnxruntime/releases/download/v1.19.0/onnxruntime-linux-x64-1.19.0.tgz -o onnxruntime-linux-x64-1.19.0.tgz
tar xvzf onnxruntime-linux-x64-1.19.0.tgz
cp onnxruntime-linux-x64-1.19.0/include/* include
cp onnxruntime-linux-x64-1.19.0/lib/* lib
curl -L https://github.com/microsoft/onnxruntime-genai/releases/download/v0.4.0/onnxruntime-genai-linux-cpu-x64-capi.zip -o onnxruntime-genai-linux-cpu-x64-capi.zip
unzip onnxruntime-genai-linux-cpu-x64-capi.zip
cd onnxruntime-genai-linux-cpu-x64-capi
tar xvzf onnxruntime-genai-0.4.0-linux-x64.tar.gz
cp onnxruntime-genai-0.4.0-linux-x64/include/* ../include
cp onnxruntime-genai-0.4.0-linux-x64/lib/* ../lib
cd ..
```

## Build this sample

### Windows

```bash
cmake -G "Visual Studio 17 2022" -A x64 -S . -B build
cd build
cmake --build . --config Release
```

### Linux

Build with CUDA:

```bash
mkdir build
cd build
cmake ../ -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc -DCMAKE_CUDA_ARCHITECTURES=80 -DUSE_CUDA=ON
cmake --build . --config Release
```

Build for CPU:

```bash
cmake .
cd build
cmake --build . --config Release
```

## Run the sample

### Run the language model

```bash
cd build\\Release
.\phi3.exe path_to_model
```

### Run the vision model

```bash
cd build\\Release
.\phi3v.exe path_to_model
```

