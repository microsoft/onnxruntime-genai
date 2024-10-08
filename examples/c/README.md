# ONNX Runtime generate() API C Example

## Setup

Clone this repo and change into the `examples/c` folder.

```bash
git clone https://github.com/microsoft/onnxruntime-genai.git
cd onnxruntime-genai/examples/c
```

If they don't already exist, create folders called `include` and `lib`.

```bash
mkdir include
mkdir lib
```

## Phi-3 mini

### Download model

This example uses the [Phi-3 mini model](https://huggingface.co/microsoft/Phi-3-mini-4k-instruct).

You can clone this entire model repository or download individual model variants. To download individual variants, you need to install the HuggingFace CLI.

```bash
huggingface-cli download microsoft/Phi-3-mini-4k-instruct-onnx --include cpu_and_mobile/cpu-int4-rtn-block-32-acc-level-4/* --local-dir .
```

### Windows x64 CPU

#### Install the onnxruntime and onnxruntime-genai binaries

Change into the `onnxruntime-genai\examples\c` folder.

1. Install onnxruntime
   
   ```cmd
   curl -L https://github.com/microsoft/onnxruntime/releases/download/v1.19.2/onnxruntime-win-x64-1.19.2.zip -o onnxruntime-win-x64-1.19.2.zip
   tar xvf onnxruntime-win-x64-1.19.2.zip
   copy onnxruntime-win-x64-1.19.2\include\* include
   copy onnxruntime-win-x64-1.19.2\lib\* lib
   ```

2. Install onnxruntime-genai

   ```cmd
   curl -L https://github.com/microsoft/onnxruntime-genai/releases/download/v0.4.0/onnxruntime-genai-win-cpu-x64-capi.zip -o onnxruntime-genai-win-cpu-x64-capi.zip
   tar xvf onnxruntime-genai-win-cpu-x64-capi.zip
   cd onnxruntime-genai-win-cpu-x64-capi
   tar xvf onnxruntime-genai-0.4.0-win-x64.zip
   copy onnxruntime-genai-0.4.0-win-x64\include\* ..\include
   copy onnxruntime-genai-0.4.0-win-x64\lib\* ..\lib
   cd ..
   ```

#### Build this sample

```bash
cmake -A x64 -S . -B build -DPHI3=ON
cd build
cmake --build . --config Release
```

#### Run the sample

```bash
cd Release
.\phi3.exe path_to_model
```

### Windows x64 DirectML

#### Install the onnxruntime and onnxruntime-genai binaries

Change into the `onnxruntime-genai\examples\c` folder.

1. Install onnxruntime
   
   ```cmd
   mkdir onnxruntime-win-x64-directml
   cd onnxruntime-win-x64-directml
   curl -L https://github.com/microsoft/onnxruntime/releases/download/v1.19.2/Microsoft.ML.OnnxRuntime.DirectML.1.19.2.nupkg -o Microsoft.ML.OnnxRuntime.DirectML.1.19.2.nupkg
   tar xvf Microsoft.ML.OnnxRuntime.DirectML.1.19.2.nupkg
   copy build\native\include\* ..\include
   copy runtimes\win-x64\native\* ..\lib
   cd ..
   ```

2. Install onnxruntime-genai

   ```cmd
   curl -L https://github.com/microsoft/onnxruntime-genai/releases/download/v0.4.0/onnxruntime-genai-win-directml-x64-capi.zip -o onnxruntime-genai-win-directml-x64-capi.zip
   tar xvf onnxruntime-genai-win-directml-x64-capi.zip
   cd onnxruntime-genai-win-directml-x64-capi
   tar xvf onnxruntime-genai-0.4.0-win-x64-dml.zip
   copy onnxruntime-genai-0.4.0-win-x64-dml\include\* ..\include
   copy onnxruntime-genai-0.4.0-win-x64-dml\lib\* ..\lib
   cd ..
   ```

#### Build this sample

```bash
cmake -A x64 -S . -B build -DPHI3=ON
cd build
cmake --build . --config Release
```

#### Run the sample

```bash
cd Release
.\phi3.exe path_to_model
```

### Windows arm64 CPU

#### Install the onnxruntime and onnxruntime-genai binaries

Change into the `onnxruntime-genai\examples\c` folder.

1. Install onnxruntime
   
   ```cmd
   curl -L https://github.com/microsoft/onnxruntime/releases/download/v1.19.2/onnxruntime-win-arm64-1.19.2.zip -o onnxruntime-win-arm64-1.19.2.zip
   tar xvf onnxruntime-win-arm64-1.19.2.zip
   copy onnxruntime-win-arm64-1.19.2\include\* include
   copy onnxruntime-win-arm64-1.19.2\lib\* lib
   ```

2. Install onnxruntime-genai

   ```cmd
   curl -L https://github.com/microsoft/onnxruntime-genai/releases/download/v0.4.0/onnxruntime-genai-win-cpu-arm64-capi.zip -o onnxruntime-genai-win-cpu-arm64-capi.zip
   tar xvf onnxruntime-genai-win-cpu-arm64-capi.zip
   cd onnxruntime-genai-win-cpu-arm64-capi
   tar xvf onnxruntime-genai-0.4.0-win-arm64.zip
   copy onnxruntime-genai-0.4.0-win-arm64\include\* ..\include
   copy onnxruntime-genai-0.4.0-win-arm64\lib\* ..\lib
   cd ..
   ```

#### Build this sample

```bash
cmake -A arm64 -S . -B build -DPHI3=ON
cd build
cmake --build . --config Release
```

#### Run the sample

```bash
cd Release
.\phi3.exe path_to_model
```

### Windows arm64 DirectML

#### Install the onnxruntime and onnxruntime-genai binaries

Change into the `onnxruntime-genai\examples\c` folder.

1. Install onnxruntime
   
   ```cmd
   mkdir onnxruntime-win-arm64-directml
   cd onnxruntime-win-arm64-directml
   curl -L https://github.com/microsoft/onnxruntime/releases/download/v1.19.2/Microsoft.ML.OnnxRuntime.DirectML.1.19.2.nupkg -o Microsoft.ML.OnnxRuntime.DirectML.1.19.2.nupkg
   tar xvf Microsoft.ML.OnnxRuntime.DirectML.1.19.2.nupkg
   copy build\native\include\* ..\include
   copy runtimes\win-arm64\native\* ..\lib
   cd ..
   ```

2. Install onnxruntime-genai

   ```cmd
   curl -L https://github.com/microsoft/onnxruntime-genai/releases/download/v0.4.0/onnxruntime-genai-win-directml-arm64-capi.zip -o onnxruntime-genai-win-directml-arm64-capi.zip
   tar xvf onnxruntime-genai-win-directml-arm64-capi.zip
   cd onnxruntime-genai-win-directml-arm64-capi
   tar xvf onnxruntime-genai-0.4.0-win-arm64-dml.zip
   copy onnxruntime-genai-0.4.0-win-arm64-dml\include\* ..\include
   copy onnxruntime-genai-0.4.0-win-arm64-dml\lib\* ..\lib
   cd ..
   ```

#### Build this sample

```bash
cmake -A arm64 -S . -B build -DPHI3=ON
cd build
cmake --build . --config Release
```

#### Run the sample

```bash
cd Release
.\phi3.exe path_to_model
```

### Linux

#### Install the onnxruntime and onnxruntime-genai binaries

Change into the onnxruntime-genai directory.

1. Install onnxruntime

   ```bash
   cd examples/c
   curl -L https://github.com/microsoft/onnxruntime/releases/download/v1.19.2/onnxruntime-linux-x64-1.19.2.tgz -o onnxruntime-linux-x64-1.19.2.tgz
   tar xvzf onnxruntime-linux-x64-1.19.2.tgz
   cp onnxruntime-linux-x64-1.19.2/include/* include
   cp onnxruntime-linux-x64-1.19.2/lib/* lib
   cd ../..
   ```

2. Build onnxruntime-genai from source and install

   This example requires onnxruntime-genai to be built from source.

   ```bash
   # This should be run from the root of the onnxruntime-genai folder
   python build.py --config Release --ort_home examples/c
   cp src/ort_genai.h examples/c/include
   cp src/ort_genai_c.h examples/c/include
   cp build/Linux/Release/libonnxruntime-genai.so examples/c/lib
   cd examples/c
   ```

#### Build this sample

Build with CUDA:

```bash
mkdir build
cd build
cmake ../ -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc -DCMAKE_CUDA_ARCHITECTURES=80 -DUSE_CUDA=ON -DPHI3=ON
cmake --build . --config Release
```

Build for CPU:

```bash
mkdir build
cd build
cmake .. -DPHI3=ON
cmake --build . --config Release
```

#### Run the sample

```bash
./phi3 path_to_model
```

## Phi-3 vision

### Download model

You can use one of the following models for this sample:
* [Phi-3 vision model for CPU](https://huggingface.co/microsoft/Phi-3-vision-128k-instruct-onnx-cpu)
* [Phi-3 vision model for CUDA](https://huggingface.co/microsoft/Phi-3-vision-128k-instruct-onnx-cuda)
* [Phi-3 vision model for DirectML](https://huggingface.co/microsoft/Phi-3-vision-128k-instruct-onnx-directml)
  
Clone one of the models above.

### Run on Windows

#### Install the required headers and binaries

Change into the onnxruntime-genai folder.

1. Install onnxruntime
   
   ```cmd
   cd examples\c
   curl -L https://github.com/microsoft/onnxruntime/releases/download/v1.19.2/onnxruntime-win-x64-1.19.2.zip -o onnxruntime-win-x64-1.19.2.zip
   tar xvf onnxruntime-win-x64-1.19.2.zip
   copy onnxruntime-win-x64-1.19.2\include\* include
   copy onnxruntime-win-x64-1.19.2\lib\* lib
   ```

2. Install onnxruntime-genai

   This example requires onnxruntime-genai to be built from source.

   ```cmd
   cd ..\..
   python build.py --config Release --ort_home examples\c
   copy src\ort_genai.h examples\c\include
   copy src\ort_genai_c.h examples\c\include
   copy build\Windows\Release\Release\*.dll examples\c\lib
   copy build\Windows\Release\Release\*.lib examples\c\lib
   cd examples\c
   ```

#### Build this sample

```bash
cmake -G "Visual Studio 17 2022" -A x64 -S . -B build -DPHI3V=ON
cd build
cmake --build . --config Release
```

#### Run the sample

```bash
cd Release
.\phi3v.exe path_to_model
```

### Run on Linux

#### Install the required headers and binaries

Change into the onnxruntime-genai directory.

1. Install onnxruntime

   ```bash
   cd examples/c
   curl -L https://github.com/microsoft/onnxruntime/releases/download/v1.19.2/onnxruntime-linux-x64-1.19.2.tgz -o onnxruntime-linux-x64-1.19.2.tgz
   tar xvzf onnxruntime-linux-x64-1.19.2.tgz
   cp onnxruntime-linux-x64-1.19.2/include/* include
   cp onnxruntime-linux-x64-1.19.2/lib/* lib
   cd ../..
   ```

2. Build onnxruntime-genai from source and install

   This example requires onnxruntime-genai to be built from source.

   ```bash
   # This should be run from the root of the onnxruntime-genai folder
   python build.py --config Release --ort_home examples/c
   cp src/ort_genai.h examples/c/include
   cp src/ort_genai_c.h examples/c/include
   cp build/Linux/Release/libonnxruntime-genai.so examples/c/lib
   cd examples/c
   ```

#### Build this sample

Build to run with CUDA:

```bash
cmake . -B build -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc -DCMAKE_CUDA_ARCHITECTURES=80 -DUSE_CUDA=ON -DPHI3V=ON
cd build
cmake --build . --config Release
```

Build for CPU:

```bash
cmake . -B build -DPHI3V=ON
cd build
cmake --build . --config Release
```

#### Run the sample

```bash
cd build/Release
./phi3v path_to_model
```

