# Run the Phi-3 Mini models with the ONNX Runtime generate() API

## Steps
1. [Download Phi-3 Mini](#download-the-model)
2. [Build ONNX Runtime shared libraries](#build-onnx-runtime-from-source)
3. [Build generate() API](#build-the-generate-api-from-source)
4. [Run Phi-3 Mini](#run-the-model)

## Download the model 

Download either or both of the [short](https://aka.ms/phi3-mini-4k-instruct-onnx) and [long](https://aka.ms/phi3-mini-128k-instruct-onnx) context Phi-3 mini models from Hugging Face.

There are ONNX models for CPU (used for mobile too), as well as DirectML and CUDA.


## Install the generate() API package

Right now, both `onnxruntime` and `onnxruntime-genai` need to be built from source. Once packages are published, this tutorial will be updated.

The instructions for how to build both packages from source are documented in the [build from source](https://onnxruntime.ai/docs/genai/howto/build-from-source.html) guide. They are repeated here for your convenience.

### Pre-requisites

#### CMake

This is included on Windows if you have Visual Studio installed. If you are running on Linux or Mac, you can install it using `conda`.

```bash
conda install cmake
```

### Build ONNX Runtime from source

#### Clone the repo 

```bash
git clone https://github.com/microsoft/onnxruntime.git
cd onnxruntime
```

#### Build ONNX Runtime for DirectML on Windows

```bash
build.bat --build_shared_lib --skip_tests --parallel --use_dml --config Release
```

#### Build ONNX Runtime for CPU on Windows

```bash
build.bat --build_shared_lib --skip_tests --parallel --config Release
```

#### Build ONNX Runtime for CUDA on Windows

```bash
build.bat --build_shared_lib --skip_tests --parallel --use_dml --config Release
```

#### Build ONNX Runtine on Linux

```bash
./build.sh --build_shared_lib --skip_tests --parallel [--use_cuda] --config Release
```

You may need to provide extra command line options for building with CUDA on Linux. An example full command is as follows.

```bash
./build.sh --parallel --build_shared_lib --use_cuda --cuda_version 11.8 --cuda_home /usr/local/cuda-11.8 --cudnn_home /usr/lib/x86_64-linux-gnu/ --config Release --build_wheel --skip_tests --cmake_extra_defines CMAKE_CUDA_ARCHITECTURES="80" --cmake_extra_defines CMAKE_CUDA_COMPILER=/usr/local/cuda-11.8/bin/nvcc
```

Replace the values given above for different versions and locations of CUDA.

#### Build ONNX Runtime on Mac

```bash
./build.sh --build_shared_lib --skip_tests --parallel --config Release
```

### Build the generate() API from source

#### Clone the repo

```bash
git clone https://github.com/microsoft/onnxruntime-genai
cd onnxruntime-genai
mkdir -p ort/include
mkdir -p ort/lib
```

#### Build the generate() API on Windows


If building for DirectML

```bash
copy ..\onnxruntime\include\onnxruntime\core\providers\dml\dml_provider_factory.h ort\include
```

```bash
copy ..\onnxruntime\include\onnxruntime\core\session\onnxruntime_c_api.h ort\include
copy ..\onnxruntime\build\Windows\Release\Release\*.dll ort\lib
copy ..\onnxruntime\build\Windows\Release\Release\onnxruntime.lib ort\lib
python build.py [--use_dml | --use_cuda]
cd build\wheel
pip install *.whl
```


#### Build the generate() API on Linux

```bash
cp ../onnxruntime/include/onnxruntime/core/session/onnxruntime_c_api.h ort/include
cp ../onnxruntime/build/Linux/Release/libonnxruntime*.so* ort/lib
python build.py [--use_cuda]
cd build/wheel
pip install *.whl
```

#### Build the generate() API on Mac

```bash
cp ../onnxruntime/include/onnxruntime/core/session/onnxruntime_c_api.h ort/include
cp ../onnxruntime/build/MacOS/Release/libonnxruntime*.dylib* ort/lib
python build.py
cd build/wheel
pip install *.whl
```

## Run the model

Run the model with [this script](https://github.com/microsoft/onnxruntime-genai/blob/main/examples/python/model-qa.py).

The script accepts a model folder and takes the generation parameters from the config in that model folder. You can also override the parameters on the command line.

```bash
python model-qa.py -m models/phi3-mini-4k-instruct-cpu-int4-rtn-block-32 
```

Once the script has loaded the model, it will ask you for input in a loop, streaming the output as it is produced the model. For example:

```bash
Input: <|user|>Tell me a joke<|end|><|assistant|>
 
Output:  Why don't scientists trust atoms?
 
Because they make up everything!
 
This joke plays on the double meaning of "make up." In science, atoms are the fundamental building blocks of matter, literally making up everything. However, in a colloquial sense, "to make up" can mean to fabricate or lie, hence the humor.
```
