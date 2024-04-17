# Gen-AI C Phi-2 Example

## Install the onnxruntime-genai library

* Install the python package

  ```bash
  pip install onnxruntime-genai
  ```

## Get the model

Install the model builder script dependencies

```bash
pip install numpy
pip install transformers
pip install torch
pip install onnx
pip install onnxruntime
```

Run the model builder script to export, optimize, and quantize the model. More details can be found [here](../../src/python/py/models/README.md)

```bash
cd examples\\phi2\\c
python -m onnxruntime_genai.models.builder -m microsoft/phi-2 -p int4 -e cpu -o phi-2\
```

## Build the CMake Project

1. Copy over all the dlls and libs over to the [lib](lib) directory.
  - onnxruntime.dll
  - onnxruntime_providers_shared.dll
  - onnxruntime_providers_cuda.dll
  - onnxruntime-genai.dll
  - onnxruntime-genai.lib
2. Copy over the `ort_genai.h` and `ort_genai_c.h` header files to the [include](include) directory.

On Windows:
```bash
cmake -G "Visual Studio 17 2022" -A x64 -S . -B build
cd build
cmake --build . --config Release
```

## Run the Phi-2 Model

```bash
cd build\\Release
.\phi2.exe path_to_model
```
