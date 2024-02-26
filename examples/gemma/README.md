# Gemma

## Clone this repo

```bash
git clone https://github.com/microsoft/onnxruntime-genai.git
cd onnxruntime-genai
```

## Install the onnxruntime-genai library

* (Temporary) Build the library according to [the build instructions](../../README.md#build-from-source)

* Install the python package

  ```bash
  cd build/wheel
  pip install onnxruntime-genai-*.whl
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
cd examples/gemma
python -m onnxruntime_genai.models.builder -m google/gemma-2b -e cuda -p fp16 -o ./example-models/gemma-2b-cuda
```

## Run gemma

```bash
python gemma.py
```