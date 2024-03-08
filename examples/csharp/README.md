# Phi-2

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
cd examples\\phi2\\csharp
python -m onnxruntime_genai.models.builder -m microsoft/phi-2 -p int4 -e cpu -o phi-2\
```

## Run the phi-2 model

Open [HelloPhi2.sln](HelloPhi2.sln) and run the console application.
