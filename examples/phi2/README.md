# Phi-2

## Install the onnxruntime-genai library

* (Temporary) Build the library according to [the build instructions](../README.md#build-from-source)

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

Run the model builder script to load, optimize, quantize and export the model. More details can be found [here](../../src/python/py/models/README.md)

```bash
python -m onnxruntime_genai.models.builder -m microsoft/phi-2 -e cpu -p int4 -o model
```

## Run the phi-2 model

```bash
python phi2.py
```

