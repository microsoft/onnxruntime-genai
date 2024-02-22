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

```bash
python -m onnxruntime_genai.models.builder -m microsoft/phi-2 -e cpu -p int4 -o model/model.onnx
```

## Run the phi-2 model

```bash
python phi2.py
```

