# Llama

## Clone this repo

```bash
git clone https://github.com/microsoft/onnxruntime-genai.git
cd onnxruntime-genai
```

## Install the onnxruntime-genai library

* (Temporary) Build the library according to [the build instructions](../README.md#build-from-source)

* Install the python package

  ```bash
  cd build/wheel
  pip install onnxruntime-genai-*.whl
  ```


## Get the model

```bash
cd examples/phi2
python -m onnxruntime_genai.models.builder.py -m meta-llama/Llama-2-7b-chat-hf -e cpu -p int4 -o model/model.onnx
```

## Run Llama

```bash
python llama.py
```