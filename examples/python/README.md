# Phi-2

## Install the onnxruntime-genai library

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
pip install onnxruntime-gpu
```

Choose a model. Examples of supported ones are:
- Phi-2
- Mistral
- Gemma 2B IT
- LLama 7B

Run the model builder script to export, optimize, and quantize the model. More details can be found [here](../../src/python/py/models/README.md)

```bash
cd examples/python
python -m onnxruntime_genai.models.builder -m microsoft/phi-2 -e cpu -p int4 -o ./example-models/phi2-int4-cpu
```

## Run the example model script

To run the chat example...

```bash
python model-chat.py -m {path to model folder} -ep {cpu or cuda}
```

To run the token generation loop example...

```bash
python model-generate.py -m {path to model folder} -ep {cpu or cuda} -i {string prompt}
```

