# Phi-2

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
wget https://github.com/microsoft/onnxruntime-genai/blob/main/src/python/models/export.py
python export.py -m microsoft/phi-2 -e cpu -p int4 -o model/model.onnx
```

## Run the phi-2 model

```bash
python phi2.py
```
