# Phi-2

## Get the model

```bash
python -m onnxruntime_genai.scripts.export.py -m microsoft/phi-2 -e cpu -p int4 -o model/model.onnx
```

## Run the phi-2 model

```bash
python phi2.py
```
