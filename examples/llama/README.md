# Llama

## Get model

```bash
python -m onnxruntime_genai.scripts.export.py -m microsoft/phi-2 -e cpu -p int4 -o model/model.onnx
```

## Run Llama

```bash
python llama.py
```