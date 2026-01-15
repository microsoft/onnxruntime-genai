# Run GPT OSS 20B with Web GPU

## Model

Download the model

```bash
https://huggingface.co/onnx-community/gpt-oss-20b-ONNX
```

Copy the files in the onnx folder into the parent folder

Change the name of model_q4f16.onnx into model.onnx (as referenced in genai_config.json)

## Runtime

```bash
pip install onnxruntime-webgpu
pip install onnxruntime-genai
pip uninstall onnxruntime (<-- we have some dependency clean up to do)
```

## Example script

```bash
curl -O https://raw.githubusercontent.com/microsoft/onnxruntime-genai/refs/heads/main/examples/python/model-qa.py
```

## Run the model

```bash
python model-qa.py -m gpt-oss-20b-ONNX
```

Note that the reasoning tokens are not processed by this script and appear in the output.

```bash
Prompt (Use quit() to exit): Hello

Output: <|channel|>analysis<|message|>We need to respond to "Hello". Simple greeting. Probably a short hello back and ask how can help.<|end|><|start|>assistant<|channel|>final<|message|>Hello! 👋 How can I help you today?

Prompt (Use quit() to exit): 
```
