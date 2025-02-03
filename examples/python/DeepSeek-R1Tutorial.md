## 1. Pre-Requisites: Make a virtual environment and install ONNX Runtime GenAI
```bash
python -m venv .venv && source .venv/bin/activate
pip install requests numpy --pre onnxruntime-genai
```

```bash
python -m venv .venv && source .venv/bin/activate
pip install requests numpy --pre onnxruntime-genai-cuda "olive-ai[gpu]"
```

## 2. Acquire model

```bash
huggingface-cli download onnxruntime/DeepSeek-R1-Distill-ONNX --include 'deepseek-r1-distill-qwen-1.5B/*' --local-dir .
```
OR choose your model and convert to ONNX

```bash
olive auto-opt --model_name_or_path deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B --output_path ./deepseek-r1-distill-qwen-1.5B --device gpu --provider CUDAExecutionProvider --precision int4 --use_model_builder --log_level 1
```

## 3. Play with model

```bash
curl -s https://raw.githubusercontent.com/microsoft/onnxruntime-genai/refs/heads/main/examples/python/model-chat.py
python model-chat.py -m deepseek-r1-distill-qwen-1.5B -e gpu --chat_template "<|begin▁of▁sentence|><|User|>{input}<|Assistant|>"
```

```bash
curl -s <https://raw.githubusercontent.com/microsoft/onnxruntime-genai/refs/heads/main/examples/python/model-chat.py>
python model-chat.py -m deepseek-r1-distill-qwen-1.5B/cpu_and_mobile/cpu-int4-rtn-block-32-acc-level-4/ -e cpu
```
