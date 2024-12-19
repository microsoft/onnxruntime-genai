# Create AWQ-quantized and optimized ONNX models from PyTorch models with AutoAWQ + ONNX Runtime GenAI

## Steps
1. [Download your PyTorch model](#1-download-your-pytorch-model)
2. [Install AutoAWQ](#2-install-autoawq)
3. [Install ONNX Runtime GenAI](#3-install-onnx-runtime-genai)
    - [CPU](#cpu)
    - [CUDA](#cuda)
    - [DirectML](#directml)
4. [Run example script](#4-run-example-script)

## Introduction

Activation-aware Weight Quantization (AWQ) works by identifying the top 1% most salient weights that are most important for maintaining accuracy and quantizing the remaining 99% of weights. This leads to less accuracy loss from quantization compared to many other quantization techniques. For more on AWQ, see [here](https://arxiv.org/abs/2306.00978).

This tutorial downloads the Phi-3 mini short context PyTorch model, applies AWQ quantization, generates the corresponding optimized & quantized ONNX model, and runs the ONNX model with ONNX Runtime GenAI. If you would like to use another model, please change the model name in the instructions below.

## 1. Download your PyTorch model

You can use Hugging Face's `from_pretrained` and `save_pretrained` methods to download and save the base PyTorch model.

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
import os

# Change these as needed
model_name = "microsoft/Phi-3-mini-4k-instruct"
cache_dir = os.path.join(".", "cache_dir")
output_dir = os.path.join(".", "phi3-mini-4k-instruct")

tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
model = AutoModelForCausalLM.from_pretrained(model_name, cache_dir=cache_dir)

tokenizer.save_pretrained(output_dir)
model.save_pretrained(output_dir)
```

Alternatively, you can use your own PyTorch model that can be loaded with Hugging Face's APIs.

## 2. Install AutoAWQ

```bash
$ git clone https://github.com/casper-hansen/AutoAWQ
$ cd AutoAWQ
$ pip install -e .
```

Note: You can try to install AutoAWQ directly with `pip install autoawq`. However, AutoAWQ will try to auto-detect the CUDA version installed on your machine. If the CUDA version it detects is incorrect, the `.whl` file that `pip` will choose will be incorrect. This will cause an error during runtime when trying to quantize. Thus, it is recommended to install AutoAWQ from source to get the right `.whl` file.

## 3. Install ONNX Runtime GenAI

Based on your desired hardware target, pick from one of the following options to install ONNX Runtime GenAI.

### CPU
```bash
$ pip install onnxruntime-genai
```

### CUDA
```bash
$ pip install onnxruntime-genai-cuda
```

### DirectML
```bash
$ pip install onnxruntime-genai-directml
```

You should now see `onnxruntime-genai` in your `pip list`.

## 4. Run example script

Run your PyTorch model with [awq-quantized-model.py](https://github.com/microsoft/onnxruntime-genai/blob/main/examples/python/awq-quantized-model.py).

```bash
# Get example script
$ curl https://raw.githubusercontent.com/microsoft/onnxruntime-genai/main/examples/python/awq-quantized-model.py -o awq-quantized-model.py

# Run example script
$ python awq-quantized-model.py --model_path /path/to/folder/containing/your/pytorch/model/ --quant_path /path/to/new/folder/to/save/quantized/pytorch/model/in/ --output_path /path/to/new/folder/to/save/quantized/and/optimized/onnx/model/in/ --execution_provider [dml|cuda]

# Example for DirectML:
# $ python awq-quantized-model.py --model_path microsoft/Phi-3-mini-4k-instruct --quant_path ./phi3-mini-4k-instruct-awq/ --output_path ./phi3-mini-4k-instruct-awq-onnx/ --execution_provider dml

# Example for CUDA:
# $ python awq-quantized-model.py --model_path microsoft/Phi-3-mini-4k-instruct --quant_path ./phi3-mini-4k-instruct-awq/ --output_path ./phi3-mini-4k-instruct-awq-onnx/ --execution_provider cuda
```

Once the ONNX model has been created and the script has loaded the model, it will ask you for input in a loop, streaming the output as it is produced the model. For example:

```bash
Input: What color is the sky?

Output:  The color of the sky can appear blue during a clear day due to Rayleigh scattering, which scatters shorter wavelengths of light (blue) more than longer wavelengths (red). However, the sky can also appear in various colors at sunrise and sunset, such as orange, pink, or purple, due to the scattering of light by the atmosphere when the sun is low on the horizon. Additionally, the sky can appear black at night when there is no sunlight.
```
