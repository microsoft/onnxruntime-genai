Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved. 

# Build your Qwen3-VL ONNX models for ONNX Runtime GenAI

## Steps
0. [Pre-requisites](#pre-requisites)
1. [Prepare Local Workspace](#prepare-local-workspace)
2. [Download necessary artifacts](#2-download-necessary-artifacts)
3. [Download a model from Hugging Face](#3-download-a-model-from-hugging-face)
4. [Export ONNX package](#4-export-onnx-package)
5. [Sanity test: text-only](#5-sanity-test-text-only)
6. [Sanity test: image + text](#6-sanity-test-image--text)
7. [Notes](#7-notes)
8. [Citation](#8-citation)

## 0. Pre-requisites

Please ensure you have the following Python packages installed to create the ONNX models.

- `backoff`
- `huggingface_hub[cli]`
- `numpy`
    - Please ensure that your `numpy` version is less than 2.0.0 after installing all of the pre-requisite packages. If it is greater than or equal to 2.0.0, please uninstall `numpy` with `pip uninstall -y numpy` and install an older version (e.g. `pip install numpy==1.26.4`).
- `onnx`
- `onnxruntime` and `onnxruntime-genai`
    - ONNX Runtime: Please install the latest nightly version. To ensure the right version is installed, please install ONNX Runtime GenAI first. Then you can uninstall the stable version of ONNX Runtime that gets auto-installed as a dependency.
    - ONNX Runtime GenAI: Please install the latest stable release package.

    - For CPU:
    ```bash
    # 1. Install ONNX Runtime GenAI wheel
    pip install onnxruntime-genai

    # 2. Uninstall stable version of ONNX Runtime that is auto-installed by ONNX Runtime GenAI
    pip uninstall -y onnxruntime

    # 3. Install nightly version of ONNX Runtime
    pip install -i https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/ORT-Nightly/pypi/simple/ --pre onnxruntime
    ```

    - For CUDA:
    ```bash
    # 1. Install ONNX Runtime GenAI wheel
    pip install onnxruntime-genai-cuda

    # 2. Uninstall stable version of ONNX Runtime that is auto-installed by ONNX Runtime GenAI
    pip uninstall -y onnxruntime-gpu

    # 3. Install nightly version of ONNX Runtime
    pip install -i https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/ORT-Nightly/pypi/simple/ --pre onnxruntime-gpu
    ```

    - For DirectML: 
    ```bash
    # 1. Install ONNX Runtime GenAI wheel
    pip install onnxruntime-genai-directml

    # 2. Uninstall stable version of ONNX Runtime that is auto-installed by ONNX Runtime GenAI
    pip uninstall -y onnxruntime-directml

    # 3. Install nightly version of ONNX Runtime
    pip install -i https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/ORT-Nightly/pypi/simple/ --pre onnxruntime-directml
    ```
- `onnxscript`
- `peft`
- `pillow`
- `requests`
- `scipy`
- `soundfile`
- `torch`
    - Please install the latest nightly version. You can install torch by following the [instructions](https://pytorch.org/get-started/locally/). For getting ONNX models that can run on CUDA or DirectML, please install torch with CUDA and ensure the CUDA version you choose in the instructions is the one you have installed.
    - For CPU:
    ```bash
    pip install torch
    ```
    - For CUDA:
    ```bash
    pip install torch --index-url https://download.pytorch.org/whl/nightly/cu124
    ```
- `torchaudio`
    - Please install the latest nightly version. You can install torchaudio by following the [instructions](https://pytorch.org/get-started/locally/). For getting ONNX models that can run on CUDA or DirectML, please install torchaudio with CUDA and ensure the CUDA version you choose in the instructions is the one you have installed.
    - For CPU:
    ```bash
    pip install torchaudio
    ```
    - For CUDA:
    ```bash
    pip install torchaudio --index-url https://download.pytorch.org/whl/nightly/cu124
    ```
- `torchvision` 
    - Please install the latest nightly version. You can install torchvision by following the [instructions](https://pytorch.org/get-started/locally/). For getting ONNX models that can run on CUDA or DirectML, please install torchvision with CUDA and ensure the CUDA version you choose in the instructions is the one you have installed.
    - For CPU:
    ```bash
    pip install torchvision
    ```
    - For CUDA:
    ```bash
    pip install torchvision --index-url https://download.pytorch.org/whl/nightly/cu124
    ```
- `transformers`

## 1. Prepare Local Workspace

Qwen3-VL has different size models, 2B, 4B, 8B and 32B. To get these ONNX models, the original PyTorch modeling files have to be modified.

## 2. Download necessary artifacts

Run commands from the `onnxruntime-genai` repo root:

```powershell
cd examples/python
mkdir qwen3-vl
cd qwen3-vl
mkdir ./pytorch_reference
hf download onnx-community/Qwen3-4B-VL-ONNX --include "modeling_qwen3_vl.py" --local-dir "./pytorch_reference"
hf download onnx-community/Qwen3-4B-VL-ONNX --include "builder.py" --local-dir "."
hf download onnx-community/Qwen3-4B-VL-ONNX --include "qwen3vl-oga-inference.py" --local-dir "."
hf download onnx-community/Qwen3-4B-VL-ONNX --include "test_images/*" --local-dir "./test_images"
```

## 3. Download a model from Hugging Face

Use either `hf download` (recommended) or `huggingface-cli download`.

#### Qwen3-VL-2B-Instruct

```powershell
hf download Qwen/Qwen3-VL-2B-Instruct --local-dir "./pytorch_2b"
```

#### Qwen3-VL-4B-Instruct

```powershell
hf download Qwen/Qwen3-VL-4B-Instruct --local-dir "./pytorch_4b"
```

#### Qwen3-VL-8B-Instruct

```powershell
hf download Qwen/Qwen3-VL-8B-Instruct --local-dir "./pytorch_8b"
```
## 4. Export ONNX package

#### FP32 vision + FP32 text

```powershell
& "python.exe" `
  "builder.py" `
  --input "./pytorch_4b" `
  --reference "./pytorch_reference" `
  --output "./qwen3-vl-4b-instruct-onnx-vision-fp32-text-fp32-cpu" `
  --precision fp32
```


#### FP32 vision + INT4 text

```powershell
# 2B
& "python.exe" `
  "builder.py" `
  --input "./pytorch_2b" `
  --reference "./pytorch_reference" `
  --output "./qwen3-vl-2b-instruct-onnx-vision-fp32-text-int4-cpu" `
  --precision int4

# 4B
& "python.exe" `
  "builder.py" `
  --input "./pytorch_4b" `
  --reference "./pytorch_reference" `
  --output "./qwen3-vl-4b-instruct-onnx-vision-fp32-text-int4-cpu" `
  --precision int4

# 8B
& "python.exe" `
  "builder.py" `
  --input "./pytorch_8b" `
  --reference "./pytorch_reference" `
  --output "./qwen3-vl-8b-instruct-onnx-vision-fp32-text-int4-cpu" `
  --precision int4
```

## 5. Sanity test: text-only

Run from the same folder:

```powershell
& "python.exe" `
  "qwen3vl-oga-inference.py" `
  -m "./qwen3-vl-8b-instruct-onnx-vision-fp32-text-int4-cpu" `
  -e follow_config `
  --non-interactive `
  -pr "Say hello in one short sentence."
```

Expected behavior: model loads and returns a short greeting (for example, `Hello!`).

## 6. Sanity test: image + text

```powershell
& "python.exe" `
  "qwen3vl-oga-inference.py" `
  -m "./qwen3-vl-8b-instruct-onnx-vision-fp32-text-int4-cpu" `
  -e follow_config `
  --non-interactive `
  --image_paths "./test_images/img_50.jpg" `
  -pr "Describe this image in one sentence."
```

Expected behavior: model returns a one-sentence description for the image.

## 7. Notes

- Current script usage is validated for single-image inference per call.
- If you pass multiple images in one call, you may hit:
  - `RuntimeError: Expected pixel_values in CHW format [C, H, W], got rank 4`
- Practical workaround: run one image per invocation.

## 8. Citation

If you use Qwen3-VL models, please cite the Qwen technical reports and model cards from the Qwen team.