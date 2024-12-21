# Build your Phi-3 vision ONNX models for ONNX Runtime GenAI

## Steps
0. [Pre-requisites](#pre-requisites)
1. [Prepare Local Workspace](#prepare-local-workspace)
2. [Build ONNX Components](#build-onnx-components)
3. [Build ORT GenAI Configs](#build-genai_configjson-and-processor_configjson)
4. [Run Phi-3 vision ONNX models](#run-phi-3-vision-onnx-models)

## 0. Pre-requisites

Please ensure you have the following Python packages installed to create the ONNX models.

- `huggingface_hub[cli]`
- `numpy`
- `onnx`
- `onnxruntime-genai`
    - For CPU:
    ```bash
    pip install onnxruntime-genai
    ```
    - For CUDA:
    ```bash
    pip install onnxruntime-genai-cuda
    ```
    - For DirectML: 
    ```bash
    pip install onnxruntime-genai-directml
    ```
- `pillow`
- `requests`
- `torch`
    - Please install torch by following the [instructions](https://pytorch.org/get-started/locally/). For getting ONNX models that can run on CUDA or DirectML, please install torch with CUDA and ensure the CUDA version you choose in the instructions is the one you have installed.
- `torchvision`
- `transformers`

## 1. Prepare Local Workspace

Phi-3 vision is a multimodal model consisting of several models internally. In order to run Phi-3 vision with ONNX Runtime GenAI, each internal model needs to be created as a separate ONNX model. To get these ONNX models, some of the original PyTorch modeling files have to be modified.

### Download the original PyTorch modeling files

First, let's download the original PyTorch modeling files.

```bash
# Download PyTorch model and files
$ mkdir -p phi3-vision-128k-instruct/pytorch
$ cd phi3-vision-128k-instruct/pytorch
$ huggingface-cli download microsoft/Phi-3-vision-128k-instruct --local-dir .
```

### Download the modified PyTorch modeling files

Now, let's download the modified PyTorch modeling files that have been uploaded to the Phi-3 vision ONNX repository on Hugging Face.

```bash
# Download modified files
$ cd ..
$ huggingface-cli download microsoft/Phi-3-vision-128k-instruct-onnx --include onnx/* --local-dir .
```

### Replace original PyTorch repo files with modified files

```bash
# In our `config.json`, we replaced `flash_attention_2` with `eager` in `_attn_implementation`
$ rm pytorch/config.json
$ mv onnx/config.json pytorch/

# In our `modeling_phi3_v.py`, we modified some classes for exporting to ONNX
$ rm pytorch/modeling_phi3_v.py
$ mv onnx/modeling_phi3_v.py pytorch/

# In our `image_embedding_phi3_v_for_onnx.py`, we created a copy of `image_embedding_phi3_v.py` and modified it for exporting to ONNX
$ mv onnx/image_embedding_phi3_v_for_onnx.py pytorch/

# Move the builder script to the root directory
$ mv onnx/builder.py .

# Delete empty `onnx` directory
$ rm -rf onnx/
```

If you have your own fine-tuned version of Phi-3 vision, you can now replace the `*.safetensors` files in the `pytorch` folder with your `*.safetensors` files.

## 2. Build ONNX Components

Here are some examples of how you can build the components as INT4 ONNX models.

```bash
# Build INT4 components with FP32 inputs/outputs for CPU
$ python3 builder.py --input ./pytorch --output ./cpu --precision fp32 --execution_provider cpu
```

```bash
# Build INT4 components with FP16 inputs/outputs for CUDA
$ python3 builder.py --input ./pytorch --output ./cuda --precision fp16 --execution_provider cuda
```

```bash
# Build INT4 components with FP16 inputs/outputs for DirectML
$ python3 builder.py --input ./pytorch --output ./dml --precision fp16 --execution_provider dml
```

## 3. Build `genai_config.json` and `processor_config.json`

Currently, both JSON files needed to run with ONNX Runtime GenAI are created by hand. Because the fields have been hand-crafted, it is recommended that you copy the already-uploaded JSON files and modify the fields as needed for your fine-tuned Phi-3 vision model. [Here](https://huggingface.co/microsoft/Phi-3-vision-128k-instruct-onnx/blob/main/cpu_and_mobile/cpu-int4-rtn-block-32-acc-level-4/genai_config.json) is an example for `genai_config.json` and [here](https://huggingface.co/microsoft/Phi-3-vision-128k-instruct-onnx/blob/main/cpu_and_mobile/cpu-int4-rtn-block-32-acc-level-4/processor_config.json) is an example for `processor_config.json`.

## 4. Run Phi-3 vision ONNX models

[Here](https://github.com/microsoft/onnxruntime-genai/blob/main/examples/python/phi3v.py) is an example of how you can run your Phi-3 vision model with ONNX Runtime GenAI.

### CUDA
```bash
$ python .\phi3v.py -m .\phi3-vision-128k-instruct\cuda -e cuda
```

### DirectML

```bash
$ python .\phi3v.py -m .\phi3-vision-128k-instruct\dml -e dml
```
