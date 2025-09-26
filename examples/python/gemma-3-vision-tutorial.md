# Build your Gemma-3 vision ONNX models for ONNX Runtime GenAI

## Steps
0. [Pre-requisites](#pre-requisites)
1. [Prepare Local Workspace](#prepare-local-workspace)
2. [Build ONNX Components](#build-onnx-components)
3. [Build ORT GenAI Configs](#build-genai_configjson-and-processor_configjson)
4. [Run Gemma-3 vision ONNX models](#run-Gemma-3-vision-onnx-models)

## 0. Pre-requisites

Please ensure you have the following Python packages installed to create the ONNX models.

- `huggingface_hub[cli]`
- `numpy`
  - Please ensure that your `numpy` version is less than 2.0.0 after installing all of the pre-requisite packages. If it is greater than or equal to 2.0.0, please uninstall `numpy` with `pip uninstall -y numpy` and install an older version (e.g. `pip install numpy==1.26.4`).
- `onnx`
- `onnxruntime` and `onnxruntime-genai`
  - ONNX Runtime: Please install the latest nightly version. To ensure the right version is installed, please install ONNX Runtime GenAI first. Then you can uninstall the stable version of ONNX Runtime that gets auto-installed as a dependency.
  - ONNX Runtime GenAI: Please install the latest nightly version.

    - For CPU:
    ```bash
    # 1. Install nightly version of ONNX Runtime GenAI
    pip install -i https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/ORT-Nightly/pypi/simple/ --pre onnxruntime-genai

    # 2. Uninstall stable version of ONNX Runtime that is auto-installed by ONNX Runtime GenAI
    pip uninstall -y onnxruntime

    # 3. Install nightly version of ONNX Runtime
    pip install -i https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/ORT-Nightly/pypi/simple/ --pre onnxruntime
    ```

    - For CUDA:
    ```bash
    # 1. Install nightly version of ONNX Runtime GenAI
    pip install -i https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/ORT-Nightly/pypi/simple/ --pre onnxruntime-genai-cuda

    # 2. Uninstall stable version of ONNX Runtime that is auto-installed by ONNX Runtime GenAI
    pip uninstall -y onnxruntime-gpu

    # 3. Install nightly version of ONNX Runtime
    pip install -i https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/ORT-Nightly/pypi/simple/ --pre onnxruntime-gpu
    ```

    - For DirectML:
    ```bash
    # 1. Install nightly version of ONNX Runtime GenAI
    pip install -i https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/ORT-Nightly/pypi/simple/ --pre onnxruntime-genai-directml

    # 2. Uninstall stable version of ONNX Runtime that is auto-installed by ONNX Runtime GenAI
    pip uninstall -y onnxruntime-directml

    # 3. Install nightly version of ONNX Runtime
    pip install -i https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/ORT-Nightly/pypi/simple/ --pre onnxruntime-directml
    ```
- `onnxscript`
  - Please install the latest nightly version of onnxscript with `pip install --pre onnxscript`.
- `pillow`
- `requests`
- `torch`
  - Please install torch by following the [instructions](https://pytorch.org/get-started/locally/). For getting ONNX models that can run on CUDA or DirectML, please install torch with CUDA and ensure the CUDA version you choose in the instructions is the one you have installed.
  - Please ensure that your `torch` version is greater than or equal to 2.7.0 after installing all of the pre-requisite packages. If it is less than 2.7.0, please uninstall `torch`, `torchaudio`, and `torchvision` with `pip uninstall -y torch torchaudio torchvision` and install a newer version (e.g. `pip install torch==2.7.0`).
- `torchvision`
- `transformers`

## 1. Prepare Local Workspace

Gemma-3 vision is a multimodal model consisting of several models internally. In order to run Gemma-3 vision with ONNX Runtime GenAI, each internal model needs to be created as a separate ONNX model. To get these ONNX models, some of the original PyTorch modeling files have to be modified.

### Download the original PyTorch modeling files

First, let's download the original PyTorch modeling files.

```bash
# Download PyTorch model and files
$ mkdir -p gemma3-vision-it/pytorch
$ cd gemma3-vision-it/pytorch

# Inside the {} below, choose between one of the following official parameter sizes (`4b`, `12b`, `27b`)
$ huggingface-cli download google/gemma-3-{}-it --local-dir .
```

### Download the modified PyTorch modeling files

Now, let's download the modified PyTorch modeling files that have been uploaded to the Gemma-3 vision ONNX repository on Hugging Face.

```bash
# Download modified files
$ cd ..
$ huggingface-cli download onnxruntime/Gemma-3-ONNX --include onnx/* --local-dir .
```

### Replace original PyTorch repo files with modified files

```bash
# In our `config.json`, we added `_attn_implementation: eager`
# Inside the {} below, choose between one of the following official parameter sizes (`4b`, `12b`, `27b`)
$ rm pytorch/config.json
$ mv onnx/{}/config.json pytorch/

# We need a copy of `configuration_gemma3.py` to load any classes modified for exporting to ONNX
$ mv onnx/configuration_gemma3.py pytorch/

# In our `modeling_gemma3.py`, we modified some classes for exporting to ONNX
$ mv onnx/modeling_gemma3.py pytorch/

# Move the builder script to the root directory
$ mv onnx/builder.py .

# Delete empty `onnx` directory
$ rm -rf onnx/
```

If you have your own fine-tuned version of Gemma-3 vision, you can now replace the `*.safetensors` files in the `pytorch` folder with your `*.safetensors` files.

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
# Build INT4 components with BF16 inputs/outputs for CUDA
$ python3 builder.py --input ./pytorch --output ./cuda --precision bf16 --execution_provider cuda
```

```bash
# Build INT4 components with FP16 inputs/outputs for DirectML
$ python3 builder.py --input ./pytorch --output ./dml --precision fp16 --execution_provider dml
```

## 3. Build `genai_config.json` and `processor_config.json`

Currently, both JSON files needed to run with ONNX Runtime GenAI are created by hand. Because the fields have been hand-crafted, it is recommended that you copy the already-uploaded JSON files and modify the fields as needed for your fine-tuned Gemma-3 vision model. [Here](https://huggingface.co/onnxruntime/Gemma-3-ONNX/blob/main/gemma-3-4b-it/cpu_and_mobile/cpu-int4-rtn-block-32-acc-level-4/genai_config.json) is an example for `genai_config.json` and [here](https://huggingface.co/onnxruntime/Gemma-3-ONNX/blob/main/gemma-3-4b-it/cpu_and_mobile/cpu-int4-rtn-block-32-acc-level-4/processor_config.json) is an example for `processor_config.json`.

## 4. Run Gemma-3 vision ONNX models

[Here](https://github.com/microsoft/onnxruntime-genai/blob/main/examples/python/model-vision.py) is an example of how you can run your Gemma-3 vision model with ONNX Runtime GenAI.

### CPU
```bash
$ python model-vision.py -m ./gemma3-vision-it/cpu -e cpu
```

### CUDA
```bash
$ python model-vision.py -m ./gemma3-vision-it/cuda -e cuda
```

### DirectML

```bash
$ python model-vision.py -m ./gemma3-vision-it/dml -e dml
```
