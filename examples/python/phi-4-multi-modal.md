# Build your Phi-4 Multimodal ONNX models for ONNX Runtime GenAI

## Steps
0. [Pre-requisites](#pre-requisites)
1. [Prepare Local Workspace](#prepare-local-workspace)
2. [Build ONNX Components](#build-onnx-components)
3. [Build ORT GenAI Configs](#build-genai_configjson-and-processor_configjson)
4. [Run Phi-4 Multimodal ONNX models](#run-phi-4-multimodal-onnx-models)

## 0. Pre-requisites

Please ensure you have the following Python packages installed to create the ONNX models.

- `backoff`
- `huggingface_hub[cli]`
- `numpy`
    - Please ensure that your `numpy` version is less than 2.0.0 after installing all of the pre-requisite packages. If it is greater than or equal to 2.0.0, please uninstall `numpy` with `pip uninstall -y numpy` and install an older version (e.g. `pip install numpy==1.26.4`).
- `onnx`
- `onnxruntime` and `onnxruntime-genai`
    - ONNX Runtime: Please install the latest nightly version. To ensure the right version is installed, please install ONNX Runtime GenAI first. Then you can uninstall the stable version of ONNX Runtime that gets auto-installed as a dependency.
    - ONNX Runtime GenAI: Please build from source until the latest changes are published in a stable release package. The build instructions can be found [here](https://onnxruntime.ai/docs/genai/howto/build-from-source.html).
    
    - For CPU:
    ```bash
    # 1. Build ONNX Runtime GenAI from source for CPU
    # Instructions: https://onnxruntime.ai/docs/genai/howto/build-from-source.html

    # 2. Install ONNX Runtime GenAI wheel produced by build.py
    pip install build/wheel/*.whl

    # 3. Uninstall stable version of ONNX Runtime that is auto-installed by ONNX Runtime GenAI
    pip uninstall -y onnxruntime

    # 4. Install nightly version of ONNX Runtime
    pip install -i https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/ORT-Nightly/pypi/simple/ --pre onnxruntime
    ```

    - For CUDA:
    ```bash
    # 1. Build ONNX Runtime GenAI from source for CUDA
    # Instructions: https://onnxruntime.ai/docs/genai/howto/build-from-source.html

    # 2. Install ONNX Runtime GenAI wheel produced by build.py
    pip install build/wheel/*.whl

    # 3. Uninstall stable version of ONNX Runtime that is auto-installed by ONNX Runtime GenAI
    pip uninstall -y onnxruntime-gpu

    # 4. Install nightly version of ONNX Runtime
    pip install -i https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/ORT-Nightly/pypi/simple/ --pre onnxruntime-gpu
    ```

    - For DirectML: 
    ```bash
    # 1. Build ONNX Runtime GenAI from source for DirectML
    # Instructions: https://onnxruntime.ai/docs/genai/howto/build-from-source.html

    # 2. Install ONNX Runtime GenAI wheel produced by build.py
    pip install build/wheel/*.whl

    # 3. Uninstall stable version of ONNX Runtime that is auto-installed by ONNX Runtime GenAI
    pip uninstall -y onnxruntime-directml

    # 4. Install nightly version of ONNX Runtime
    pip install -i https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/ORT-Nightly/pypi/simple/ --pre onnxruntime-directml
    ```
- `onnxscript`
- `peft`
- `pillow`
- `requests`
- `scipy`
- `soundfile`
- `torch`
    - Please install the Jan 25, 2025 nightly version. You can install torch by following the [instructions](https://pytorch.org/get-started/locally/). For getting ONNX models that can run on CUDA or DirectML, please install torch with CUDA and ensure the CUDA version you choose in the instructions is the one you have installed.
    - For CPU:
    ```bash
    pip install torch==2.7.0.dev20250125
    ```
    - For CUDA:
    ```bash
    pip install torch==2.7.0.dev20250125+cu124 --index-url https://download.pytorch.org/whl/nightly/cu124
    ```
- `torchaudio`
    - Please install the Jan 25, 2025 nightly version. You can install torchaudio by following the [instructions](https://pytorch.org/get-started/locally/). For getting ONNX models that can run on CUDA or DirectML, please install torchaudio with CUDA and ensure the CUDA version you choose in the instructions is the one you have installed.
    - For CPU:
    ```bash
    pip install torchaudio==2.6.0.dev20250125+cu124
    ```
    - For CUDA:
    ```bash
    pip install torchaudio==2.6.0.dev20250125+cu124 --index-url https://download.pytorch.org/whl/nightly/cu124
    ```
- `torchvision` 
    - Please install the Jan 25, 2025 nightly version. You can install torchvision by following the [instructions](https://pytorch.org/get-started/locally/). For getting ONNX models that can run on CUDA or DirectML, please install torchvision with CUDA and ensure the CUDA version you choose in the instructions is the one you have installed.
    - For CPU:
    ```bash
    pip install torchvision==0.22.0.dev20250125+cu124
    ```
    - For CUDA:
    ```bash
    pip install torchvision==0.22.0.dev20250125+cu124 --index-url https://download.pytorch.org/whl/nightly/cu124
    ```
- `transformers`

## 1. Prepare Local Workspace

Phi-4 multimodal is a multi-modal model consisting of several models internally. In order to run Phi-4 multimodal with ONNX Runtime GenAI, each internal model needs to be created as a separate ONNX model. To get these ONNX models, some of the original PyTorch modeling files have to be modified.

### Download the original PyTorch modeling files

First, let's download the original PyTorch modeling files.

```bash
# Download PyTorch model and files
$ mkdir -p phi4-multi-modal/pytorch
$ cd phi4-multi-modal/pytorch
$ huggingface-cli download microsoft/Phi-4-multimodal-instruct --local-dir .
```

### Download the modified PyTorch modeling files

Now, let's download the modified PyTorch modeling files that have been uploaded to the Phi-4 multimodal ONNX repository on Hugging Face.

```bash
# Download modified files
$ cd ..
$ huggingface-cli download microsoft/Phi-4-multimodal-instruct-onnx --include onnx/* --local-dir .
```

### Replace original PyTorch repo files with modified files

```bash
# In our `config.json`, we replaced `flash_attention_2` with `eager` in `_attn_implementation`
$ rm pytorch/config.json
$ mv onnx/config.json pytorch/

# In our `modeling_phi4mm.py`, we modified some classes for exporting to ONNX
$ rm pytorch/modeling_phi4mm.py
$ mv onnx/modeling_phi4mm.py pytorch/

# In our `speech_conformer_encoder.py`, we modified some classes for exporting to ONNX
$ rm pytorch/speech_conformer_encoder.py
$ mv onnx/speech_conformer_encoder.py pytorch/

# In our `vision_siglip_navit.py`, we modified some classes for exporting to ONNX
$ rm pytorch/vision_siglip_navit.py
$ mv onnx/vision_siglip_navit.py pytorch/

# In our `processing_phi4mm.py`, we modified some classes for exporting to ONNX
$ rm pytorch/processing_phi4mm.py
$ mv onnx/processing_phi4mm.py pytorch/

# Move the builder script to the root directory
$ mv onnx/builder.py .

# Delete empty `onnx` directory
$ rm -rf onnx/
```

If you have your own fine-tuned version of Phi-4 multi-modal, you can now replace the `*.safetensors` files in the `pytorch` folder with your `*.safetensors` files.

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

## 3. Build `genai_config.json`, `speech_processor.json`, and `vision_processor.json`

Currently, the JSON files needed to run with ONNX Runtime GenAI are created by hand. Because the fields have been hand-crafted, it is recommended that you copy the already-uploaded JSON files and modify the fields as needed for your fine-tuned Phi-4 multimodal model.

- [Here](https://huggingface.co/microsoft/Phi-4-multimodal-instruct-onnx/blob/main/cpu_and_mobile/cpu-int4-rtn-block-32-acc-level-4/genai_config.json) is an example for `genai_config.json`
- [Here](https://huggingface.co/microsoft/Phi-4-multimodal-instruct-onnx/blob/main/cpu_and_mobile/cpu-int4-rtn-block-32-acc-level-4/speech_processor.json) is an example for `speech_processor.json`
- [Here](https://huggingface.co/microsoft/Phi-4-multimodal-instruct-onnx/blob/main/cpu_and_mobile/cpu-int4-rtn-block-32-acc-level-4/vision_processor.json) is an example for `vision_processor.json`

## 4. Run Phi-4 Multimodal ONNX models

[Here](https://github.com/microsoft/onnxruntime-genai/blob/main/examples/python/phi4-mm.py) is an example of how you can run your Phi-4 multimodal model with ONNX Runtime GenAI.

### CPU
```bash
$ python3 phi4-mm.py -m ./phi4-mm/cpu -e cpu
```

### CUDA
```bash
$ python3 phi4-mm.py -m ./phi4-mm/cuda -e cuda
```

### DirectML

```bash
$ python phi4-mm.py -m ./phi4-mm/dml -e dml
```
