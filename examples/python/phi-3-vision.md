# Build your Phi-3 vision ONNX models for ONNX Runtime GenAI

## Steps
0. [Pre-requisites](#pre-requisites)
1. [Prepare Local Workspace](#prepare-local-workspace)
2. [Build Vision Component](#build-vision-component)
3. [Build Text Embedding Component](#build-text-embedding-component)
4. [Build Text Component](#build-text-component)
5. [Build ORT GenAI Config](#build-genai_configjson-and-processor_configjson)
6. [Run Phi-3 vision ONNX models](#run-phi-3-vision-onnx-models)

## 0. Pre-requisites

Please ensure you have the following `pip` packages installed to create the ONNX models.

- `huggingface_hub[cli]`
- `numpy`
- `onnx`
- `ort-nightly>=1.19.0.dev20240601002` or `ort-nightly-gpu>=1.19.0.dev20240601002`
    - [ORT nightly package](https://onnxruntime.ai/docs/install/#inference-install-table-for-all-languages) is needed until the latest changes are in the newest ORT stable package
    - For CPU: 
    ```bash
    pip install ort-nightly --index-url=https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/ORT-Nightly/pypi/simple/
    ```
    - For CUDA 11.X:
    ```bash
    pip install ort-nightly-gpu --index-url=https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/ORT-Nightly/pypi/simple/
    ```
    - For CUDA 12.X: 
    ```bash
    pip install ort-nightly-gpu --index-url=https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/ort-cuda-12-nightly/pypi/simple/
    ```
- `pillow`
- `requests`
- `torch`
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

Now, let's download the modified PyTorch modeling files that have been uploaded to the Phi-3 vision ONNX repositories on Hugging Face.

### Download the modified PyTorch modeling files
```bash
# Download modified files
$ cd ..
$ huggingface-cli download microsoft/Phi-3-vision-128k-instruct-onnx-cpu --include onnx/* --local-dir .
```

### Replace original PyTorch repo files with modified files

```bash
# In our `config.json`, we replaced `flash_attention_2` with `eager` in `_attn_implementation`
$ rm pytorch/config.json
$ mv onnx/config.json pytorch/

# In our `modeling_phi3_v.py`, we replaced `from .image_embedding_phi3_v import Phi3ImageEmbedding`
# with `from .image_embedding_phi3_v_for_onnx import Phi3ImageEmbedding`
$ rm pytorch/modeling_phi3_v.py
$ mv onnx/modeling_phi3_v.py pytorch/

# In our `image_embedding_phi3_v_for_onnx.py`, we created a copy of `image_embedding_phi3_v.py` and modified it for exporting to ONNX
$ mv onnx/image_embedding_phi3_v_for_onnx.py pytorch/

# Move the export script to the root directory
$ mv onnx/export.py .

# Delete empty `onnx` directory
$ rm -rf onnx/
```

If you have your own fine-tuned version of Phi-3 vision, you can now replace the `*.safetensors` files in the `pytorch` folder with your `*.safetensors` files.

## 2. Build Vision Component

The vision component of Phi-3 vision is similar to the OpenAI CLIP models with some extra pre-processing and post-processing steps. Currently some of the post-processing steps are not easily exportable, so the original PyTorch modeling code has been modified to omit that logic. The logic is instead embedded within the ONNX Runtime GenAI package directly while performing inference.

Here's an example of how you can build the vision component for INT4 CPU.

```bash
# Export vision component as FP32
$ python3 export.py --input ./pytorch/ --output ./cpu --precision fp32 --execution_provider cpu

# Optimize vision component
$ python3 -m onnxruntime.transformers.optimizer --input /path/to/model.onnx --output /path/to/model_opt.onnx --model_type clip --num_heads 16 --hidden_size 1024 --use_external_data_format --opt_level 0

# Quantize vision component to INT4
# See https://github.com/microsoft/onnxruntime/blob/main/docs/ContribOperators.md#com.microsoft.MatMulNBits for more information about accuracy level
$ python3 -m onnxruntime.quantization.matmul_4bits_quantizer --input_model /path/to/model_opt.onnx --output_model /path/to/model_quant.onnx --block_size 32 --accuracy_level 4
```

Here's an example of how you can build the vision component for INT4 CUDA.

```bash
# Export vision component as FP16
$ python3 export.py --input ./pytorch/ --output ./cuda --precision fp16 --execution_provider cuda

# Optimize vision component
$ python3 -m onnxruntime.transformers.optimizer --input /path/to/model.onnx --output /path/to/model_opt.onnx --model_type clip --num_heads 16 --hidden_size 1024 --use_external_data_format --opt_level 0

# Quantize vision component to INT4
# See https://github.com/microsoft/onnxruntime/blob/main/docs/ContribOperators.md#com.microsoft.MatMulNBits for more information about accuracy level
$ python3 -m onnxruntime.quantization.matmul_4bits_quantizer --input_model /path/to/model_opt.onnx --output_model /path/to/model_quant.onnx --block_size 32
```

## 3. Build Text Embedding Component

The text embedding component of Phi-3 vision is shared by both the vision component and the text component to handle the scenario where a user inputs both image and text as well as the scenario where a user only inputs text. As such, it is a 1-node ONNX graph that looks as follows and can be easily created by hand.

```
    input_ids
        |
      Gather
        |
  inputs_embeds
```

The above linked script that you used to build the vision component has already built the corresponding text embedding component for you.

## 4. Build Text Component

The text component of Phi-3 vision is identical to the Phi-3 mini model. The model builder in ONNX Runtime GenAI already has support for Phi-3 mini. Let's re-use that work to generate the text component.

```bash
# From wheel:
python3 -m onnxruntime_genai.models.builder -i path_to_local_folder_on_disk -o path_to_output_folder -p precision -e execution_provider -c cache_dir_to_store_temp_files --extra_options exclude_embeds=true filename=phi-3-v-128k-text.onnx

# From source:
python3 builder.py -i path_to_local_folder_on_disk -o path_to_output_folder -p precision -e execution_provider -c cache_dir_to_store_temp_files --extra_options exclude_embeds=true filename=phi-3-v-128k-text.onnx
```

Note that `exclude_embeds=true` is used because there is a separate text embedding component in Phi-3 vision, so the text embedding component does not need to be in the text component.

## 5. Build `genai_config.json` and `processor_config.json`

Currently, both JSON files needed to run with ONNX Runtime GenAI are created by hand. Because the fields have been hand-crafted, it is recommended that you copy the already-uploaded JSON files and modify the fields as needed for your fine-tuned Phi-3 vision model. [Here](https://huggingface.co/microsoft/Phi-3-vision-128k-instruct-onnx-cpu/blob/main/cpu-int4-rtn-block-32-acc-level-4/genai_config.json) is an example for `genai_config.json` and [here](https://huggingface.co/microsoft/Phi-3-vision-128k-instruct-onnx-cpu/blob/main/cpu-int4-rtn-block-32-acc-level-4/processor_config.json) is an example for `processor_config.json`.

## 6. Run Phi-3 vision ONNX models

[Here](https://github.com/microsoft/onnxruntime-genai/blob/main/examples/python/phi3v.py) is an example of how you can run Phi-3 vision with the ONNX Runtime generate() API.
