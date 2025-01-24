# ONNX Runtime GenAI Model Builder

This folder contains the model builder for quickly creating optimized and quantized ONNX models within a few minutes that run with ONNX Runtime GenAI.

# Contents

- [Current Support](#current-support)
- [Usage](#usage)
  - [Full Usage](#full-usage)
  - [Original PyTorch Model from Hugging Face](#original-pytorch-model-from-hugging-face)
  - [Original PyTorch Model from Disk](#original-pytorch-model-from-disk)
  - [Customized or Finetuned PyTorch Model](#customized-or-finetuned-pytorch-model)
  - [Quantized PyTorch Model](#quantized-pytorch-model)
  - [GGUF Model](#gguf-model)
  - [Extra Options](#extra-options)
    - [Config Only](#config-only)
    - [Hugging Face Authentication](#hugging-face-authentication)
    - [Exclude Embedding Layer](#exclude-embedding-layer)
    - [Exclude Language Modeling Head](#exclude-language-modeling-head)
    - [Include Last Hidden States Output](#include-last-hidden-states-output)
    - [Enable CUDA Graph](#enable-cuda-graph)
    - [Use 8 Bits Quantization in QMoE](#use-8-bits-quantization-in-qmoe)
    - [Use QDQ Pattern for Quantization](#use-qdq-pattern-for-quantization)
    - [LoRA Models](#lora-models)
  - [Unit Testing Models](#unit-testing-models)
    - [Option 1: Use the model builder directly](#option-1-use-the-model-builder-directly)
    - [Option 2: Edit the config.json file](#option-2-edit-the-configjson-file-on-disk-and-then-run-the-model-builder)
- [Design](#design)

## Current Support

The tool currently supports the following model architectures.

- ChatGLM
- Gemma
- Granite
- LLaMA
- Mistral
- Nemotron
- Phi
- Qwen
- AMD OLMo

It is intended for supporting the latest, popular state-of-the-art models.

## Usage

### Full Usage

For all available options, please use the `-h/--help` flag.

```
# From wheel:
python3 -m onnxruntime_genai.models.builder --help

# From source:
python3 builder.py --help
```

### Original PyTorch Model from Hugging Face

This scenario is where your PyTorch model is not downloaded locally (either in the default Hugging Face cache directory or in a local folder on disk).

```
# From wheel:
python3 -m onnxruntime_genai.models.builder -m model_name -o path_to_output_folder -p precision -e execution_provider -c cache_dir_to_save_hf_files

# From source:
python3 builder.py -m model_name -o path_to_output_folder -p precision -e execution_provider -c cache_dir_to_save_hf_files
```

### Original PyTorch Model from Disk

This scenario is where your PyTorch model is already downloaded locally (either in the default Hugging Face cache directory or in a local folder on disk).

```
# From wheel:
python3 -m onnxruntime_genai.models.builder -m model_name -o path_to_output_folder -p precision -e execution_provider -c cache_dir_where_hf_files_are_saved

# From source:
python3 builder.py -m model_name -o path_to_output_folder -p precision -e execution_provider -c cache_dir_where_hf_files_are_saved
```

### Customized or Finetuned PyTorch Model

This scenario is where your PyTorch model has been customized or finetuned for one of the currently supported model architectures and your model can be loaded in Hugging Face.

```
# From wheel:
python3 -m onnxruntime_genai.models.builder -i path_to_local_folder_on_disk -o path_to_output_folder -p precision -e execution_provider -c cache_dir_to_store_temp_files

# From source:
python3 builder.py -i path_to_local_folder_on_disk -o path_to_output_folder -p precision -e execution_provider -c cache_dir_to_store_temp_files
```

### Quantized PyTorch Model

This scenario is where your PyTorch model is one of the currently supported model architectures, has already been quantized to INT4 precision, and your model can be loaded in the Hugging Face style via [AutoGPTQ](https://github.com/AutoGPTQ/AutoGPTQ) or [AutoAWQ](https://github.com/casper-hansen/AutoAWQ).

```
# From wheel:
python3 -m onnxruntime_genai.models.builder -i path_to_local_folder_on_disk -o path_to_output_folder -p int4 -e execution_provider -c cache_dir_to_store_temp_files

# From source:
python3 builder.py -i path_to_local_folder_on_disk -o path_to_output_folder -p int4 -e execution_provider -c cache_dir_to_store_temp_files
```

### GGUF Model

This scenario is where your float16/float32 GGUF model is already on disk.

```
# From wheel:
python3 -m onnxruntime_genai.models.builder -m model_name -i path_to_gguf_file -o path_to_output_folder -p precision -e execution_provider -c cache_dir_for_hf_files

# From source:
python3 builder.py -m model_name -i path_to_gguf_file -o path_to_output_folder -p precision -e execution_provider -c cache_dir_for_hf_files
```

### Extra Options

This scenario is for when you want to have control over some specific settings. The below example shows how you can pass key-value arguments to `--extra_options`.

```
# From wheel:
python3 -m onnxruntime_genai.models.builder -m model_name -o path_to_output_folder -p precision -e execution_provider -c cache_dir_for_hf_files --extra_options filename=decoder.onnx

# From source:
python3 builder.py -m model_name -o path_to_output_folder -p precision -e execution_provider -c cache_dir_for_hf_files --extra_options filename=decoder.onnx
```

To see all available options through `--extra_options`, please use the `help` commands in the `Full Usage` section above.

#### Config Only

This scenario is for when you already have your optimized and/or quantized ONNX model and you need to create the config files to run with ONNX Runtime GenAI.

```
# From wheel:
python3 -m onnxruntime_genai.models.builder -m model_name -o path_to_output_folder -p precision -e execution_provider -c cache_dir_for_hf_files --extra_options config_only=true

# From source:
python3 builder.py -m model_name -o path_to_output_folder -p precision -e execution_provider -c cache_dir_for_hf_files --extra_options config_only=true
```

Afterwards, please open the `genai_config.json` file in the output folder and modify the fields as needed for your model. You should store your ONNX model in the output folder as well.

#### Hugging Face Authentication

This scenario is for when you need to disable the Hugging Face authentication or use a different authentication token than the one stored in [huggingface-cli login](https://huggingface.co/docs/huggingface_hub/main/en/guides/cli#huggingface-cli-login).

```
# From wheel:
python3 -m onnxruntime_genai.models.builder -m model_name -o path_to_output_folder -p precision -e execution_provider -c cache_dir_for_hf_files --extra_options hf_token=false

# From source:
python3 builder.py -m model_name -o path_to_output_folder -p precision -e execution_provider -c cache_dir_for_hf_files --extra_options hf_token=false
```

#### Exclude Embedding Layer

This scenario is for when you want to exclude the embedding layer from your ONNX model.

```
# From wheel:
python3 -m onnxruntime_genai.models.builder -i path_to_local_folder_on_disk -o path_to_output_folder -p precision -e execution_provider -c cache_dir_to_store_temp_files --extra_options exclude_embeds=true

# From source:
python3 builder.py -i path_to_local_folder_on_disk -o path_to_output_folder -p precision -e execution_provider -c cache_dir_to_store_temp_files --extra_options exclude_embeds=true
```

#### Exclude Language Modeling Head

This scenario is for when you want to exclude the language modeling head from your ONNX model.

```
# From wheel:
python3 -m onnxruntime_genai.models.builder -i path_to_local_folder_on_disk -o path_to_output_folder -p precision -e execution_provider -c cache_dir_to_store_temp_files --extra_options exclude_lm_head=true

# From source:
python3 builder.py -i path_to_local_folder_on_disk -o path_to_output_folder -p precision -e execution_provider -c cache_dir_to_store_temp_files --extra_options exclude_lm_head=true
```

#### Include Last Hidden States Output

This scenario is for when you want to include the last hidden states as an output to your ONNX model.

```
# From wheel:
python3 -m onnxruntime_genai.models.builder -i path_to_local_folder_on_disk -o path_to_output_folder -p precision -e execution_provider -c cache_dir_to_store_temp_files --extra_options include_hidden_states=true

# From source:
python3 builder.py -i path_to_local_folder_on_disk -o path_to_output_folder -p precision -e execution_provider -c cache_dir_to_store_temp_files --extra_options include_hidden_states=true
```

Note that this is the same as outputting embeddings since the last hidden states are also known as the embeddings.

#### Enable CUDA Graph

This scenario is for when you want to enable CUDA graph for your ONNX model.

```
# From wheel:
python3 -m onnxruntime_genai.models.builder -i path_to_local_folder_on_disk -o path_to_output_folder -p precision -e execution_provider -c cache_dir_to_store_temp_files --extra_options enable_cuda_graph=true

# From source:
python3 builder.py -i path_to_local_folder_on_disk -o path_to_output_folder -p precision -e execution_provider -c cache_dir_to_store_temp_files --extra_options enable_cuda_graph=true
```

#### Use 8 Bits Quantization in QMoE

This scenario is for when you want to use 8-bit quantization for MoE layers. Default is using 4-bit quantization.

```
# From wheel:
python3 -m onnxruntime_genai.models.builder -i path_to_local_folder_on_disk -o path_to_output_folder -p precision -e execution_provider -c cache_dir_to_store_temp_files --extra_options use_8bits_moe=true

# From source:
python3 builder.py -i path_to_local_folder_on_disk -o path_to_output_folder -p precision -e execution_provider -c cache_dir_to_store_temp_files --extra_options use_8bits_moe=true
```

#### Use QDQ Pattern for Quantization

This scenario is for when you want to use the QDQ pattern when quantizing the model to 4 bits.

```
# From wheel:
python3 -m onnxruntime_genai.models.builder -i path_to_local_folder_on_disk -o path_to_output_folder -p precision -e execution_provider -c cache_dir_to_store_temp_files --extra_options use_qdq=true

# From source:
python3 builder.py -i path_to_local_folder_on_disk -o path_to_output_folder -p precision -e execution_provider -c cache_dir_to_store_temp_files --extra_options use_qdq=true
```

#### LoRA Models

This scenario is where you have a finetuned model with LoRA adapters and your model can be loaded in the Hugging Face style via [PEFT](https://github.com/huggingface/peft).

```
# From wheel:
python3 -m onnxruntime_genai.models.builder -i path_to_local_folder_on_disk -o path_to_output_folder -p fp16 -e execution_provider -c cache_dir_to_store_temp_files --extra_options adapter_path=path_to_adapter_files

# From source:
python3 builder.py -i path_to_local_folder_on_disk -o path_to_output_folder -p fp16 -e execution_provider -c cache_dir_to_store_temp_files --extra_options adapter_path=path_to_adapter_files
```

Base weights should be located in `path_to_local_folder_on_disk` and adapter weights should be located in `path_to_adapter_files`.

### Unit Testing Models

This scenario is where your PyTorch model is already downloaded locally (either in the default Hugging Face cache directory or in a local folder on disk). If it is not already downloaded locally, here is an example of how you can download it.

```
from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "your_model_name"
cache_dir = "cache_dir_to_save_hf_files"

model = AutoModelForCausalLM.from_pretrained(model_name, cache_dir=cache_dir)
model.save_pretrained(cache_dir)

tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
tokenizer.save_pretrained(cache_dir)
```

#### Option 1: Use the model builder directly

This option is the simplest but it will download another copy of the PyTorch model onto disk to accommodate the change in the number of hidden layers.

```
# From wheel:
python3 -m onnxruntime_genai.models.builder -m model_name -o path_to_output_folder -p precision -e execution_provider --extra_options num_hidden_layers=4

# From source:
python3 builder.py -m model_name -o path_to_output_folder -p precision -e execution_provider --extra_options num_hidden_layers=4
```

#### Option 2: Edit the config.json file on disk and then run the model builder

1. Navigate to where the PyTorch model and its associated files are saved on disk.
2. Modify `num_hidden_layers` in `config.json` to your desired target (e.g. 4 layers).
3. Run the below command for the model builder.

```
# From wheel:
python3 -m onnxruntime_genai.models.builder -m model_name -o path_to_output_folder -p precision -e execution_provider -c cache_dir_where_hf_files_are_saved

# From source:
python3 builder.py -m model_name -o path_to_output_folder -p precision -e execution_provider -c cache_dir_where_hf_files_are_saved
```

## Design

Please read the [design document](DESIGN.md) for more details and for how to contribute.
