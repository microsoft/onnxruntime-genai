# Download Options for ONNX Runtime GenAI Models

This guide covers two easy ways to download models for use with ONNX Runtime GenAI:

Using Foundry Local

Using Hugging Face CLI

## Download via Foundry Local

1. Download and install [foundry-local](https://github.com/microsoft/Foundry-Local/releases)
2. List available models: `foundry model list`
3. Download a model you would like to run. For example: `foundry model download Phi-4-generic-cpu`
4. Find out where the model is saved on disk: `foundry cache location`
5. Identify the path to the model on disk. For example: `C:\Users\<user>\.foundry\Microsoft\Phi-4-generic-cpu\cpu-int4-rtn-block-32-acc-level-4`

> 📝 **Note:** Foundry Local CLI is not available on Linux at the moment. Please download the model from a Windows or a macOS machine and copy it over to your Linux machine if you would like to run on Linux.


## Download via Hugging Face Hub

1. Install the Hugging Face CLI
   ```
   pip install huggingface-hub[cli]
   ```
2. Login
   ```
   huggingface-cli login
   ```
3. Download a model
   ```
   huggingface-cli download <model_name> --include <subfolder_name>/* --local-dir .
   ```

   For example, to download the Phi-4 mini instruct gpu model:
   ```
   huggingface-cli download microsoft/Phi-4-mini-instruct-onnx --include gpu/* --local-dir .
   ```
5. Identify the path to the model on disk. For example: gpu/gpu-int4-rtn-block-32


## Build a Model

Alternatively, you can build your own model locally via the [Model Builder](https://github.com/microsoft/onnxruntime-genai/blob/main/src/python/py/models/README.md) or via [Olive](https://microsoft.github.io/Olive/examples.html)
