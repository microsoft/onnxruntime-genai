# Run the Phi-3 models with ONNX Runtime GenAI

## Steps
1. [Setup](#setup)
2. [Choose your platform](#choose-your-platform)
3. [Run with DirectML](#run-with-directml)
4. [Run with NVIDIA CUDA](#run-with-nvidia-cuda)
5. [Run on CPU](#run-on-cpu)

## Introduction

There are many Phi-3 models to choose from: Phi-3 mini, Phi-3 small, Phi-3 medium, and Phi-3 vision. With the Phi-3 models, there are also short (4K/8K) context versions and long (128K) context versions to choose from. The long context version can accept much longer prompts and produce longer output text, but it does consume more memory.

The Phi-3 ONNX models are hosted [here](https://huggingface.co/collections/microsoft/phi-3-6626e15e9585a200d2d761e3) in a collection on Hugging Face.

This tutorial downloads and runs the Phi-3 mini short context model. If you would like to use another model, please change the model name in the instructions below.

## Setup

1. Install the git large file system extension

   Hugging Face uses `git` for version control. To download the ONNX models you need `git lfs` to be installed, if you do not already have it.

   * Windows: `winget install -e --id GitHub.GitLFS` (If you don't have winget, download and run the `exe` from the [official source](https://docs.github.com/en/repositories/working-with-files/managing-large-files/installing-git-large-file-storage?platform=windows))
   * Linux: `apt-get install git-lfs`
   * MacOS: `brew install git-lfs`

   Then run `git lfs install`

2. Install the HuggingFace CLI

   ```bash
   pip install huggingface-hub[cli]
   ```

## Choose your platform

Are you on a Windows machine with GPU?
* I don't know &rarr; Review [this guide](https://www.microsoft.com/en-us/windows/learning-center/how-to-check-gpu) to see whether you have a GPU in your Windows machine.
* Yes &rarr; Follow the instructions for [DirectML](#run-with-directml).
* No &rarr; Do you have an NVIDIA GPU?
  * I don't know &rarr; Review [this guide](https://docs.nvidia.com/cuda/cuda-installation-guide-microsoft-windows/index.html#verify-you-have-a-cuda-capable-gpu) to see whether you have a CUDA-capable GPU.
  * Yes &rarr; Follow the instructions for [NVIDIA CUDA GPU](#run-with-nvidia-cuda).
  * No &rarr; Follow the instructions for [CPU](#run-on-cpu).
 
**Note: Only one package and model is required based on your hardware. That is, only execute the steps for one of the following sections.**

## Run with DirectML

1. Download the model

   ```bash
   huggingface-cli download microsoft/Phi-3-mini-4k-instruct-onnx --include directml/* --local-dir .
   ```

   This command downloads the model into a folder called `directml`.


2. Install ONNX Runtime GenAI

   ```bash
   pip install onnxruntime-genai-directml
   ```

   You should now see `onnxruntime-genai-directml` in your `pip list`.

3. Run the model

   Run the model with [phi3-qa.py](https://github.com/microsoft/onnxruntime-genai/blob/main/examples/python/phi3-qa.py).

   ```bash
   curl https://raw.githubusercontent.com/microsoft/onnxruntime-genai/main/examples/python/phi3-qa.py -o phi3-qa.py
   python phi3-qa.py -m directml\directml-int4-awq-block-128 -e dml
   ```

   Once the script has loaded the model, it will ask you for input in a loop, streaming the output as it is produced the model. For example:

   ```bash
   Input: Tell me a joke about GPUs

   Certainly! Here's a light-hearted joke about GPUs:


   Why did the GPU go to school? Because it wanted to improve its "processing power"!


   This joke plays on the double meaning of "processing power," referring both to the computational abilities of a GPU and the idea of a student wanting to improve their academic skills.
   ```

## Run with NVIDIA CUDA

1. Download the model

   ```bash
   huggingface-cli download microsoft/Phi-3-mini-4k-instruct-onnx --include cuda/cuda-int4-rtn-block-32/* --local-dir .
   ```

   This command downloads the model into a folder called `cuda`.

2. Install ONNX Runtime GenAI

   ```bash
   pip install onnxruntime-genai-cuda
   ```

3. Run the model

   Run the model with [phi3-qa.py](https://github.com/microsoft/onnxruntime-genai/blob/main/examples/python/phi3-qa.py).

   ```bash
   curl https://raw.githubusercontent.com/microsoft/onnxruntime-genai/main/examples/python/phi3-qa.py -o phi3-qa.py
   python phi3-qa.py -m cuda/cuda-int4-rtn-block-32 -e cuda
   ```

      Once the script has loaded the model, it will ask you for input in a loop, streaming the output as it is produced the model. For example:

   ```bash
   Input: Tell me a joke about creative writing
 
   Output:  Why don't writers ever get lost? Because they always follow the plot! 
   ```

## Run on CPU

1. Download the model

   ```bash
   huggingface-cli download microsoft/Phi-3-mini-4k-instruct-onnx --include cpu_and_mobile/cpu-int4-rtn-block-32-acc-level-4/* --local-dir .
   ```

   This command downloads the model into a folder called `cpu_and_mobile`

2. Install ONNX Runtime GenAI
   
   ```bash
   pip install onnxruntime-genai
   ```

3. Run the model

   Run the model with [phi3-qa.py](https://github.com/microsoft/onnxruntime-genai/blob/main/examples/python/phi3-qa.py).

   ```bash
   curl https://raw.githubusercontent.com/microsoft/onnxruntime-genai/main/examples/python/phi3-qa.py -o phi3-qa.py
   python phi3-qa.py -m cpu_and_mobile/cpu-int4-rtn-block-32-acc-level-4 -e cpu
   ```

   Once the script has loaded the model, it will ask you for input in a loop, streaming the output as it is produced the model. For example:

   ```bash
   Input: Tell me a joke about generative AI

   Output:  Why did the generative AI go to school?

   To improve its "creativity" algorithm!


   This joke plays on the double meaning of "creativity" in the context of AI. Generative AI is often associated with its ability to produce creative content, but in this joke, it's humorously suggested that the AI is going to school to enhance its creative skills, as if it were a human student. 
   ```
