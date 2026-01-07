# ONNX Runtime GenAI

## Status

[![Latest version](https://img.shields.io/nuget/vpre/Microsoft.ML.OnnxRuntimeGenAI.Managed?label=latest)](https://www.nuget.org/packages/Microsoft.ML.OnnxRuntimeGenAI.Managed/absoluteLatest)

[![Nightly Build](https://github.com/microsoft/onnxruntime-genai/actions/workflows/linux-cpu-x64-nightly-build.yml/badge.svg)](https://github.com/microsoft/onnxruntime-genai/actions/workflows/linux-cpu-x64-nightly-build.yml)

## Description

Run generative AI models with ONNX Runtime. This API gives you an easy, flexible and performant way of running LLMs on device. It implements the generative AI loop for ONNX models, including pre and post processing, inference with ONNX Runtime, logits processing, search and sampling, KV cache management, and grammar specification for tool calling.

ONNX Runtime GenAI powers Foundry Local, Windows ML, and the Visual Studio Code AI Toolkit.

See documentation at the [ONNX Runtime website](https://onnxruntime.ai/docs/genai) for more details.

| Support matrix | Supported now | Under development | On the roadmap|
| -------------- | ------------- | ----------------- | -------------- |
| Model architectures | ChatGLM</br>DeepSeek</br>Ernie</br>Fara</br>Gemma</br>GPTOSS</br>Granite</br>Llama</br>Mistral</br>Nemotron</br>OLMo</br>Phi</br>Phi3V</br>Phi4MM</br>Qwen</br>Qwen-2.5VL</br>SmolLM3</br>Whisper</br>| Stable diffusion ||
| API| Python <br/>C# <br/>C/C++ <br/> Java ^ | Objective-C ||
| O/S | Linux <br/> Windows <br/>Mac  <br/>Android   || iOS |||
| Architecture | x86 <br/> x64 <br/> arm64 ||||
| Hardware Acceleration | CPU <br/> CUDA <br/> DirectML <br/> NvTensorRtRtx (TRT-RTX) <br/> OpenVINO <br/> QNN <br/> WebGPU | | AMD GPU |
| Features | Multi-LoRA <br/> Continuous decoding <br/> Constrained decoding | | Speculative decoding |

^ Requires build from source

## Installation

See [installation instructions](https://onnxruntime.ai/docs/genai/howto/install) or [build from source](https://onnxruntime.ai/docs/genai/howto/build-from-source.html)

## Sample code for Phi-3 in Python

1. Download the model

   ```shell
   huggingface-cli download microsoft/Phi-3-mini-4k-instruct-onnx --include cpu_and_mobile/cpu-int4-rtn-block-32-acc-level-4/* --local-dir .
   ```

2. Install the API

   ```shell
   pip install numpy
   pip install --pre onnxruntime-genai
   ```

3. Run the model

   ```python
   import onnxruntime_genai as og

   model = og.Model('cpu_and_mobile/cpu-int4-rtn-block-32-acc-level-4')
   tokenizer = og.Tokenizer(model)
   tokenizer_stream = tokenizer.create_stream()
    
   # Set the max length to something sensible by default,
   # since otherwise it will be set to the entire context length
   search_options = {}
   search_options['max_length'] = 2048
   search_options['batch_size'] = 1

   chat_template = '<|user|>\n{input} <|end|>\n<|assistant|>'

   text = input("Input: ")
   if not text:
      print("Error, input cannot be empty")
      exit()

   prompt = f'{chat_template.format(input=text)}'

   input_tokens = tokenizer.encode(prompt)

   params = og.GeneratorParams(model)
   params.set_search_options(**search_options)
   generator = og.Generator(model, params)
  
   print("Output: ", end='', flush=True)

   try:
      generator.append_tokens(input_tokens)
      while True:
         generator.generate_next_token()
         if generator.is_done():
            break
         new_token = generator.get_next_tokens()[0]
         print(tokenizer_stream.decode(new_token), end='', flush=True)
   except KeyboardInterrupt:
         print("  --control+c pressed, aborting generation--")

   print()
   del generator
   ```

### Choose the correct version of the examples

Due to the evolving nature of this project and ongoing feature additions, examples in the `main` branch may not always align with the latest stable release. This section outlines how to ensure compatibility between the examples and the corresponding version.

### Stable version

Install the package according to the [installation instructions](https://onnxruntime.ai/docs/genai/howto/install). For example, install the Python package.

```bash
pip install onnxruntime-genai
```

Get the version of the package

```bash
pip list | grep onnxruntime-genai
```

Checkout the version of the examples that correspond to that release.

```bash
# Clone the repo
git clone https://github.com/microsoft/onnxruntime-genai.git && cd onnxruntime-genai
# Checkout the branch for the version you are using
git checkout v0.11.4
cd examples
```

### Nightly version (main branch)

Checkout the main branch of the repo

```bash
git clone https://github.com/microsoft/onnxruntime-genai.git && cd onnxruntime-genai
```

Build from source, using these [instructions](https://onnxruntime.ai/docs/genai/howto/build-from-source.html). For example, to build the Python wheel:

```bash
python build.py
```

Navigate to the examples folder in the main branch.

```bash
cd examples
```

## Breaking API changes

### v0.11.0

Between `v0.11.0` and `v0.10.1`, there is a breaking API usage change to improve model quality during multi-turn conversations.

Previously, the decoding loop could be written as follows.

```
while not IsDone():
    GenerateToken()
    GetLastToken()
    PrintLastToken()
```

In 0.11.0, the decoding loop should now be written as follows.

```
while True:
    GenerateToken()
    if IsDone():
        break
    GetLastToken()
    PrintLastToken()
```

## Roadmap

See the [Discussions](https://github.com/microsoft/onnxruntime-genai/discussions) to request new features and up-vote existing requests.


## Contributing

This project welcomes contributions and suggestions.  Most contributions require you to agree to a
Contributor License Agreement (CLA) declaring that you have the right to, and actually do, grant us
the rights to use your contribution. For details, visit https://cla.opensource.microsoft.com.

When you submit a pull request, a CLA bot will automatically determine whether you need to provide
a CLA and decorate the PR appropriately (e.g., status check, comment). Simply follow the instructions
provided by the bot. You will only need to do this once across all repos using our CLA.

This project has adopted the [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct/).
For more information see the [Code of Conduct FAQ](https://opensource.microsoft.com/codeofconduct/faq/) or
contact [opencode@microsoft.com](mailto:opencode@microsoft.com) with any additional questions or comments.

### Linting

This project enables [lintrunner](https://github.com/suo/lintrunner) for linting. You can install the dependencies and initialize with

```sh
pip install -r requirements-lintrunner.txt
lintrunner init
```

This will install lintrunner on your system and download all the necessary dependencies to run linters locally.

To format local changes:

```bash
lintrunner -a
```

To format all files:

```bash
lintrunner -a --all-files
```

## Trademarks

This project may contain trademarks or logos for projects, products, or services. Authorized use of Microsoft 
trademarks or logos is subject to and must follow [Microsoft's Trademark & Brand Guidelines](https://www.microsoft.com/en-us/legal/intellectualproperty/trademarks/usage/general). Use of Microsoft trademarks or logos in modified versions of this project must not cause confusion or imply Microsoft sponsorship. Any use of third-party trademarks or logos are subject to those third-party's policies.
