# ONNX Runtime GenAI

Note: between 0.9.0 and the 0.8.3 release, there is a breaking API change. Previously, the inputs for non-LLMs would be set with `params.SetInputs(inputs)`. Now, inputs for non-LLMs are set with `generator.SetInputs(inputs)`. With this change, inputs for all models and their modalities are set on the `generator` instead of the `generatorParams`. The inputs for LLMs are set with `generator.append_tokens(tokens)` and the inputs for non-LLMs are set with `generator.SetInputs(inputs)`.

[![Latest version](https://img.shields.io/nuget/vpre/Microsoft.ML.OnnxRuntimeGenAI.Managed?label=latest)](https://www.nuget.org/packages/Microsoft.ML.OnnxRuntimeGenAI.Managed/absoluteLatest)

[![Nightly Build](https://github.com/microsoft/onnxruntime-genai/actions/workflows/linux-cpu-x64-nightly-build.yml/badge.svg)](https://github.com/microsoft/onnxruntime-genai/actions/workflows/linux-cpu-x64-nightly-build.yml)

Run generative AI models with ONNX Runtime.

This API gives you an easy, flexible and performant way of running LLMs on device. 

It implements the generative AI loop for ONNX models, including pre and post processing, inference with ONNX Runtime, logits processing, search and sampling, and KV cache management.

See documentation at https://onnxruntime.ai/docs/genai.

|Support matrix|Supported now|Under development|On the roadmap|
| -------------- | ------------- | ----------------- | -------------- |
| Model architectures | AMD OLMo <br/> ChatGLM <br/> DeepSeek <br/> ERNIE 4.5 <br/> Gemma <br/> gpt-oss <br/> Granite <br/> Llama * <br/> Mistral + <br/> Nemotron <br/> Phi (language + vision) <br/> Qwen <br/> SmolLM3 | Whisper | Stable diffusion |
|API| Python <br/>C# <br/>C/C++ <br/> Java ^ |Objective-C||
|Platform| Linux <br/> Windows <br/>Mac ^ <br/>Android ^  ||iOS |||
|Architecture|x86 <br/> x64 <br/> Arm64 ~ ||||
|Hardware Acceleration|CUDA<br/>DirectML<br/>NvTensorRtRtx<br/>|QNN <br/> OpenVINO <br/> ROCm |  |
|Features|MultiLoRA <br/> Continuous decoding (session continuation)^ | Constrained decoding | Speculative decoding |

\* The Llama model architecture supports similar model families such as CodeLlama, Vicuna, Yi, and more.

\+ The Mistral model architecture supports similar model families such as Zephyr.

\^ Requires build from source

\~ Windows builds available, requires build from source for other platforms

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
      exit

   prompt = f'{chat_template.format(input=text)}'

   input_tokens = tokenizer.encode(prompt)

   params = og.GeneratorParams(model)
   params.set_search_options(**search_options)
   generator = og.Generator(model, params)
  
   print("Output: ", end='', flush=True)

   try:
      generator.append_tokens(input_tokens)
      while not generator.is_done():
        generator.generate_next_token()

        new_token = generator.get_next_tokens()[0]
        print(tokenizer_stream.decode(new_token), end='', flush=True)
   except KeyboardInterrupt:
       print("  --control+c pressed, aborting generation--")

   print()
   del generator
   ```

### Choosing the Right Examples: Release vs. Main Branch

Due to evolving nature of this project and ongoing feature additions, examples in the `main` branch may not always align with the latest stable release. This section outlines how to ensure compatibility between the examples and the corresponding version. Majority of the steps would remain same, just the package installation and the model example file would change.

### Stable version
Install the package according to the [installation instructions](https://onnxruntime.ai/docs/genai/howto/install). Let's say you installed 0.5.2 version of ONNX Runtime GenAI, so the instructions would look like this:

```bash
# Clone the repo
git clone https://github.com/microsoft/onnxruntime-genai.git && cd onnxruntime-genai
# Checkout the branch for the version you are using
git checkout v0.5.2
cd examples
```

### Nightly version (Main Branch)
Build the package from source using these [instructions](https://onnxruntime.ai/docs/genai/howto/build-from-source.html). Now just go to the folder location where all the examples are present.

```bash
# Clone the repo
git clone https://github.com/microsoft/onnxruntime-genai.git && cd onnxruntime-genai
cd examples
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

## Trademarks

This project may contain trademarks or logos for projects, products, or services. Authorized use of Microsoft 
trademarks or logos is subject to and must follow 
[Microsoft's Trademark & Brand Guidelines](https://www.microsoft.com/en-us/legal/intellectualproperty/trademarks/usage/general).
Use of Microsoft trademarks or logos in modified versions of this project must not cause confusion or imply Microsoft sponsorship.
Any use of third-party trademarks or logos are subject to those third-party's policies.
