# ONNX Runtime Generative AI

[![PyPI version](https://img.shields.io/pypi/v/onnxruntime-genai)](https://pypi.org/project/onnxruntime-genai/)
[![NuGet version](https://img.shields.io/nuget/v/Microsoft.ML.OnnxRuntimeGenAI.Managed)](https://www.nuget.org/packages/Microsoft.ML.OnnxRuntimeGenAI.Managed)
[![NuGet pre-release version](https://img.shields.io/nuget/vpre/Microsoft.ML.OnnxRuntimeGenAI.Managed)](https://www.nuget.org/packages/Microsoft.ML.OnnxRuntimeGenAI.Managed)

Run generative AI models with ONNX Runtime.

This library provides the generative AI loop for ONNX models, including inference with ONNX Runtime, logits processing, search and sampling, and KV cache management.

Users can call a high level `generate()` method, or run each iteration of the model in a loop.

* Support greedy/beam search and TopP, TopK sampling to generate token sequences
* Built in logits processing like repetition penalties
* Easy custom scoring

See full documentation at [https://onnxruntime.ai/docs/genai].

## Features

* Supported model architectures:
  * Phi-3
  * Phi-2
  * Gemma
  * LLaMA
  * Mistral
* Supported targets:   
  * GPU (DirectML)
  * GPU (CUDA)
  * CPU
* Supported sampling features
  * Beam search
  * Greedy search
  * Top P/Top K
* APIs
  * Python
  * C#
  * C/C++  

## Coming very soon

* Support for the encoder decoder model architectures, such as whisper, T5 and BART.

## Coming soon

* Support for mobile devices (Android and iOS) with Java and Objective-C bindings

## Roadmap

* Stable diffusion pipeline
* Automatic model download and cache
* More model architectures

## Installation

### DirectML

```bash
pip install [--pre] numpy onnxruntime-genai-directml
```

### CPU

```bash
pip install [--pre] numpy onnxruntime-genai
```

### CUDA

```bash
pip install numpy onnxruntime-genai-cuda --pre --index-url=https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/onnxruntime-genai/pypi/simple/
```

## Sample code for phi-2 in Python

[Install](https://onnxruntime.ai/docs/genai/howto/install) the onnxruntime-genai Python package.

1. Build the model

```shell
python -m onnxruntime_genai.models.builder -m microsoft/phi-2 -e cpu -p int4 -o ./models/phi2
# You can append --extra_options enable_cuda_graph=1 to build an onnx model that supports using cuda graph in ORT.
```

2. Run inference

```python
import os
import onnxruntime_genai as og

model_path = os.path.abspath("./models/phi2")

model = og.Model(model_path)

tokenizer = og.Tokenizer(model)

prompt = '''def print_prime(n):
    """
    Print all primes between 1 and n
    """'''

tokens = tokenizer.encode(prompt)

params = og.GeneratorParams(model)
params.set_search_options({"max_length":200})
# Add the following line to enable cuda graph by passing the maximum batch size.
# params.try_use_cuda_graph_with_max_batch_size(16)
params.input_ids = tokens

output_tokens = model.generate(params)

text = tokenizer.decode(output_tokens)

print("Output:")
print(text)
```

## Model download and export

ONNX models are run from a local folder, via a string supplied to the `Model()` method. 

You can bring your own ONNX model or use the model builder utility, included in this package. 

Install model builder dependencies.

```bash
pip install numpy install transformers torch onnx onnxruntime
```

Export int4 CPU version 
```bash
huggingface-cli login --token <your HuggingFace token>
python -m onnxruntime_genai.models.builder -m microsoft/phi-2 -p int4 -e cpu -o <model folder>
```


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
