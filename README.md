# ONNX Runtime Generative AI

Run generative AI models with ONNX Runtime.

This library provides the generative AI loop for ONNX models, including inference with ONNX Runtime, logits processing, search and sampling, and KV cache management.

Users can call a high level `generate()` method, or run each iteration of the model in a loop.

* Search techniques like greedy/beam search to generate token sequences
* Built in scoring tools like repetition penalties
* Easy custom scoring

## Sample code for phi-2 in Python

Install onnxruntime-genai.

(Temporary) Build and install from source according to the instructions below.


```python
import onnxruntime_genai as og

model=og.Model(f'models/microsoft/phi-2', device_type)

tokenizer = model.CreateTokenizer()

prompt = '''def print_prime(n):
    """
    Print all primes between 1 and n
    """'''

tokens = tokenizer.encode(prompt)

params=og.SearchParams(model)
params.max_length = 200
params.input_ids = tokens

output_tokens=model.Generate(params)

text = tokenizer.decode(output_tokens)

print("Output:")
print(text)
```


## Features

* Supported model architectures:
  * Phi-2
  * Llama
  * GPT
* Supported targets:   
  * CPU
  * GPU (CUDA)
* Supported sampling features
  * Beam search
  * Greedy search
  * Top P/Top K
* APIs
  * Python
  * C/C++  

## Coming very soon

* Support for the Mistral and Whisper model architectures
* C# API
* Support for DirectML

## Roadmap

* Automatic model download and cache
* More model architectures

## Build from source

This step requires `cmake` to be installed.

1. Clone this repo

   ```bash
   git clone https://github.com/microsoft/onnxruntime-genai
   ```

2. Install ONNX Runtime

   ```bash
   mkdir -p ort
   cd ort
   wget https://github.com/microsoft/onnxruntime/releases/download/v1.17.0/onnxruntime-linux-x64-gpu-1.17.0.tgz
   tar xvzf onnxruntime-linux-x64-gpu-1.17.0.tgz 
   mv onnxruntime-linux-x64-gpu-1.17.0/include .
   mv onnxruntime-linux-x64-gpu-1.17.0/lib .
   ```

3. Build onnxruntime-genai

   ```bash
   cd ..
   python build.py
   ```
   
4. Install Python wheel

   ```bash
   cd build/wheel
   pip install *.whl
   ```

## Model download and export

ONNX models are run from a local folder, via a string supplied to the `Model()` method. 

To source `microsoft/phi-2` optimized for your target, download and run the following script:

```bash
wget https://raw.githubusercontent.com/microsoft/onnxruntime-genai/kvaishnavi/models/src/python/models/export.py
```

Export int4 CPU version 
```bash
python export.py python models/export.py -m microsoft/phi-2 -p int4 -e cpu -o phi2-int4-cpu.onnx
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
