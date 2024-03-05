# ONNX Runtime Generative AI

Run generative AI models with ONNX Runtime.

This library provides the generative AI loop for ONNX models, including inference with ONNX Runtime, logits processing, search and sampling, and KV cache management.

Users can call a high level `generate()` method, or run each iteration of the model in a loop.

* Support greedy/beam search and TopP, TopK sampling to generate token sequences
* Built in logits processing like repetition penalties
* Easy custom scoring

## Features

* Supported model architectures:
  * Phi-2
  * Llama
  * Gemma
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

## Sample code for phi-2 in Python

Install onnxruntime-genai.

(Temporary) Build and install from source according to the instructions below.


```python
import onnxruntime_genai as og

model=og.Model(f'models/microsoft/phi-2', device_type)

tokenizer = model.create_tokenizer()

prompt = '''def print_prime(n):
    """
    Print all primes between 1 and n
    """'''

tokens = tokenizer.encode(prompt)

params=og.SearchParams(model)
params.max_length = 200
params.input_ids = tokens

output_tokens=model.generate(params)

text = tokenizer.decode(output_tokens)

print("Output:")
print(text)
```


## Build from source

This step requires `cmake` to be installed.

1. Clone this repo

   ```bash
   git clone https://github.com/microsoft/onnxruntime-genai
   cd onnxruntime-genai
   ```

2. Install ONNX Runtime

    By default, the onnxruntime-genai build expects to find the ONNX Runtime include and binaries in a folder called `ort` in the root directory of onnxruntime-genai. You can put the ONNX Runtime files in a different location and specify this location to the onnxruntime-genai build. These instructions use ORT_HOME as the location.

    * Install from release

      These instructions are for the Linux GPU build of ONNX Runtime. Replace the location with the operating system and target of choice. 

      ```bash
      cd $ORT_HOME
      wget https://github.com/microsoft/onnxruntime/releases/download/v1.17.1/onnxruntime-linux-x64-gpu-1.17.1.tgz
      tar xvzf onnxruntime-linux-x64-gpu-1.17.1.tgz 
      mv onnxruntime-linux-x64-gpu-1.17.1/include .
      mv onnxruntime-linux-x64-gpu-1.17.1/lib .
      ```

    * Or build from source

      ```
      git clone https://github.com/microsoft/onnxruntime.git
      cd onnxruntime
      ```

      Create include and lib folders in the ORT_HOME directory

      ```bash
      mkdir $ORT_HOME/include
      mkdir $ORT_HOME/lib
      ```

      Build from source and copy the include and libraries into ORT_HOME

      On Windows

      ```cmd
      build.bat --config RelWithDebInfo --build_shared_lib --skip_tests --parallel [--use_cuda]
      copy include\onnxruntime\core\session\onnxruntime_c_api.h $ORT_HOME\include
      copy build\Windows\RelWithDebInfo\RelWithDebInfo\*.dll $ORT_HOME\lib
      ```

      On Linux

      ```cmd
      ./build.sh --build_shared_lib --skip_tests --parallel [--use_cuda]
      cp include/onnxruntime/core/session/onnxruntime_c_api.h $ORT_HOME/include
      cp build/Linux/RelWithDebInfo/libonnxruntime*.so* $ORT_HOME/lib
      ```

3. Build onnxruntime-genai

   If you are building for CUDA, add the cuda_home argument.

   ```bash
   cd ..
   python build.py [--cuda_home <path_to_cuda_home>]
   ```
   
4. Install Python wheel

   ```bash
   cd build/wheel
   pip install *.whl
   ```

## Model download and export

ONNX models are run from a local folder, via a string supplied to the `Model()` method. 

To source `microsoft/phi-2` optimized for your target, download and run the following script. You will need to be logged into HuggingFace via the CLI to run the script.

Install model builder dependencies.

```bash
pip install numpy
pip install transformers
pip install torch
pip install onnx
pip install onnxruntime
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
