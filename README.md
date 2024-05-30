# ONNX Runtime generate() API

![Latest version](https://img.shields.io/nuget/vpre/Microsoft.ML.OnnxRuntimeGenAI.Managed?label=latest)

Run Llama, Phi, Gemma, Mistral with ONNX Runtime.

This API gives you an easy, flexible and performant way of running LLMs on device. 

It implements the generative AI loop for ONNX models, including pre and post processing, inference with ONNX Runtime, logits processing, search and sampling, and KV cache management.

You can call a high level `generate()` method to generate all of the output at once, or stream the output one token at a time.

See documentation at https://onnxruntime.ai/docs/genai.

|Support matrix|Supported now|Under development|On the roadmap|
|-|-|-|-|
|Model architectures|  Gemma <br/> Llama * <br/> Mistral + <br/>Phi <br/>|Whisper|Stable diffusion|
|API| Python <br/>C# <br/>C/C++ |Java|Objective-C|||
|Platform| Linux <br/> Windows  | Android | Mac <br/> iOS |||
|Architecture|x86 <br/> x64 <br/> | Arm64 |||
|Hardware Acceleration|CUDA<br/>DirectML<br/>|QNN <br/> ROCm |OpenVINO

\* The Llama model architecture supports similar model families such as CodeLlama, Vicuna, Yi, and more.

\+ The Mistral model architecture supports similar model families such as Zephyr.


## Installation

See https://onnxruntime.ai/docs/genai/howto/install

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

   chat_template = '<|user|>\n{input} <|end|>\n<|assistant|>'

   text = input("Input: ")
   if not text:
      print("Error, input cannot be empty")
      exit

   prompt = f'{chat_template.format(input=text)}'

   input_tokens = tokenizer.encode(prompt)

   params = og.GeneratorParams(model)
   params.set_search_options(**search_options)
   params.input_ids = input_tokens
   generator = og.Generator(model, params)
  
   print("Output: ", end='', flush=True)

   try:
      while not generator.is_done():
        generator.compute_logits()
        generator.generate_next_token()

        new_token = generator.get_next_tokens()[0]
        print(tokenizer_stream.decode(new_token), end='', flush=True)
   except KeyboardInterrupt:
       print("  --control+c pressed, aborting generation--")

   print()
   del generator
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
