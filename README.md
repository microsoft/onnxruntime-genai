# ONNX Runtime Generative AI

Run generative AI models with ONNX Runtime.

This library provides the generative AI loop for ONNX models run with ONNX Runtime, including logits processing, search and sampling, and KV cache management.

Users can call a high level `generate()` method, or provide their own customizations of the loop.

* Search techniques like greedy/beam search to generate token sequences
* Built in scoring tools like repetition penalties
* Easy custom scoring

## GPT C++ Usage Example

    std::vector<int32_t> input_ids{0, 0, 0, 52, 0, 0, 195, 731};
     
    auto model=Generators::CreateModel(*ort_env, "models/gpt2_fp32.onnx");

    Generators::GeneratorParams params{model};
    params.batch_size = 2;
    params.sequence_length = 4;
    params.input_ids = input_ids;
    params.max_length = max_length;
    params.num_beams = 4;
 
    auto generator = Generators::CreateGenerator(*model, params);
 
    while (!generator->IsDone()) {
      generator->ComputeLogits();
 
      // Scoring
      generator->Apply_MinLength(5);
      generator->Apply_RepetitionPenalty(1.1f);
 
      generator->AppendNextToken_Top();
    }

    // Access resulting sequences of tokens
    for(unsigned i=0;i<params.batch_size;i++) {
      auto result=generator.GetSequence(i);
    }

## GPT Python End to End Example

    import onnxruntime_genai as og
    import numpy as np
    from transformers import GPT2Tokenizer

    text = "The best hotel in bay area"

    # Generate input tokens from the text prompt
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    input_tokens = tokenizer.encode(text, return_tensors='np')

    model=og.Model("../../python/onnx_models", og.DeviceType.CUDA)

    params=og.SearchParams(model)
    params.max_length = 64
    params.input_ids = input_tokens

    generator=og.Generator(model, params)

    print("Inputs:")
    print(input_tokens)
    print("Input prompt:", text)

    print("Running greedy search loop...")
    while not generator.IsDone():
      generator.ComputeLogits()
      generator.AppendNextToken_Top()

    print("Output:")
    output_tokens=generator.GetSequence(0).GetArray()
    decoded_output=tokenizer.decode(output_tokens)
    print(decoded_output)

# Features

* Built in Model Support:
  * GPT2
  * Llama2
* CPU & CUDA
* Beam & Greedy Searches
* C++ static library
* Python Bindings

# Future

* Make model code stateless, move state into search? This would allow for multiple searches with one model loaded
* Support more models built-in, T5/Whisper/Llama
* Tokenizer?

# Building

## Windows

* Copy onnxruntime library into the ort/ folder
  * Can either build Onnxruntime from source in release mode, then copy the files specified in install_ort.bat
  * Or download a release from https://github.com/microsoft/onnxruntime/releases
  * Files in ort\ should be:
    * onnxruntime.dll
    * onnxruntime.lib
    * onnxruntime_providers_shared.dll (if using cuda)
    * onnxruntime_providers_cuda.dll (if using cuda)
    * onnxruntime_c_api.h
* Run the build.bat script to generate build files
* Open build\Generators.sln in visual studio

To run the python scripts, use PYTHONPATH: `set PYTHONPATH=/path/to/onnxruntime-genai/build/Release/`

## Linux

* Copy onnxruntime library into the ort/ folder
  * Can either build Onnxruntime from source in release mode, then copy the files specified in install_ort.sh
  * Or download a release from https://github.com/microsoft/onnxruntime/releases
  * Files in ort\ should be:
    * libonnxruntime.so
    * libonnxruntime.so.(version #)
    * libonnxruntime_providers_shared.so (if using cuda)
    * libonnxruntime_providers_cuda.so (if using cuda)
    * onnxruntime_c_api.h
* Run the build.sh script to build

To run the python scripts, use PYTHONPATH: `export PYTHONPATH=/path/to/onnxruntime-genai/build/`

## Prerequites

* Onnxruntime
* cmake

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
