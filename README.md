# ONNX Runtime Generative AI

Run generative AI models with ONNX Runtime.

This library provides the generative AI loop for ONNX models run with ONNX Runtime, including logits processing, search and sampling, and KV cache management.

Users can call a high level `generate()` method, or provide their own customizations of the loop.

* Search techniques like greedy/beam search to generate token sequences
* Built in scoring tools like repetition penalties
* Easy custom scoring

## GPT C++ Usage Example

    std::vector<int64_t> input_ids_shape{2, 4};
    std::vector<int32_t> input_ids{0, 0, 0, 52, 0, 0, 195, 731};

    auto input_ids_tensor = OrtValue::CreateTensor(
            *info, input_ids.data(), input_ids.size(), input_ids_shape.data(), input_ids_shape.size());
     
    Generators::Gpt_Model model(*ort_env, ORT_TSTR("models/gpt2_fp32.onnx"));

    Generators::SearchParams params;
    params.batch_size = static_cast<int>(input_ids_shape[0]);
    params.sequence_length = static_cast<int>(input_ids_shape[1]);
    params.input_ids = input_ids;
    params.max_length = max_length;
    params.num_beams = 4;
    params.vocab_size = model.GetVocabSize();
 
    Generators::BeamSearch search{params};
    Generators::Gpt_State state{search.sequence_lengths_, params};
 
    while (!search.IsDone()) {
      search.SetLogits(state.Run(search.GetNextTokens(), search.GetNextIndices(), search.GetSequenceLength());
 
      // Scoring
      Processors::MinLength(search, 5);
      Processors::RepetitionPenalty(search, 1.1f);
 
      search.SelectTop();
    }

    // Access resulting sequences of tokens
    for(unsigned i=0;i<params.batch_size;i++) {
      auto result=search.GetSequence(0);
    }

## GPT Python End to End Example

    import onnxruntime_genai as og
    import numpy as np
    from transformers import GPT2Tokenizer

    text = "The best hotel in bay area"

    # Generate input tokens from the text prompt
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    input_tokens = tokenizer.encode(text, return_tensors='np')

    model=og.Gpt_Model("../../python/onnx_models/gpt2.onnx", og.DeviceType.CUDA)

    params=og.SearchParams()
    params.max_length = 64
    params.batch_size = input_tokens.shape[0]
    params.sequence_length = input_tokens.shape[1]
    params.input_ids = input_tokens
    params.vocab_size = model.GetVocabSize()
    params.eos_token_id = tokenizer.eos_token_id
    params.pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else params.eos_token_id

    search=og.GreedySearch(params, model.DeviceType)
    state=og.Gpt_State(model, search.GetSequenceLengths(), params)

    print("Inputs:")
    print(input_tokens)
    print("Input prompt:", text)

    print("Running greedy search loop...")
    while not search.IsDone():
      search.SetLogits(state.Run(search.GetNextTokens(), search.GetSequenceLength())
      search.SelectTop();

    print("Output:")
    output_tokens=search.GetSequence(0).GetArray()
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
