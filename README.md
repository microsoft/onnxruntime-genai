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
     
    Generators::Gpt gpt(*ort_env, ORT_TSTR("models/gpt2_fp32.onnx"));

    Generators::SearchParams params;
    params.batch_size = static_cast<int>(input_ids_shape[0]);
    params.sequence_length = static_cast<int>(input_ids_shape[1]);
    params.input_ids = input_ids;
    params.max_length = max_length;
    params.num_beams = 4;
    params.vocab_size = gpt.GetVocabSize();
 
    Generators::BeamSearch search{params};
    gpt.CreateInputs(search.sequence_lengths_, params);
 
    while (!search.IsDone()) {
      gpt.Run(search.GetNextTokens(), search.GetNextIndices(), search.GetSequenceLength());
      search.SetLogits(gpt.GetLogits());
 
      // Scoring
      Processors::MinLength(search, 5);
      Processors::RepetitionPenalty(search, 1.1f);
 
      search.SelectTopK();
    }

    // Access resulting sequences of tokens
    for(unsigned i=0;i<params.batch_size;i++) {
      auto result=search.GetSequence(0);
    }

## GPT Python End to End Example

    import ort_generators as og
    import numpy as np
    from transformers import GPT2Tokenizer

    text = "The best hotel in bay area"

    # Generate input tokens from the text prompt
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    input_tokens = tokenizer.encode(text, return_tensors='np')

    gpt=og.Gpt("../../python/onnx_models/gpt2.onnx")

    params=og.SearchParams()
    params.max_length = 64
    params.batch_size = input_tokens.shape[0]
    params.sequence_length = input_tokens.shape[1]
    params.input_ids = input_tokens
    params.vocab_size = gpt.GetVocabSize()
    params.eos_token_id = tokenizer.eos_token_id
    params.pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else params.eos_token_id

    search=og.GreedySearch(params)
    gpt.CreateInputs(search.GetSequenceLengths(), params)

    print("Inputs:")
    print(input_tokens)
    print("Input prompt:", text)

    print("Running greedy search loop...")
    while not search.IsDone():
        gpt.Run(search.GetNextTokens(), search.GetSequenceLength())
        search.SetLogits(gpt.GetLogits())

    search.SelectTop1();

    print("Output:")
    output_tokens=search.GetSequence(0).GetArray()
    decoded_output=tokenizer.decode(output_tokens)
    print(decoded_output)

# Features

* CPU & CUDA
* Beam & Greedy Searches
* GPT2 Model code example
* C++ static library
* Python Bindings

# Future

* Make model code stateless, move state into search? This would allow for multiple searches with one model loaded
* Support more models built-in, T5/Whisper/Llama
* Tokenizer?

# Building

## Windows
## Linux

## Prerequites

* Onnxruntime
* cmake
