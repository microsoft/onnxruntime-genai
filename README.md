# Generators
More easily run generator models using Onnx Runtime

Currently in Onnxruntime to use greedy search/beam search requires embedding the search inside the model itself through creation of a custom model per search type. This prevents users from being able to easily and dynamically modify the search as it all happens within a single Run() call of the model. This library moves the search outside of the model and lets the user be involved in every iteration of the search.

* Tools to load models like Gpt/T5/Whisper
* Search techniques like greedy/beam search to generate better token sequences
* Built in scoring tools like repetition penalties
* Easy custom scoring

## GPT Usage Example

Link to below example in code here: https://github.com/RyanUnderhill/generators/blob/175abd8da6dd13ca0ef28a48c4397d4681c8722a/Generators/Tests.cpp#L264

    std::vector<int64_t> input_ids_shape{2, 4};
    std::vector<int32_t> input_ids{0, 0, 0, 52, 0, 0, 195, 731};

    auto input_ids_tensor = OrtValue::CreateTensor(
            *info, input_ids.data(), input_ids.size(), input_ids_shape.data(), input_ids_shape.size());
     
    Gpt gpt(*ort_env,
            ORT_TSTR("models/gpt2_fp32.onnx"), // Init
            ORT_TSTR("models/gpt2_fp32.onnx"), // Decode
            std::move(input_ids_tensor));
 
    Search search{gpt};
 
    while (!search.IsDone()) {
      search.RunModel();
 
      // Scoring
      Processors::MinLength(search, 5);
      Processors::RepetitionPenalty(search, 1.1f);
      // Custom scoring goes here
 
      // Sampling goes here
 
      search.NextTokensFromLogits();
      search.CheckForEOS();
      search.AppendNextTokensToSequences();
    }

    auto result=search.GetSequence(0);

# Complete

* GPT2 model loading
* CPU Greedy search
* CPU Scoring examples

# Future

* Beam search
* Make model code stateless, move state into search? This would allow for multiple searches with one model loaded
* Remove dead code after design change
* Support more models built-in, T5/Whisper
* CUDA support
* Sampling?
* Tokenizer?

# Building

TODO

## Prerequites

* Onnxruntime
* cmake
