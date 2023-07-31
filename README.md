# Generators
Easily run generator models and apply custom scoring methods (beam/greedy/etc)

## GPT Example

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

# In Progress

* Support more models built-in, T5/Whisper
* Beam search
* Remove dead code after design change
* CUDA support

# Building

TODO

## Prerequites

* Onnxruntime
* cmake
