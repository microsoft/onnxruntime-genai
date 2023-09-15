# Generators
More easily run generator models using Onnx Runtime

Currently in Onnxruntime to use greedy search/beam search requires embedding the search inside the model itself through creation of a custom model per search type. This prevents users from being able to easily and dynamically modify the search as it all happens within a single Run() call of the model. This library moves the search outside of the model and lets the user be involved in every iteration of the search.

* Tools to load models like Gpt/T5/Whisper
* Search techniques like greedy/beam search to generate better token sequences
* Built in scoring tools like repetition penalties
* Easy custom scoring

## GPT Usage Example

Link to below example in code here: https://github.com/RyanUnderhill/generators/blob/0b96f546474bb5cbf7a802f9b017b26cee86ec14/src/tests/tests.cpp#L122

    std::vector<int64_t> input_ids_shape{2, 4};
    std::vector<int32_t> input_ids{0, 0, 0, 52, 0, 0, 195, 731};

    auto input_ids_tensor = OrtValue::CreateTensor(
            *info, input_ids.data(), input_ids.size(), input_ids_shape.data(), input_ids_shape.size());
     
    Generators::Gpt gpt(*ort_env,
            ORT_TSTR("models/gpt2_fp32.onnx"), // Init
            ORT_TSTR("models/gpt2_fp32.onnx"), // Decode
            );

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
      // Custom scoring goes here
 
      // Sampling goes here
 
      search.NextTokensFromLogits();
      search.CheckForEOS();
      search.AppendNextTokensToSequences();
    }

    // Access resulting sequences of tokens
    for(unsigned i=0;i<params.batch_size;i++) {
      auto result=search.GetSequence(0);
    }

# Complete

* CPU & CUDA GPT2 model loading
* CPU & CUDA Beam & Greedy Searches
* CPU & CUDA Scoring examples

# Future

* Make model code stateless, move state into search? This would allow for multiple searches with one model loaded
* Support more models built-in, T5/Whisper
* Sampling?
* Tokenizer?

# Building

TODO

## Prerequites

* Onnxruntime
* cmake
