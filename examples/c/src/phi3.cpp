// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <cassert>
#include <chrono>
#include <iomanip>
#include <iostream>
#include <string>
#include "ort_genai.h"

using Clock = std::chrono::high_resolution_clock;
using TimePoint = std::chrono::time_point<Clock>;
using Duration = std::chrono::duration<double>;

// `Timing` is a utility class for measuring performance metrics.
class Timing {
 public:
  Timing(const Timing&) = delete;
  Timing& operator=(const Timing&) = delete;

  Timing() = default;

  ~Timing() = default;

  void RecordStartTimestamp() {
    assert(start_timestamp_.time_since_epoch().count() == 0);
    start_timestamp_ = Clock::now();
  }

  void RecordFirstTokenTimestamp() {
    assert(first_token_timestamp_.time_since_epoch().count() == 0);
    first_token_timestamp_ = Clock::now();
  }

  void RecordEndTimestamp() {
    assert(end_timestamp_.time_since_epoch().count() == 0);
    end_timestamp_ = Clock::now();
  }

  void Log(const int prompt_tokens_length, const int new_tokens_length) {
    assert(start_timestamp_.time_since_epoch().count() != 0);
    assert(first_token_timestamp_.time_since_epoch().count() != 0);
    assert(end_timestamp_.time_since_epoch().count() != 0);

    Duration prompt_time = (first_token_timestamp_ - start_timestamp_);
    Duration run_time = (end_timestamp_ - first_token_timestamp_);

    const auto default_precision{std::cout.precision()};
    std::cout << std::endl;
    std::cout << "-------------" << std::endl;
    std::cout << std::fixed << std::showpoint << std::setprecision(2)
              << "Prompt length: " << prompt_tokens_length << ", New tokens: " << new_tokens_length
              << ", Time to first: " << prompt_time.count() << "s"
              << ", Prompt tokens per second: " << prompt_tokens_length / prompt_time.count() << " tps"
              << ", New tokens per second: " << new_tokens_length / run_time.count() << " tps"
              << std::setprecision(default_precision) << std::endl;
    std::cout << "-------------" << std::endl;
  }

 private:
  TimePoint start_timestamp_;
  TimePoint first_token_timestamp_;
  TimePoint end_timestamp_;
};

// C++ API Example

void CXX_API(const char* model_path) {
  std::cout << "Creating model..." << std::endl;
  auto model = OgaModel::Create(model_path);
  std::cout << "Creating tokenizer..." << std::endl;
  auto tokenizer = OgaTokenizer::Create(*model);
  auto tokenizer_stream = OgaTokenizerStream::Create(*tokenizer);

  while (true) {
    std::string text;
    std::cout << "Prompt: (Use quit() to exit)" << std::endl;
    std::getline(std::cin, text);

    if (text == "quit()") {
      break;  // Exit the loop
    }

    const std::string prompt = "<|user|>\n" + text + "<|end|>\n<|assistant|>";

    bool is_first_token = true;
    Timing timing;
    timing.RecordStartTimestamp();

    auto sequences = OgaSequences::Create();
    tokenizer->Encode(prompt.c_str(), *sequences);

    std::cout << "Generating response..." << std::endl;
    auto params = OgaGeneratorParams::Create(*model);
    params->SetSearchOption("max_length", 1024);
    params->SetInputSequences(*sequences);

    auto generator = OgaGenerator::Create(*model, *params);

    while (!generator->IsDone()) {
      generator->ComputeLogits();
      generator->GenerateNextToken();

      if (is_first_token) {
        timing.RecordFirstTokenTimestamp();
        is_first_token = false;
      }

      // Show usage of GetOutput
      std::unique_ptr<OgaTensor> output_logits = generator->GetOutput("logits");

      // Assuming output_logits.Type() is float as it's logits
      // Assuming shape is 1 dimensional with shape[0] being the size
      auto logits = reinterpret_cast<float*>(output_logits->Data());

      // Print out the logits using the following snippet, if needed
      //auto shape = output_logits->Shape();
      //for (size_t i=0; i < shape[0]; i++)
      //   std::cout << logits[i] << " ";
      //std::cout << std::endl;

      const auto num_tokens = generator->GetSequenceCount(0);
      const auto new_token = generator->GetSequenceData(0)[num_tokens - 1];
      std::cout << tokenizer_stream->Decode(new_token) << std::flush;
    }

    timing.RecordEndTimestamp();
    const int prompt_tokens_length = sequences->SequenceCount(0);
    const int new_tokens_length = generator->GetSequenceCount(0) - prompt_tokens_length;
    timing.Log(prompt_tokens_length, new_tokens_length);

    for (int i = 0; i < 3; ++i)
      std::cout << std::endl;
  }
}

// C API Example

void CheckResult(OgaResult* result) {
  if (result) {
    std::string string = OgaResultGetError(result);
    OgaDestroyResult(result);
    throw std::runtime_error(string);
  }
}

void C_API(const char* model_path) {
  OgaModel* model;
  std::cout << "Creating model..." << std::endl;
  OgaCreateModel(model_path, &model);

  OgaTokenizer* tokenizer;
  std::cout << "Creating tokenizer..." << std::endl;
  CheckResult(OgaCreateTokenizer(model, &tokenizer));

  OgaTokenizerStream* tokenizer_stream;
  CheckResult(OgaCreateTokenizerStream(tokenizer, &tokenizer_stream));

  while (true) {
    std::string text;
    std::cout << "Prompt: (Use quit() to exit)" << std::endl;
    std::getline(std::cin, text);

    if (text == "quit()") {
      break;  // Exit the loop
    }

    const std::string prompt = "<|user|>\n" + text + "<|end|>\n<|assistant|>";

    bool is_first_token = true;
    Timing timing;
    timing.RecordStartTimestamp();

    OgaSequences* sequences;
    CheckResult(OgaCreateSequences(&sequences));
    CheckResult(OgaTokenizerEncode(tokenizer, prompt.c_str(), sequences));

    std::cout << "Generating response..." << std::endl;
    OgaGeneratorParams* params;
    CheckResult(OgaCreateGeneratorParams(model, &params));
    CheckResult(OgaGeneratorParamsSetSearchNumber(params, "max_length", 1024));
    CheckResult(OgaGeneratorParamsSetInputSequences(params, sequences));

    OgaGenerator* generator;
    CheckResult(OgaCreateGenerator(model, params, &generator));

    while (!OgaGenerator_IsDone(generator)) {
      CheckResult(OgaGenerator_ComputeLogits(generator));
      CheckResult(OgaGenerator_GenerateNextToken(generator));

      if (is_first_token) {
        timing.RecordFirstTokenTimestamp();
        is_first_token = false;
      }

      const int32_t num_tokens = OgaGenerator_GetSequenceCount(generator, 0);
      int32_t new_token = OgaGenerator_GetSequenceData(generator, 0)[num_tokens - 1];
      const char* new_token_string;
      CheckResult(OgaTokenizerStreamDecode(tokenizer_stream, new_token, &new_token_string));
      std::cout << new_token_string << std::flush;
    }

    timing.RecordEndTimestamp();
    const int prompt_tokens_length = OgaSequencesGetSequenceCount(sequences, 0);
    const int new_tokens_length = OgaGenerator_GetSequenceCount(generator, 0) - prompt_tokens_length;
    timing.Log(prompt_tokens_length, new_tokens_length);

    for (int i = 0; i < 3; ++i)
      std::cout << std::endl;

    OgaDestroyGeneratorParams(params);
    OgaDestroySequences(sequences);
  }

  OgaDestroyTokenizerStream(tokenizer_stream);
  OgaDestroyTokenizer(tokenizer);
  OgaDestroyModel(model);
}

static void print_usage(int /*argc*/, char** argv) {
  std::cerr << "usage: " << argv[0] << " model_path" << std::endl;
}

int main(int argc, char** argv) {
  if (argc != 2) {
    print_usage(argc, argv);
    return -1;
  }

  // Responsible for cleaning up the library during shutdown
  OgaHandle handle;

  std::cout << "-------------" << std::endl;
  std::cout << "Hello, Phi-3!" << std::endl;
  std::cout << "-------------" << std::endl;

#ifdef USE_CXX
  std::cout << "C++ API" << std::endl;
  CXX_API(argv[1]);
#else
  std::cout << "C API" << std::endl;
  C_API(argv[1]);
#endif

  return 0;
}