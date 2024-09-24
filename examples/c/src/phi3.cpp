// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <iostream>
#include <string>
#include "ort_genai.h"

// C++ API Example

void CXX_API(const char* model_path) {
  std::cout << "Creating model..." << std::endl;
  auto model = OgaModel::Create(model_path);
  std::cout << "Creating tokenizer..." << std::endl;
  auto tokenizer = OgaTokenizer::Create(*model);
  auto tokenizer_stream = OgaTokenizerStream::Create(*tokenizer);

  while (true) {
    std::string text;
    std::cout << "Prompt: " << std::endl;
    std::getline(std::cin, text);

    const std::string prompt = "<|user|>\n" + text + "<|end|>\n<|assistant|>";

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
    std::cout << "Prompt: " << std::endl;
    std::getline(std::cin, text);

    const std::string prompt = "<|user|>\n" + text + "<|end|>\n<|assistant|>";

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

      const int32_t num_tokens = OgaGenerator_GetSequenceCount(generator, 0);
      int32_t new_token = OgaGenerator_GetSequenceData(generator, 0)[num_tokens - 1];
      const char* new_token_string;
      CheckResult(OgaTokenizerStreamDecode(tokenizer_stream, new_token, &new_token_string));
      std::cout << new_token_string << std::flush;
    }

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