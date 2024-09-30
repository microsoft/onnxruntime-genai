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

    const std::string prompt = "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\nYou are a helpful AI assistant. Give a short answer to the following<|eot_id|><|start_header_id|>user<|end_header_id|>" + text + "<|eot_id|><|start_header_id|>assistant<|end_header_id|>";

    auto sequences = OgaSequences::Create();
    tokenizer->Encode(prompt.c_str(), *sequences);

    std::cout << "Generating response..." << std::endl;
    auto params = OgaGeneratorParams::Create(*model);
    params->SetSearchOption("max_length", 1024);
    params->SetSearchOptionBool("do_sample", true);
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
  std::cout << "Run Llama" << std::endl;
  std::cout << "-------------" << std::endl;

#ifdef USE_CXX
  std::cout << "C++ API" << std::endl;
  CXX_API(argv[1]);
#endif

  return 0;
}