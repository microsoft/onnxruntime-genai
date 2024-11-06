// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <cassert>
#include <chrono>
#include <iomanip>
#include <iostream>
#include <string>
#include "ort_genai.h"

#include <thread>
#include <vector>

// C++ API Example for Terminating Session

void Generate_Output_CXX(OgaGenerator* generator, std::unique_ptr<OgaTokenizerStream> tokenizer_stream) {
  try {
    while (!generator->IsDone()) {
      generator->ComputeLogits();

      generator->GenerateNextToken();

      std::unique_ptr<OgaTensor> output_logits = generator->GetOutput("logits");

      auto logits = reinterpret_cast<float*>(output_logits->Data());

      const auto num_tokens = generator->GetSequenceCount(0);
      const auto new_token = generator->GetSequenceData(0)[num_tokens - 1];
      std::cout << tokenizer_stream->Decode(new_token) << std::flush;
    }
  }
  catch (const std::exception& e) {
    std::cout << "Session Terminated: " << e.what() << std::endl;
  }
}

void Generator_SetTerminate_Call(OgaGenerator* generator) {
  std::cout << "Calling SetTerminate for Phi3 example after 1 second" << std::endl;
  std::this_thread::sleep_for(std::chrono::seconds(1));
  generator->SetRuntimeOption("terminate_session", "1");
}


void CXX_API(const char* model_path) {
  std::cout << "Creating model..." << std::endl;
  auto model = OgaModel::Create(model_path);
  std::cout << "Creating tokenizer..." << std::endl;
  auto tokenizer = OgaTokenizer::Create(*model);
  

  bool runTerminate = true;

  try {
    while (true) {
      auto tokenizer_stream = OgaTokenizerStream::Create(*tokenizer);
      std::string text;
      std::cout << "Prompt: (Use quit() to exit)" << std::endl;
      std::getline(std::cin, text);

      if (text == "quit()") {
        break;  // Exit the loop
      }

      const std::string prompt = "<|user|>\n" + text + "<|end|>\n<|assistant|>";

      auto sequences = OgaSequences::Create();
      tokenizer->Encode(prompt.c_str(), *sequences);

      std::cout << "Generating response..." << std::endl;
      auto params = OgaGeneratorParams::Create(*model);
      params->SetSearchOption("max_length", 1024);
      params->SetInputSequences(*sequences);

      auto generator = OgaGenerator::Create(*model, *params);

      std::vector<std::thread> threads;

      threads.push_back(std::thread(Generate_Output_CXX, generator.get(), std::move(tokenizer_stream)));

      if (runTerminate){
        threads.push_back(std::thread(Generator_SetTerminate_Call, generator.get()));
        runTerminate = false;
      }

      for (auto& th : threads) {
        th.join();  // Wait for each thread to finish
      }

      for (int i = 0; i < 3; ++i)
        std::cout << std::endl;
    }
  }
  catch (const std::exception& e) {
    std::cout << "Error: " << e.what() << std::endl;
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
  std::cout << "Hello, Phi-3!" << std::endl;
  std::cout << "-------------" << std::endl;

#ifdef USE_CXX
  std::cout << "C++ API" << std::endl;
  CXX_API(argv[1]);
#endif

  return 0;
}