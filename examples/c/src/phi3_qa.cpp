// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <iomanip>
#include <string>
#include <cstring>
#include "ort_genai.h"
#include <thread>
#include <csignal>
#include <atomic>
#include <functional>
#include "common.h"

// C++ API Example

static TerminateSession catch_terminate;

void signalHandlerWrapper(int signum) {
  catch_terminate.signalHandler(signum);
}

void CXX_API(const char* model_path, const char* execution_provider) {
  std::cout << "Creating config..." << std::endl;
  auto config = OgaConfig::Create(model_path);

  std::string provider(execution_provider);
  append_provider(*config, provider);

  std::cout << "Creating model..." << std::endl;
  auto model = OgaModel::Create(*config);

  std::cout << "Creating tokenizer..." << std::endl;
  auto tokenizer = OgaTokenizer::Create(*model);
  auto tokenizer_stream = OgaTokenizerStream::Create(*tokenizer);

  while (true) {
    signal(SIGINT, signalHandlerWrapper);
    std::string text;
    std::cout << "Prompt: (Use quit() to exit) Or (To terminate current output generation, press Ctrl+C)" << std::endl;
    // Clear Any cin error flags because of SIGINT
    std::cin.clear();
    std::getline(std::cin, text);

    if (text == "quit()") {
      break;  // Exit the loop
    }

    const std::string prompt = tokenizer->ApplyChatTemplate("", text.c_str(), "", true);

    bool is_first_token = true;
    Timing timing;
    timing.RecordStartTimestamp();

    auto sequences = OgaSequences::Create();
    tokenizer->Encode(prompt.c_str(), *sequences);

    std::cout << "Generating response..." << std::endl;

    auto params = OgaGeneratorParams::Create(*model);
    params->SetSearchOption("max_length", 1024);
    auto generator = OgaGenerator::Create(*model, *params);
    std::thread th(std::bind(&TerminateSession::Generator_SetTerminate_Call, &catch_terminate, generator.get()));
    generator->AppendTokenSequences(*sequences);

    try {
      while (!generator->IsDone()) {
        generator->GenerateNextToken();

        if (is_first_token) {
          timing.RecordFirstTokenTimestamp();
          is_first_token = false;
        }

        const auto num_tokens = generator->GetSequenceCount(0);
        const auto new_token = generator->GetSequenceData(0)[num_tokens - 1];
        std::cout << tokenizer_stream->Decode(new_token) << std::flush;
      }
    } catch (const std::exception& e) {
      std::cout << "Session Terminated: " << e.what() << std::endl;
    }

    timing.RecordEndTimestamp();
    const int prompt_tokens_length = sequences->SequenceCount(0);
    const int new_tokens_length = generator->GetSequenceCount(0) - prompt_tokens_length;
    timing.Log(prompt_tokens_length, new_tokens_length);

    if (th.joinable()) {
      th.join();  // Join the thread if it's still running
    }

    for (int i = 0; i < 3; ++i)
      std::cout << std::endl;
  }
}

int main(int argc, char** argv) {
  std::string model_path, ep;
  if (!parse_args(argc, argv, model_path, ep)) {
    return -1;
  }

  // Responsible for cleaning up the library during shutdown
  OgaHandle handle;

  std::cout << "-------------" << std::endl;
  std::cout << "Hello, Phi-3!" << std::endl;
  std::cout << "-------------" << std::endl;

  std::cout << "C++ API" << std::endl;
  CXX_API(model_path.c_str(), ep.c_str());

  return 0;
}