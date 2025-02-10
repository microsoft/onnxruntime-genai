// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <string>
#include <cstring>
#include "ort_genai.h"
#include <thread>
#include <csignal>
#include <atomic>
#include <functional>
#include "../common.h"

static TerminateSession catch_terminate;

void signalHandlerWrapper(int signum) {
  catch_terminate.signalHandler(signum);
}

// C API Example

void CheckResult(OgaResult* result) {
  if (result) {
    std::string string = OgaResultGetError(result);
    OgaDestroyResult(result);
    throw std::runtime_error(string);
  }
}

bool CheckIfSessionTerminated(OgaResult* result, OgaGenerator* generator) {
  if (result) {
    if (OgaGenerator_IsSessionTerminated(generator)){
      return true;
    }
    std::string string = OgaResultGetError(result);
    OgaDestroyResult(result);
    throw std::runtime_error(string);
  }
  return false;
}

void C_API(const char* model_path, const char* execution_provider) {
  OgaConfig* config;
  std::cout << "Creating config..." << std::endl;
  CheckResult(OgaCreateConfig(model_path, &config));

  CheckResult(OgaConfigClearProviders(config));
  if (strcmp(execution_provider, "cpu") != 0) {
    CheckResult(OgaConfigAppendProvider(config, execution_provider));
    if (strcmp(execution_provider, "cuda") == 0) {
      CheckResult(OgaConfigSetProviderOption(config, execution_provider, "enable_cuda_graph", "0"));
    }
  }

  OgaModel* model;
  std::cout << "Creating model..." << std::endl;
  CheckResult(OgaCreateModelFromConfig(config, &model));

  OgaTokenizer* tokenizer;
  std::cout << "Creating tokenizer..." << std::endl;
  CheckResult(OgaCreateTokenizer(model, &tokenizer));

  OgaTokenizerStream* tokenizer_stream;
  CheckResult(OgaCreateTokenizerStream(tokenizer, &tokenizer_stream));

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

    OgaGenerator* generator;
    CheckResult(OgaCreateGenerator(model, params, &generator));
    CheckResult(OgaGenerator_AppendTokenSequences(generator, sequences));

    std::thread th(std::bind(&TerminateSession::Generator_SetTerminate_Call_C, &catch_terminate, generator));

    while (!OgaGenerator_IsDone(generator)) {
      if (CheckIfSessionTerminated(OgaGenerator_GenerateNextToken(generator), generator))
        break;

      if (is_first_token) {
        timing.RecordFirstTokenTimestamp();
        is_first_token = false;
      }

      const int32_t num_tokens = OgaGenerator_GetSequenceCount(generator, 0);
      int32_t new_token = OgaGenerator_GetSequenceData(generator, 0)[num_tokens - 1];
      const char* new_token_string;
      if (CheckIfSessionTerminated(OgaTokenizerStreamDecode(tokenizer_stream, new_token, &new_token_string), generator))
        break;
      std::cout << new_token_string << std::flush;
    }

    timing.RecordEndTimestamp();
    const int prompt_tokens_length = OgaSequencesGetSequenceCount(sequences, 0);
    const int new_tokens_length = OgaGenerator_GetSequenceCount(generator, 0) - prompt_tokens_length;
    timing.Log(prompt_tokens_length, new_tokens_length);

    if (th.joinable()) {
      th.join();  // Join the thread if it's still running
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

int main(int argc, char** argv) {
  if (argc != 3) {
    print_usage(argc, argv);
    return -1;
  }

  // Responsible for cleaning up the library during shutdown
  OgaHandle handle;

  std::cout << "-------------" << std::endl;
  std::cout << "Hello, Phi-3!" << std::endl;
  std::cout << "-------------" << std::endl;

  std::cout << "C API" << std::endl;
  C_API(argv[1], argv[2]);

  return 0;
}