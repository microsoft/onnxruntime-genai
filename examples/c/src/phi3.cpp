// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <cassert>
#include <chrono>
#include <iomanip>
#include <iostream>
#include <string>
#include <cstring>
#include "ort_genai.h"
#include <thread>
#include <csignal>
#include <atomic>
#include <condition_variable>
#include <mutex>
#include <functional>

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

class TerminateSession {
 public:
  std::condition_variable cv;
  std::mutex mtx;
  bool stopFlag = false;

  void signalHandler(int signum) {
    std::cout << "Interrupt signal received. Terminating current session...\n";
    std::unique_lock<std::mutex> lock(mtx);
    stopFlag = true;
    cv.notify_one();
  }

  void Generator_SetTerminate_Call(OgaGenerator* generator) {
    std::unique_lock<std::mutex> lock(mtx);
    while (!generator->IsDone()) {
      if (stopFlag) {
        generator->SetRuntimeOption("terminate_session", "1");
        stopFlag = false;
        break;
      }
      // Wait for stopflag to become true or it will timeout after 1000 ms
      auto timeout = std::chrono::milliseconds(1000);
      cv.wait_for(lock, timeout, [this] { return stopFlag; });
    }
  }

  void Generator_SetTerminate_Call_C(OgaGenerator* generator) {
    std::unique_lock<std::mutex> lock(mtx);
    while (!OgaGenerator_IsDone(generator)) {
      if (stopFlag) {
        OgaGenerator_SetRuntimeOption(generator, "terminate_session", "1");
        stopFlag = false;
        break;
      }
      // Wait for stopflag to become true or it will timeout after 1000 ms
      auto timeout = std::chrono::milliseconds(1000);
      cv.wait_for(lock, timeout, [this] { return stopFlag; });
    }
  }
};

static TerminateSession catch_terminate;

void signalHandlerWrapper(int signum) {
  catch_terminate.signalHandler(signum);
}

void CXX_API(const char* model_path, const char* execution_provider) {
  std::cout << "Creating config..." << std::endl;
  auto config = OgaConfig::Create(model_path);

  config->ClearProviders();
  std::string provider(execution_provider);
  if (provider.compare("cpu") != 0) {
    config->AppendProvider(execution_provider);
    if (provider.compare("cuda") == 0) {
      config->SetProviderOption(execution_provider, "enable_cuda_graph", "0");
    }
  }

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

    const std::string prompt = "<|user|>\n" + text + "<|end|>\n<|assistant|>";

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
    }
    catch (const std::exception& e) {
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

static void print_usage(int /*argc*/, char** argv) {
  std::cerr << "usage: " << argv[0] << std::endl;
  std::cerr << "model_path = " << argv[1] << std::endl;
  std::cerr << "execution_provider = " << argv[2] << std::endl;
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

#ifdef USE_CXX
  std::cout << "C++ API" << std::endl;
  CXX_API(argv[1], argv[2]);
#else
  std::cout << "C API" << std::endl;
  C_API(argv[1], argv[2]);
#endif

  return 0;
}