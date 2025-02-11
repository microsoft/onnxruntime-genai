// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "common.h"
#include <cassert>

void Timing::RecordStartTimestamp() {
  assert(start_timestamp_.time_since_epoch().count() == 0);
  start_timestamp_ = Clock::now();
}

void Timing::RecordFirstTokenTimestamp() {
  assert(first_token_timestamp_.time_since_epoch().count() == 0);
  first_token_timestamp_ = Clock::now();
}

void Timing::RecordEndTimestamp() {
  assert(end_timestamp_.time_since_epoch().count() == 0);
  end_timestamp_ = Clock::now();
}

void Timing::Log(const int prompt_tokens_length, const int new_tokens_length) {
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

void TerminateSession::signalHandler(int signum) {
  std::cout << "Interrupt signal received. Terminating current session...\n";
  std::unique_lock<std::mutex> lock(mtx);
  stopFlag = true;
  cv.notify_one();
}

void TerminateSession::Generator_SetTerminate_Call(OgaGenerator* generator) {
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

void TerminateSession::Generator_SetTerminate_Call_C(OgaGenerator* generator) {
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

bool FileExists(const char* path) {
  return static_cast<bool>(std::ifstream(path));
}

std::string trim(const std::string& str) {
  const size_t first = str.find_first_not_of(' ');
  if (std::string::npos == first) {
    return str;
  }
  const size_t last = str.find_last_not_of(' ');
  return str.substr(first, (last - first + 1));
}

void print_usage(int /*argc*/, char** argv) {
  std::cerr << "usage: " << argv[0] << std::endl;
  std::cerr << "model_path = " << argv[1] << std::endl;
  std::cerr << "execution_provider = " << argv[2] << std::endl;
}