// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include <chrono>
#include <condition_variable>
#include <mutex>
#include <iostream>
#include <fstream>
#include <iomanip>
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

  void RecordStartTimestamp();
  void RecordFirstTokenTimestamp();
  void RecordEndTimestamp();
  void Log(const int prompt_tokens_length, const int new_tokens_length);

 private:
  TimePoint start_timestamp_;
  TimePoint first_token_timestamp_;
  TimePoint end_timestamp_;
};


class TerminateSession {
 public:
  std::condition_variable cv;
  std::mutex mtx;
  bool stopFlag = false;

  void signalHandler(int signum);
  void Generator_SetTerminate_Call(OgaGenerator* generator);
  void Generator_SetTerminate_Call_C(OgaGenerator* generator);
};

bool FileExists(const char* path);

std::string trim(const std::string& str);

void print_usage(int /*argc*/, char** argv);