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

bool FileExists(const char* path);

std::string Trim(const std::string& str);

// Returns true if model_path & ep were able to be set from user cmd-line args.
// Returns false if insufficient cmd-line arguments were passed.
// Note: ep will be set to "follow_config" if user only gives model_path
bool parse_args(int /*argc*/, char** argv, std::string& model_path, std::string& ep);

// Append provider / options to config.
// This is a no-op if provider=="follow_config"
void append_provider(OgaConfig& config, const std::string& provider);