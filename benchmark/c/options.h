// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <string>

namespace benchmark {

struct Options {
  std::string model_path;
  size_t num_prompt_tokens{16};
  size_t num_tokens_to_generate{128};
  size_t batch_size{1};
  size_t num_iterations{5};
  size_t num_warmup_iterations{1};
  bool verbose{};
};

Options ParseOptionsFromCommandLine(int argc, const char* const* argv);

}  // namespace benchmark
