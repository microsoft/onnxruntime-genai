// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <string>
#include <variant>

namespace benchmark {

// The number of tokens to generate and use as a prompt or the actual prompt.
using PromptNumberOfTokensOrContent = std::variant<size_t, std::string>;

struct Options {
  std::string model_path;
  PromptNumberOfTokensOrContent prompt_num_tokens_or_content{size_t{16}};
  size_t num_tokens_to_generate{128};
  size_t batch_size{1};
  size_t num_iterations{5};
  size_t num_warmup_iterations{1};
  bool verbose{};
};

Options ParseOptionsFromCommandLine(int argc, const char* const* argv);

}  // namespace benchmark
