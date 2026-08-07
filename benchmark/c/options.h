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
  std::string execution_provider{"cpu"};
  PromptNumberOfTokensOrContent prompt_num_tokens_or_content{size_t{16}};
  size_t num_tokens_to_generate{128};
  size_t batch_size{1};
  size_t num_iterations{5};
  size_t num_warmup_iterations{1};
  int64_t max_length{0};
  bool verbose{};
  bool reuse_generator{};
  bool use_random_tokens{};

  // Optional ORT profiling for the "middle" benchmark iteration only.
  // Selected iteration index (0-based) = num_iterations / 2
  //   n=1 -> 0 (1st), n=2 -> 1 (2nd), n=3 -> 1 (2nd), n=4 -> 2 (3rd), ...
  // When true, profiling is toggled on around that phase only.
  // ORT writes the resulting JSON to "<prefix>_<timestamp>.json":
  //   prefill phase    -> prefix "prefill_profile"
  //   generation phase -> prefix "generation_profile"
  // (one file per Run(): one for prefill, one per generated token for generation).
  bool profile_prefill{};
  bool profile_generation{};
};

Options ParseOptionsFromCommandLine(int argc, const char* const* argv);

}  // namespace benchmark
