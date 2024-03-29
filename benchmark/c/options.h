#pragma once

#include <string>

namespace benchmark {

struct Options {
  std::string model_path{};
  size_t num_tokens_to_generate{128};
  size_t num_iterations{5};
  size_t num_warmup_iterations{1};
};

Options ParseOptionsFromCommandLine(int argc, const char** argv);

}  // namespace benchmark
