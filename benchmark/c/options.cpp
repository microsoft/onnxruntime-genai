// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "options.h"

#include <cstdlib>
#include <charconv>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <system_error>

namespace benchmark {

namespace {

[[noreturn]] void PrintHelpAndExit(const char* program_name, int exit_code) {
  Options defaults{};
  std::ostringstream s;

  s << "Usage: " << program_name << " -i <model path> <other options>\n"
    << "  Options:\n"
    << "    -i,--input_folder <path>\n"
    << "      Path to the ONNX model directory to benchmark, compatible with onnxruntime-genai.\n"
    << "    -b,--batch_size <number>\n"
    << "      Number of sequences to generate in parallel. Default: " << defaults.batch_size << "\n"
    << "    -l,--prompt_length <number>\n"
    << "      Number of tokens in the prompt. Default: " << defaults.num_prompt_tokens << "\n"
    << "    -g,--generation_length <number>\n"
    << "      Number of tokens to generate. Default: " << defaults.num_tokens_to_generate << "\n"
    << "    -r,--repetitions <number>\n"
    << "      Number of times to repeat the benchmark. Default: " << defaults.num_iterations << "\n"
    << "    -w,--warmup <number>\n"
    << "      Number of warmup runs before benchmarking. Default: " << defaults.num_warmup_iterations << "\n"
    << "    -v,--verbose\n"
    << "      Show more informational output.\n"
    << "    -h,--help\n"
    << "      Show this help message and exit.\n";

  std::cerr << s.str();
  std::exit(exit_code);
}

template <typename T>
T ParseNumber(std::string_view s) {
  T n;
  const auto *s_begin = s.data(), *s_end = s.data() + s.size();
  const auto [ptr, ec] = std::from_chars(s_begin, s_end, n);
  if (ec != std::errc{} || ptr != s_end) {
    throw std::runtime_error(std::string{"Failed to parse option value as number: "}.append(s));
  }
  return n;
}

void VerifyOptions(const Options& opts) {
  if (opts.model_path.empty()) {
    throw std::runtime_error("ONNX model directory path must be provided.");
  }
}

}  // namespace

Options ParseOptionsFromCommandLine(int argc, const char* const* argv) {
  const char* const program_name = argc > 0 ? argv[0] : "model_benchmark";
  try {
    Options opts{};

    auto next_arg = [argc, argv](int& idx) {
      if (idx + 1 >= argc) {
        throw std::runtime_error("Option value not provided.");
      }
      return std::string_view{argv[++idx]};
    };

    for (int i = 1; i < argc; ++i) {
      std::string_view arg{argv[i]};

      if (arg == "-i" || arg == "--input_folder") {
        opts.model_path = next_arg(i);
      } else if (arg == "-b" || arg == "--batch_size") {
        opts.batch_size = ParseNumber<size_t>(next_arg(i));
      } else if (arg == "-l" || arg == "--prompt_length") {
        opts.num_prompt_tokens = ParseNumber<size_t>(next_arg(i));
      } else if (arg == "-g" || arg == "--generation_length") {
        opts.num_tokens_to_generate = ParseNumber<size_t>(next_arg(i));
      } else if (arg == "-r" || arg == "--repetitions") {
        opts.num_iterations = ParseNumber<size_t>(next_arg(i));
      } else if (arg == "-w" || arg == "--warmup") {
        opts.num_warmup_iterations = ParseNumber<size_t>(next_arg(i));
      } else if (arg == "-v" || arg == "--verbose") {
        opts.verbose = true;
      } else if (arg == "-h" || arg == "--help") {
        PrintHelpAndExit(program_name, 0);
      } else {
        throw std::runtime_error(std::string{"Unknown option: "}.append(arg));
      }
    }

    VerifyOptions(opts);

    return opts;
  } catch (const std::exception& e) {
    std::cerr << "Error: " << e.what() << "\n";
    PrintHelpAndExit(program_name, 1);
  }
}

}  // namespace benchmark
