// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "options.h"

#include <cstdlib>
#include <charconv>
#include <fstream>
#include <iostream>
#include <iterator>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <system_error>

namespace benchmark {

namespace {

[[noreturn]] void PrintHelpAndExit(const char* program_name, int exit_code) {
  const Options defaults{};
  const auto default_prompt_num_tokens = std::get<size_t>(defaults.prompt_num_tokens_or_content);

  std::ostringstream s;

  s << "Usage: " << program_name << " -i <model path> <other options>\n"
    << "  Options:\n"
    << "    -i,--input_folder <path>\n"
    << "      Path to the ONNX model directory to benchmark, compatible with onnxruntime-genai.\n"
    << "    -e,--execution_provider <provider>\n"
    << "      Execution provider to use. Valid values are: cpu, cuda, dml, NvTensorRtRtx. Default: " << defaults.execution_provider << "\n"
    << "    -b,--batch_size <number>\n"
    << "      Number of sequences to generate in parallel. Default: " << defaults.batch_size << "\n"
    << "    Prompt options:\n"
    << "      -l,--prompt_length <number>\n"
    << "        Number of tokens in the generated prompt. Default: " << default_prompt_num_tokens << "\n"
    << "      --prompt <prompt text>\n"
    << "        Prompt text to use. Default: See --prompt_length.\n"
    << "      --prompt_file <file containing prompt text>\n"
    << "        Path to file containing prompt text to use. Default: See --prompt_length.\n"
    << "      Note: --prompt_length, --prompt, and --prompt_file are mutually exclusive.\n"
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

std::string ReadFileContent(std::string_view file_path) {
  // `file_path` is assumed to be a null-terminated string.
  std::ifstream input_stream{file_path.data()};
  if (!input_stream) {
    throw std::runtime_error(std::string{"Failed to read file: "}.append(file_path));
  }

  std::istreambuf_iterator<char> input_begin{input_stream}, input_end{};
  return std::string{input_begin, input_end};
}

void ValidateExecutionProvider(const std::string& provider) {
  if (provider != "cpu" && provider != "cuda" && provider != "dml" && provider != "NvTensorRtRtx") {
    throw std::runtime_error("Invalid execution provider: " + provider + ". Valid values are: cpu, cuda, dml, NvTensorRtRtx");
  }
}

void VerifyOptions(const Options& opts) {
  if (opts.model_path.empty()) {
    throw std::runtime_error("ONNX model directory path must be provided.");
  }

  // validate execution provider since it has a valid value
  ValidateExecutionProvider(opts.execution_provider);
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

    std::optional<PromptNumberOfTokensOrContent> prompt_num_tokens_or_content{};

    for (int i = 1; i < argc; ++i) {
      std::string_view arg{argv[i]};

      if (arg == "-i" || arg == "--input_folder") {
        opts.model_path = next_arg(i);
      } else if (arg == "-e" || arg == "--execution_provider") {
        opts.execution_provider = next_arg(i);
      } else if (arg == "-b" || arg == "--batch_size") {
        opts.batch_size = ParseNumber<size_t>(next_arg(i));
      } else if (arg == "-l" || arg == "--prompt_length") {
        prompt_num_tokens_or_content = ParseNumber<size_t>(next_arg(i));
      } else if (arg == "--prompt") {
        prompt_num_tokens_or_content = std::string{next_arg(i)};
      } else if (arg == "--prompt_file") {
        prompt_num_tokens_or_content = ReadFileContent(next_arg(i));
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

    if (prompt_num_tokens_or_content.has_value()) {
      opts.prompt_num_tokens_or_content = std::move(*prompt_num_tokens_or_content);
    }

    VerifyOptions(opts);

    return opts;
  } catch (const std::exception& e) {
    std::cerr << "Error: " << e.what() << "\n";
    PrintHelpAndExit(program_name, 1);
  }
}

}  // namespace benchmark
