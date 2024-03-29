#include "options.h"

#include <cstdlib>
#include <charconv>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string_view>

namespace benchmark {

namespace {

[[noreturn]] void PrintHelpAndExit(const char* program_name, int exit_code) {
  Options defaults{};
  std::ostringstream s;

  s << "Usage: " << program_name << " -i <model path> <other options>\n"
    << "  Options:\n"
    << "    -i <path>     - Path to the ONNX model directory to benchmark, compatible with onnxruntime-genai.\n"
    << "    -g <number>   - Number of tokens to generate. Default: " << defaults.num_tokens_to_generate << "\n"
    << "    -r <number>   - Number of times to repeat the benchmark. Default: " << defaults.num_iterations << "\n"
    << "    -w <number>   - Number of warmup runs before benchmarking. Default: " << defaults.num_warmup_iterations << "\n"
    << "    -h            - Show this help message and exit.\n";

  std::cerr << s.str();
  std::exit(exit_code);
}

template <typename T>
T ParseNumber(std::string_view s) {
  T n;
  const auto *s_begin = s.data(), *s_end = s.data() + s.size();
  const auto [ptr, ec] = std::from_chars(s_begin, s_end, n);
  if (ec != std::errc{} || ptr != s_end) {
    throw std::runtime_error("Failed to parse option value.");
  }
  return n;
}

}  // namespace

Options ParseOptionsFromCommandLine(int argc, const char** argv) {
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

      if (arg == "-i") {
        opts.model_path = next_arg(i);
      } else if (arg == "-g") {
        opts.num_tokens_to_generate = ParseNumber<size_t>(next_arg(i));
      } else if (arg == "-r") {
        opts.num_iterations = ParseNumber<size_t>(next_arg(i));
      } else if (arg == "-w") {
        opts.num_warmup_iterations = ParseNumber<size_t>(next_arg(i));
      } else if (arg == "-h") {
        PrintHelpAndExit(program_name, 0);
      } else {
        throw std::runtime_error("Unknown option.");
      }
    }

    if (opts.model_path.empty()) {
      throw std::runtime_error("ONNX model directory path must be provided.");
    }

    return opts;
  } catch (const std::exception& e) {
    std::cerr << "Error: " << e.what() << "\n";
    PrintHelpAndExit(program_name, 1);
  }
}

}  // namespace benchmark
