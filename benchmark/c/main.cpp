// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <cmath>
#include <algorithm>
#include <chrono>
#include <iostream>
#include <numeric>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#include "ort_genai.h"

#include "options.h"
#include "resource_utils.h"

namespace {

using Clock = std::chrono::steady_clock;

using Duration = Clock::duration;
using DurationFp = std::chrono::duration<float, Duration::period>;

class Timing {
 public:
  Timing(const Timing&) = delete;
  Timing& operator=(const Timing&) = delete;

  Timing(std::vector<Duration>& measurements)
      : measurements_{measurements}, start_{Clock::now()} {
  }

  ~Timing() {
    const auto measurement = Clock::now() - start_;
    measurements_.push_back(measurement);
  }

 private:
  std::vector<Duration>& measurements_;
  const Clock::time_point start_;
};

struct Statistics {
  DurationFp average{};
  DurationFp stddev{};
  DurationFp p50{};
  DurationFp p90{};
  DurationFp p99{};
  size_t n{};
};

Statistics ComputeStats(const std::vector<Duration>& measurements) {
  Statistics stats{};
  if (measurements.empty()) {
    return stats;
  }

  stats.n = measurements.size();

  const auto sum = std::accumulate(measurements.begin(), measurements.end(), Duration{0});
  stats.average = DurationFp{sum} / stats.n;

  std::vector<Duration> sorted = measurements;
  std::sort(sorted.begin(), sorted.end());

  stats.p50 = sorted[static_cast<size_t>(stats.n * 0.5)];
  stats.p90 = sorted[static_cast<size_t>(stats.n * 0.9)];
  stats.p99 = sorted[static_cast<size_t>(stats.n * 0.99)];

  if (stats.n > 1) {
    const float variance =
        std::accumulate(
            measurements.begin(), measurements.end(),
            0.0f,
            [mean = stats.average.count()](float accumulator, const Duration& m) -> float {
              const float distance_from_mean = m.count() - mean;
              return accumulator + distance_from_mean * distance_from_mean;
            }) /
        (stats.n - 1);

    const float stddev = std::sqrt(variance);
    stats.stddev = DurationFp{stddev};
  }

  return stats;
}

void WritePerTokenStats(std::string_view label,
                        const Statistics& stats,
                        const size_t tokens_per_measurement) {
  using MicrosecondsFp = std::chrono::duration<float, std::chrono::microseconds::period>;
  const auto avg_us = MicrosecondsFp{stats.average};
  std::cout << label << ":"
            << "\n\tavg (us):       " << avg_us.count()
            << "\n\tavg (tokens/s): " << 1.0e6f / avg_us.count() * tokens_per_measurement
            << "\n\tp50 (us):       " << MicrosecondsFp{stats.p50}.count()
            << "\n\tstddev (us):    " << MicrosecondsFp{stats.stddev}.count()
            << "\n\tn:              " << stats.n << " * " << tokens_per_measurement << " token(s)"
            << "\n";
}

void WriteE2EStats(std::string_view label,
                   const Statistics& stats) {
  using MillisecondsFp = std::chrono::duration<float, std::chrono::milliseconds::period>;
  std::cout << label << ":"
            << "\n\tavg (ms):       " << MillisecondsFp{stats.average}.count()
            << "\n\tp50 (ms):       " << MillisecondsFp{stats.p50}.count()
            << "\n\tstddev (ms):    " << MillisecondsFp{stats.stddev}.count()
            << "\n\tn:              " << stats.n
            << "\n";
}

std::string GeneratePrompt(size_t num_prompt_tokens, const OgaModel& model, const OgaTokenizer& tokenizer, size_t batch_size) {
  const char* const base_prompt = "A";
  auto base_prompt_sequences = OgaSequences::Create();
  for (size_t i = 0; i < batch_size; ++i) {
    tokenizer.Encode(base_prompt, *base_prompt_sequences);
  }

  auto params = OgaGeneratorParams::Create(model);
  params->SetSearchOption("max_length", static_cast<double>(num_prompt_tokens));
  params->SetSearchOption("min_length", static_cast<double>(num_prompt_tokens));

  auto generator = OgaGenerator::Create(model, *params);
  generator->AppendTokenSequences(*base_prompt_sequences);
  while (!generator->IsDone()) {
    generator->GenerateNextToken();
  }

  const auto output_sequence_length = generator->GetSequenceCount(0);
  const auto* output_sequence_data = generator->GetSequenceData(0);
  return std::string{tokenizer.Decode(output_sequence_data, output_sequence_length)};
}

void RunBenchmark(const benchmark::Options& opts) {
  std::unique_ptr<OgaModel> model;

  if (opts.batch_size > 1 && opts.execution_provider == "NvTensorRtRtx") {
    // Use OgaConfig::Overlay instead of RuntimeSettings for cleaner implementation
    auto config = OgaConfig::Create(opts.model_path.c_str());

    // Create JSON overlay for batch_size
    std::string batch_size_overlay = R"({
  "search": {
    "batch_size": )" + std::to_string(opts.batch_size) +
                                     R"(
  }
})";

    config->Overlay(batch_size_overlay.c_str());
    model = OgaModel::Create(*config);
  } else {
    model = OgaModel::Create(opts.model_path.c_str());
  }

  auto tokenizer = OgaTokenizer::Create(*model);

  const auto prompt = [&]() -> std::string {
    if (const size_t* num_prompt_tokens = std::get_if<size_t>(&opts.prompt_num_tokens_or_content)) {
      return GeneratePrompt(*num_prompt_tokens, *model, *tokenizer, opts.batch_size);
    }
    return std::get<std::string>(opts.prompt_num_tokens_or_content);
  }();

  auto prompt_sequences = OgaSequences::Create();

  if (opts.batch_size < 1) {
    throw std::runtime_error("Batch size must be at least 1.");
  }

  for (size_t i = 0; i < opts.batch_size; ++i) {
    tokenizer->Encode(prompt.c_str(), *prompt_sequences);
  }

  const size_t num_prompt_tokens = prompt_sequences->SequenceCount(0);
  const size_t num_tokens = num_prompt_tokens + opts.num_tokens_to_generate;

  auto make_generator_params = [&] {
    auto params = OgaGeneratorParams::Create(*model);
    params->SetSearchOption("max_length", static_cast<double>(num_tokens));
    params->SetSearchOption("min_length", static_cast<double>(num_tokens));
    return params;
  };

  const auto generator_params = make_generator_params();

  // warmup
  if (opts.verbose) std::cout << "Running warmup iterations (" << opts.num_warmup_iterations << ")...\n";
  for (size_t i = 0; i < opts.num_warmup_iterations; ++i) {
    auto generator = OgaGenerator::Create(*model, *generator_params);
    generator->AppendTokenSequences(*prompt_sequences);
    while (!generator->IsDone()) {
      generator->GenerateNextToken();
    }

    if (opts.verbose && i == 0) {
      // show prompt and output on first iteration
      std::cout << "[PROMPT BEGIN]" << prompt << "[PROMPT END]\n";
      const auto output_sequence_length = generator->GetSequenceCount(0);
      const auto* output_sequence_data = generator->GetSequenceData(0);
      const auto output = tokenizer->Decode(output_sequence_data, output_sequence_length);
      std::cout << "[OUTPUT BEGIN]" << output << "[OUTPUT END]\n";
    }
  }

  std::vector<Duration> e2e_gen_times, prompt_processing_times, token_gen_times, sampling_times;
  // note: be sure to reserve enough to avoid vector reallocations in the measured code
  e2e_gen_times.reserve(opts.num_iterations);
  prompt_processing_times.reserve(opts.num_iterations);
  token_gen_times.reserve(opts.num_iterations * (opts.num_tokens_to_generate - 1));
  sampling_times.reserve(opts.num_iterations * opts.num_tokens_to_generate);

  if (opts.verbose) std::cout << "Running iterations (" << opts.num_iterations << ")...\n";
  for (size_t i = 0; i < opts.num_iterations; ++i) {
    auto generator = OgaGenerator::Create(*model, *generator_params);

    {
      Timing e2e_gen_timing{e2e_gen_times};

      {
        Timing prompt_processing_timing{prompt_processing_times};
        generator->AppendTokenSequences(*prompt_sequences);
      }

      bool generator_done = false;

      {
        Timing sampling_timing{sampling_times};
        generator->GenerateNextToken();
        generator_done = generator->IsDone();
      }

      while (!generator_done) {
        {
          Timing token_gen_timing{token_gen_times};
          generator->GenerateNextToken();
          // Enforce stream synchronize to compute accurate token generation times
          generator_done = generator->IsDone();
        }
      }
    }
  }

  {
    std::cout << "Batch size: " << opts.batch_size
              << ", prompt tokens: " << num_prompt_tokens
              << ", tokens to generate: " << opts.num_tokens_to_generate
              << "\n";

    const auto e2e_gen_stats = ComputeStats(e2e_gen_times);
    const auto prompt_processing_stats = ComputeStats(prompt_processing_times);
    const auto token_gen_stats = ComputeStats(token_gen_times);
    const auto sampling_stats = ComputeStats(sampling_times);

    WritePerTokenStats("Prompt processing (time to first token)",
                       prompt_processing_stats, opts.batch_size * num_prompt_tokens);
    WritePerTokenStats("Token generation", token_gen_stats, opts.batch_size);
    WritePerTokenStats("Token sampling", sampling_stats, opts.batch_size);
    WriteE2EStats("E2E generation (entire generation loop)", e2e_gen_stats);

    std::cout << "Peak working set size (bytes): " << benchmark::utils::GetPeakWorkingSetSizeInBytes() << "\n";
  }
}

}  // namespace

int main(int argc, char** argv) {
  OgaHandle handle;
  try {
    const auto opts = benchmark::ParseOptionsFromCommandLine(argc, argv);
    RunBenchmark(opts);
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "Exception: " << e.what() << "\n";
    return 1;
  }
}
