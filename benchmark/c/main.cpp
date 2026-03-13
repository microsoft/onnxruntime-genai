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

static std::unique_ptr<OgaGeneratorParams> MakeGeneratorParams(const benchmark::Options& opts, const OgaModel& model, size_t num_tokens) {
  auto params = OgaGeneratorParams::Create(model);
  if (opts.max_length != -1) {
    auto max_length = num_tokens;
    if (opts.max_length > 0)
      max_length = static_cast<size_t>(opts.max_length);
    params->SetSearchOption("max_length", static_cast<double>(max_length));
  }
  params->SetSearchOption("min_length", static_cast<double>(num_tokens));
  return params;
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

  if (opts.batch_size < 1) {
    throw std::runtime_error("Batch size must be at least 1.");
  }

  // Determine num_prompt_tokens early so the generator is created with the full
  // capacity (prompt + generation tokens) and its KV cache is large enough for
  // all subsequent iterations.
  size_t num_prompt_tokens;
  bool need_generate_prompt = false;
  std::string prompt;

  if (const size_t* npt = std::get_if<size_t>(&opts.prompt_num_tokens_or_content)) {
    num_prompt_tokens = *npt;
    need_generate_prompt = true;
  } else {
    prompt = std::get<std::string>(opts.prompt_num_tokens_or_content);
    auto temp_sequences = OgaSequences::Create();
    tokenizer->Encode(prompt.c_str(), *temp_sequences);
    num_prompt_tokens = temp_sequences->SequenceCount(0);
  }

  const size_t num_tokens = num_prompt_tokens + opts.num_tokens_to_generate;
  const auto generator_params = MakeGeneratorParams(opts, *model, num_tokens);

  // When reuse_generator is enabled, create a single generator and reuse it for
  // prompt generation, warmup, and benchmark iterations via RewindTo(0).
  // This avoids recreating the generator (and reallocating KV cache) each iteration.
  // Otherwise, create a fresh generator for each iteration.
  std::unique_ptr<OgaGenerator> generator;
  if (opts.reuse_generator) {
    generator = OgaGenerator::Create(*model, *generator_params);
  }

  if (need_generate_prompt) {
    // Use a generator to produce the prompt
    std::unique_ptr<OgaGenerator> temp_gen;
    if (!opts.reuse_generator) {
      temp_gen = OgaGenerator::Create(*model, *generator_params);
    }
    auto* gen = opts.reuse_generator ? generator.get() : temp_gen.get();

    const char* const base_prompt = "A";
    auto base_prompt_sequences = OgaSequences::Create();
    for (size_t i = 0; i < opts.batch_size; ++i) {
      tokenizer->Encode(base_prompt, *base_prompt_sequences);
    }
    gen->AppendTokenSequences(*base_prompt_sequences);
    while (!gen->IsDone() && gen->TokenCount() < num_prompt_tokens) {
      gen->GenerateNextToken();
    }
    const auto output_sequence_length = gen->TokenCount();
    const auto* output_sequence_data = gen->GetSequenceData(0);
    prompt = std::string{tokenizer->Decode(output_sequence_data, output_sequence_length)};
  }

  auto prompt_sequences = OgaSequences::Create();
  for (size_t i = 0; i < opts.batch_size; ++i) {
    tokenizer->Encode(prompt.c_str(), *prompt_sequences);
  }

  // warmup
  if (opts.verbose) std::cout << "Running warmup iterations (" << opts.num_warmup_iterations << ")...\n";
  for (size_t i = 0; i < opts.num_warmup_iterations; ++i) {
    std::unique_ptr<OgaGenerator> new_gen;
    if (opts.reuse_generator) {
      generator->RewindTo(0);
    } else {
      new_gen = OgaGenerator::Create(*model, *generator_params);
    }
    auto* gen = opts.reuse_generator ? generator.get() : new_gen.get();

    gen->AppendTokenSequences(*prompt_sequences);
    const size_t target_token_count = gen->TokenCount() + opts.num_tokens_to_generate;
    while (!gen->IsDone() && gen->TokenCount() < target_token_count) {
      gen->GenerateNextToken();
    }

    if (opts.verbose && i == 0) {
      // show prompt and output on first iteration
      std::cout << "[PROMPT BEGIN]" << prompt << "[PROMPT END]\n";
      const auto output_sequence_length = gen->TokenCount();
      const auto* output_sequence_data = gen->GetSequenceData(0);
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
    std::unique_ptr<OgaGenerator> new_gen;
    if (opts.reuse_generator) {
      generator->RewindTo(0);
    } else {
      new_gen = OgaGenerator::Create(*model, *generator_params);
    }
    auto* gen = opts.reuse_generator ? generator.get() : new_gen.get();

    {
      Timing e2e_gen_timing{e2e_gen_times};

      {
        Timing prompt_processing_timing{prompt_processing_times};
        gen->AppendTokenSequences(*prompt_sequences);
      }

      const size_t target_token_count = gen->TokenCount() + opts.num_tokens_to_generate;
      bool generator_done = false;

      {
        Timing sampling_timing{sampling_times};
        gen->GenerateNextToken();
        generator_done = gen->IsDone();
      }

      while (!generator_done && gen->TokenCount() < target_token_count) {
        {
          Timing token_gen_timing{token_gen_times};
          gen->GenerateNextToken();
          // Enforce stream synchronize to compute accurate token generation times
          generator_done = gen->IsDone();
        }
      }
    }
  }

  // Release the generator before printing results
  generator.reset();

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
