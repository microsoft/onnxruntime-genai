#include <algorithm>
#include <chrono>
#include <iostream>
#include <numeric>
#include <vector>

#include "ort_genai.h"

#include "options.h"

namespace {

using Clock = std::chrono::steady_clock;
using Duration = Clock::duration;
using DurationFp = std::chrono::duration<float, Duration::period>;

using MicrosecondsFp = std::chrono::duration<float, std::chrono::microseconds::period>;

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

  stats.p50 = sorted[stats.n * 0.5];
  stats.p90 = sorted[stats.n * 0.9];
  stats.p99 = sorted[stats.n * 0.99];

  return stats;
}

}  // namespace

void RunBenchmark(const benchmark::Options& opts) {
  auto model = OgaModel::Create(opts.model_path.c_str());

  auto tokenizer = OgaTokenizer::Create(*model);

  const char* prompt = "My perfect Sunday is ";
  auto prompt_sequences = OgaSequences::Create();

  tokenizer->Encode(prompt, *prompt_sequences);

  const size_t num_prompt_tokens = prompt_sequences->Get(0).size();
  const size_t num_tokens = num_prompt_tokens + opts.num_tokens_to_generate;

  auto make_generator_params = [&] {
    auto params = OgaGeneratorParams::Create(*model);
    params->SetSearchOption("max_length", num_tokens);
    params->SetSearchOption("min_length", num_tokens);
    params->SetInputSequences(*prompt_sequences);
    return params;
  };

  // warmup
  for (size_t i = 0; i < opts.num_warmup_iterations; ++i) {
    auto params = make_generator_params();

    auto generator = OgaGenerator::Create(*model, *params);
    while (!generator->IsDone()) {
      generator->ComputeLogits();
      generator->GenerateNextToken();
    }
  }

  std::vector<Duration> e2e_gen_times, prompt_processing_times, token_gen_times, sampling_times;
  // note: be sure to reserve enough to avoid vector reallocations in the measured code
  e2e_gen_times.reserve(opts.num_iterations);
  prompt_processing_times.reserve(opts.num_iterations);
  token_gen_times.reserve(opts.num_iterations * (opts.num_tokens_to_generate - 1));
  sampling_times.reserve(opts.num_iterations * opts.num_tokens_to_generate);

  for (size_t i = 0; i < opts.num_iterations; ++i) {
    auto params = make_generator_params();

    auto generator = OgaGenerator::Create(*model, *params);

    {
      Timing e2e_gen_timing{e2e_gen_times};

      {
        Timing prompt_processing_timing{prompt_processing_times};
        generator->ComputeLogits();
      }

      {
        Timing sampling_timing{sampling_times};
        generator->GenerateNextToken();
      }

      while (!generator->IsDone()) {
        {
          Timing token_gen_timing{token_gen_times};
          generator->ComputeLogits();
        }

        {
          Timing sampling_timing{sampling_times};
          generator->GenerateNextToken();
        }
      }
    }

    // auto output_sequence = generator->GetSequence(0);
    // auto out_string = tokenizer->Decode(output_sequence);
  }

  {
    const auto e2e_gen_stats = ComputeStats(e2e_gen_times);
    const auto prompt_processing_stats = ComputeStats(prompt_processing_times);
    const auto token_gen_stats = ComputeStats(token_gen_times);
    const auto sampling_stats = ComputeStats(sampling_times);

    auto write_per_token_stats = [](const std::string& label,
                                    const Statistics& stats,
                                    const size_t tokens_per_measurement) {
      const auto avg_us = MicrosecondsFp{stats.average};
      std::cout << label << ":\n"
                << "\tavg latency (us): " << avg_us.count() / tokens_per_measurement
                << ",\tavg throughput (tps): " << 1.0e6f / avg_us.count() * tokens_per_measurement
                << ",\tp50 latency (us): " << MicrosecondsFp{stats.p50}.count() / tokens_per_measurement
                << ",\tn: " << stats.n << " * " << tokens_per_measurement
                << "\n";
    };

    auto write_stats = [](const std::string& label,
                          const Statistics& stats) {
      const auto avg_us = MicrosecondsFp{stats.average};
      std::cout << label << ":\n"
                << "\tavg latency (us): " << avg_us.count()
                << ",\tp50 latency (us): " << MicrosecondsFp{stats.p50}.count()
                << ",\tn: " << stats.n
                << "\n";
    };

    write_per_token_stats("prompt processing per token", prompt_processing_stats, num_prompt_tokens);
    write_per_token_stats("token generation", token_gen_stats, 1);
    write_per_token_stats("token sampling", sampling_stats, 1);
    write_stats("e2e generation", e2e_gen_stats);
  }
}

int main(int argc, const char** argv) {
  try {
    const auto opts = benchmark::ParseOptionsFromCommandLine(argc, argv);
    RunBenchmark(opts);
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "Exception: " << e.what() << "\n";
    return 1;
  }
}
