// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <array>
#include <cstring>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include "models/model.h"
#include "generator/mtp_generator.h"

#if USE_CUDA
namespace Generators::test {
namespace {

constexpr char kDisableSync[] = "disable_synchronize_execution_providers";

class RunSynchronizationRecorder {
 public:
  RunSynchronizationRecorder() : original_{Ort::api}, recorded_api_{*original_} {
    recorded_api_.Run = Run;
    recorded_api_.RunWithBinding = RunWithBinding;
    active_ = this;
    Ort::api = &recorded_api_;
  }

  ~RunSynchronizationRecorder() {
    Ort::api = original_;
    active_ = nullptr;
  }

  std::vector<bool> asynchronous_runs;

 private:
  void Record(const OrtRunOptions* options) {
    const char* value = original_->GetRunConfigEntry(options, kDisableSync);
    asynchronous_runs.push_back(value != nullptr && std::strcmp(value, "1") == 0);
  }

  static OrtStatus* ORT_API_CALL Run(OrtSession* session, const OrtRunOptions* options,
                                     const char* const* input_names, const OrtValue* const* inputs,
                                     size_t input_count, const char* const* output_names,
                                     size_t output_count, OrtValue** outputs) noexcept {
    active_->Record(options);
    return active_->original_->Run(session, options, input_names, inputs, input_count,
                                   output_names, output_count, outputs);
  }

  static OrtStatus* ORT_API_CALL RunWithBinding(OrtSession* session, const OrtRunOptions* options,
                                                const OrtIoBinding* binding) noexcept {
    active_->Record(options);
    return active_->original_->RunWithBinding(session, options, binding);
  }

  static inline RunSynchronizationRecorder* active_{};
  const OrtApi* original_;
  OrtApi recorded_api_;
};

std::shared_ptr<Model> LoadCudaModel(const char* component, const char* configured_sync) {
  const auto path = std::string{MODEL_PATH} + "nemotron-h/" + component;
  auto config = CreateConfig(GetOrtEnv(), path.c_str());
  ClearProviders(*config);
  SetProviderOption(*config, "cuda", "enable_cuda_graph", "0");
  if (configured_sync != nullptr)
    config->model.decoder.run_options = Config::RunOptions{{kDisableSync, configured_sync}};
  return CreateModel(GetOrtEnv(), std::move(config));
}

TEST(MtpPrefillTests, CudaHostPrefillSynchronizesBeforeDeviceDraftsIncludingReset) {
  auto main = LoadCudaModel("decoder", nullptr);
  auto head = LoadCudaModel("mtp", nullptr);
  // Copy only a runtime API table at least as large as the build's header declaration.
  if (Ort::api_version < ORT_API_VERSION) {
    GTEST_SKIP() << "Recording the full API table requires a matching or newer runtime.";
  }
  RunSynchronizationRecorder recorder;
  for (const char* configured_sync : std::array<const char*, 3>{nullptr, "0", "1"}) {
    SCOPED_TRACE(configured_sync == nullptr ? "unset" : configured_sync);
    main = LoadCudaModel("decoder", configured_sync);
    head = LoadCudaModel("mtp", configured_sync);
    for (int draft_tokens : {1, 2}) {
      SCOPED_TRACE(draft_tokens);
      for (bool sampling : {false, true}) {
        SCOPED_TRACE(sampling);
        for (int chunk_size : {0, 1, 2}) {
          SCOPED_TRACE(chunk_size);
          GeneratorParams params{*main};
          params.search.max_length = 24;
          params.search.past_present_share_buffer = false;
          params.search.chunk_size = chunk_size;
          params.search.do_sample = sampling;
          params.search.top_k = 1;
          params.search.top_p = 1.0f;
          params.search.temperature = 1.0f;
          params.search.random_seed = 1;
          params.speculative.max_draft_tokens = draft_tokens;
          MtpGenerator generator{*main, *head, params};
          for (const auto& prompt : {std::vector<int32_t>{1, 2, 3}, std::vector<int32_t>{3}}) {
            SCOPED_TRACE(prompt.size());
            recorder.asynchronous_runs.clear();
            generator.AppendTokens(cpu_span<const int32_t>{prompt});
            ASSERT_FALSE(recorder.asynchronous_runs.empty());
            for (bool asynchronous : recorder.asynchronous_runs)
              EXPECT_FALSE(asynchronous) << "Host-token prefill must finish before temporary inputs are released.";

            recorder.asynchronous_runs.clear();
            while (!generator.IsDone()) generator.GenerateNextToken();
            ASSERT_FALSE(recorder.asynchronous_runs.empty());
            for (bool asynchronous : recorder.asynchronous_runs)
              EXPECT_EQ(asynchronous, draft_tokens > 1 && !sampling &&
                                          (configured_sync == nullptr || std::strcmp(configured_sync, "1") == 0))
                  << "Only persistent device-token drafts may use asynchronous execution.";
            EXPECT_GT(generator.Trials(), 0u);
            if (!sampling) {
              EXPECT_GT(generator.Accepts(), 0u);
              EXPECT_GT(generator.Trials(), generator.Accepts());
            }
            generator.Reset();
          }
        }
      }
    }
  }
}

}  // namespace
}  // namespace Generators::test
#endif
