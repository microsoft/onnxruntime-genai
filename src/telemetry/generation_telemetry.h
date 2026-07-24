// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <cstddef>
#include <cstdint>
#include <string_view>

#if defined(ORTGENAI_ENABLE_TELEMETRY)
#include <chrono>
#include <string>
#endif

namespace Generators {

class GenerationTelemetry {
 public:
#if defined(ORTGENAI_ENABLE_TELEMETRY)
  class AppendTrackingSuppression {
   public:
    explicit AppendTrackingSuppression(GenerationTelemetry& telemetry);
    ~AppendTrackingSuppression();

    AppendTrackingSuppression(const AppendTrackingSuppression&) = delete;
    AppendTrackingSuppression& operator=(const AppendTrackingSuppression&) = delete;

   private:
    GenerationTelemetry& telemetry_;
    bool previous_value_;
  };

  GenerationTelemetry(uint32_t session_id, bool is_transducer);
  ~GenerationTelemetry();

  uint32_t GeneratorId() const { return generator_id_; }
  void LogGeneratorCreate(int batch_size, int num_beams, int max_length,
                          int top_k, float top_p, float temperature,
                          bool do_sample, bool use_graph_capture, bool has_guidance);
  void LogAdapterActivated();
  bool BeginAppend();
  void CompleteAppend(size_t input_token_count, int num_beams, std::string_view input_modality);
  void AddAudioDurationMs(double duration_ms);
  void OnTokenGenerated(int64_t active_token_count);
  void OnRewind(int64_t rewound_token_count);
  AppendTrackingSuppression SuppressAppendTracking();

 private:
  bool IsEnabled() const;
  void Finish();
  void Reset();
  void CloseRequestBoundary();

  uint32_t session_id_;
  uint32_t generator_id_;
  int64_t prompt_tokens_{0};
  int64_t generated_tokens_{0};
  int64_t rewind_count_{0};
  int64_t rewound_tokens_{0};
  double audio_duration_ms_{0.0};
  bool first_token_logged_{false};
  bool generate_start_logged_{false};
  bool generation_abandoned_{false};
  bool append_tracking_suppressed_{false};
  bool append_tracking_{false};
  bool is_transducer_;
  std::string input_modality_;
  std::chrono::steady_clock::time_point start_time_;
  std::chrono::steady_clock::time_point first_token_time_;
  std::chrono::steady_clock::time_point last_token_time_;
  std::chrono::steady_clock::time_point append_start_time_;
#else
  class AppendTrackingSuppression {};

  GenerationTelemetry(uint32_t, bool) {}
  ~GenerationTelemetry() = default;

  uint32_t GeneratorId() const { return 0; }
  void LogGeneratorCreate(int, int, int, int, float, float, bool, bool, bool) {}
  void LogAdapterActivated() {}
  bool BeginAppend() { return false; }
  void CompleteAppend(size_t, int, std::string_view) {}
  void AddAudioDurationMs(double) {}
  void OnTokenGenerated(int64_t) {}
  void OnRewind(int64_t) {}
  AppendTrackingSuppression SuppressAppendTracking() { return {}; }
#endif
};

}  // namespace Generators
