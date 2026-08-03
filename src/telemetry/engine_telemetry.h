// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <cstddef>
#include <cstdint>

#if defined(ORTGENAI_ENABLE_TELEMETRY)
#include <chrono>
#endif

namespace Generators {

struct EngineRequestInfo;

uint32_t AllocateEngineTelemetryId();

class EngineRequestTelemetry {
 public:
#if defined(ORTGENAI_ENABLE_TELEMETRY)
  using FinalizeCallback = void (*)(
      uint32_t session_id, uint32_t engine_id, uint32_t request_id,
      const EngineRequestInfo& info, int64_t end_timestamp_ms);

  EngineRequestTelemetry() = default;
  ~EngineRequestTelemetry();

#if defined(__GNUC__) && !defined(_WIN32)
  [[gnu::visibility("hidden")]]
#endif
  static void SetFinalizeCallbackForTest(FinalizeCallback callback);

  void Begin(uint32_t session_id, uint32_t engine_id, bool dynamic_batching,
             int64_t prompt_tokens);
  void OnScheduled();
  void OnStep(int64_t sequence_length, size_t batch_size, bool completed);
  void OnRemoved(bool completed);

 private:
  enum class Outcome {
    Completed,
    Removed,
    Abandoned,
  };

  bool IsEnabled() const;
  void Finish(Outcome outcome, std::chrono::steady_clock::time_point end_time);
  void Reset();

  uint32_t session_id_{};
  uint32_t engine_id_{};
  uint32_t request_id_{};
  int64_t prompt_tokens_{};
  int64_t generated_tokens_{};
  int64_t step_count_{};
  int64_t batch_size_total_{};
  int64_t max_batch_size_{};
  bool dynamic_batching_{};
  bool active_{};
  bool scheduled_{};
  bool first_token_recorded_{};
  bool completed_{};
  std::chrono::steady_clock::time_point assigned_time_;
  std::chrono::steady_clock::time_point scheduled_time_;
  std::chrono::steady_clock::time_point first_token_time_;
  std::chrono::steady_clock::time_point completed_time_;
  std::chrono::steady_clock::time_point steady_clock_anchor_;
  std::chrono::system_clock::time_point system_clock_anchor_;
#else
  ~EngineRequestTelemetry() = default;

  void Begin(uint32_t, uint32_t, bool, int64_t) {}
  void OnScheduled() {}
  void OnStep(int64_t, size_t, bool) {}
  void OnRemoved(bool) {}
#endif
};

}  // namespace Generators
