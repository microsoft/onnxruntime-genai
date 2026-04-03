// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

// Telemetry for ONNX Runtime GenAI.
// Build with CMake option ENABLE_TELEMETRY=ON to enable telemetry.
//
// Telemetry traces the library lifecycle using paired start/end events,
// following the same pattern as ONNX Runtime's telemetry system.
//
// Opt-out mechanisms (in priority order):
//   1. Compile-time:  ENABLE_TELEMETRY=OFF (zero overhead, no telemetry code)
//   2. Environment:   ORTGENAI_TELEMETRY_ENABLED=0 (no events sent)
//   3. Runtime API:   OgaSetTelemetryEnabled(false) (dynamic control)

#pragma once

#include <atomic>
#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <string_view>

namespace Generators {

struct Config;
struct GeneratorParams;

class GenAiTelemetry {
 public:
  static GenAiTelemetry& Instance();

  // Returns true if the singleton has been destroyed (during static deinitialization)
  static bool IsDestroyed();

  // Lifecycle
  void Initialize();
  void Shutdown();

  // Control
  bool IsEnabled() const;
  void SetEnabled(bool enabled);

  // Allocate a unique session ID for correlating events across a model lifecycle
  uint32_t AllocateSessionId();

  // --- Events matching ORT's telemetry pattern ---

  // ProcessInfo: Emitted once per process on first model load.
  // Device fingerprint + library version for MAD/DAD.
  void LogProcessInfo();

  // ModelLoadStart/End: Paired events tracing model creation lifecycle.
  void LogModelLoadStart(uint32_t session_id);
  void LogModelLoad(uint32_t session_id,
                    const std::string& model_type,
                    const std::string& execution_providers,
                    const std::string& selected_device,
                    int vocab_size,
                    int context_length,
                    bool is_in_memory);
  void LogModelLoadEnd(uint32_t session_id, bool is_success, double load_time_ms,
                       const std::string& error_message = "");

  // GeneratorCreate: Emitted when a generator is created (GenAI-specific).
  void LogGeneratorCreate(uint32_t session_id,
                          int batch_size, int num_beams, int max_length,
                          float top_k, float top_p, float temperature,
                          bool use_graph_capture, bool has_guidance);

  // GenerateStart/End: Paired events tracing inference lifecycle.
  void LogGenerateStart(uint32_t session_id, uint32_t generator_id, int prompt_tokens);
  void LogGenerateEnd(uint32_t session_id, uint32_t generator_id,
                      int total_tokens, int prompt_tokens,
                      double time_to_first_token_ms, double total_time_ms,
                      double tokens_per_second, const std::string& execution_provider);

  // RuntimeError: Emitted on failures.
  void LogRuntimeError(uint32_t session_id,
                       const std::string& error_type,
                       const std::string& error_message,
                       const std::string& context);

  // RuntimePerf: Aggregate performance stats (emitted on shutdown).
  void LogRuntimePerf(uint32_t session_id, int total_runs, double total_run_duration_ms);

 private:
  GenAiTelemetry() = default;
  ~GenAiTelemetry();
  GenAiTelemetry(const GenAiTelemetry&) = delete;
  GenAiTelemetry& operator=(const GenAiTelemetry&) = delete;

  struct Impl;
  std::unique_ptr<Impl> impl_;
  std::atomic<bool> enabled_{true};
  std::atomic<bool> initialized_{false};
  std::atomic<bool> process_info_logged_{false};
  std::atomic<uint32_t> next_session_id_{1};
  std::mutex init_mutex_;
};

}  // namespace Generators
