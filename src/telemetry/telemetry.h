// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

// Telemetry for ONNX Runtime GenAI.
// Build with CMake option ENABLE_TELEMETRY=ON to enable telemetry.
//
// Telemetry traces the library lifecycle using paired start/end events,
// following the same pattern as ONNX Runtime's telemetry system.
//
// Telemetry controls (in priority order):
//   1. Compile-time:  ENABLE_TELEMETRY=OFF (telemetry SDK not linked; API calls are no-ops)
//   2. Environment:   ORT_TELEMETRY_DISABLED=1 (disables non-essential events)
//   3. Runtime API:   OgaSetTelemetryEnabled(false) (disables non-essential events dynamically)

#pragma once

#include <atomic>
#include <cstdint>
#include <memory>
#include <mutex>
#include <shared_mutex>
#include <string>
#include <string_view>
#include <utility>

namespace Generators {

struct Config;
struct GeneratorParams;

// Model/device configuration captured at model load time.
struct ModelLoadInfo {
  std::string model_type;
  std::string model_family;         // Aggregate family above type (qwen3_vl -> qwen)
  std::string execution_providers;  // Configured EP list (comma-separated)
  std::string selected_device;      // Actual device chosen (CPU/CUDA/DML/...)
  std::string modality;             // Supported modality (grouped): text / vision / audio / multimodal
  std::string attention_type;       // full / gqa / sliding_window / hybrid
  std::string transcription_mode;   // Audio models only: streaming / batch ("" otherwise)
  int vocab_size{};
  int context_length{};
  int num_hidden_layers{};
  int hidden_size{};
  int num_attention_heads{};
  int num_key_value_heads{};
  int intra_op_num_threads{-1};      // -1 = unset
  int graph_optimization_level{-1};  // ORT enum value; -1 = unset
  int gpu_memory_mb{};               // Total device memory in MB; 0 = unknown/not applicable
  bool is_in_memory{};
};

struct GenerateEndInfo {
  int64_t total_tokens{};
  int64_t generated_tokens{};
  int64_t rewind_count{};
  int64_t rewound_tokens{};
  double audio_duration_ms{};
  double time_to_first_token_ms{};
  double total_time_ms{};
  double tokens_per_second{};
};

class GenAiTelemetry {
 public:
  static GenAiTelemetry& Instance();

  // Returns true once telemetry shutdown or static deinitialization has begun.
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
  // Returns true only when the start event was accepted by the live logger.
  bool LogModelLoadStart(uint32_t session_id);
  // Call only when LogModelLoadStart returned true.
  void LogModelLoad(uint32_t session_id, const ModelLoadInfo& info);
  void LogModelLoadEnd(uint32_t session_id, bool is_success, double load_time_ms,
                       const std::string& error_message = "");

  // GeneratorCreate: Emitted when a generator is created (GenAI-specific).
  void LogGeneratorCreate(uint32_t session_id, uint32_t generator_id,
                          int batch_size, int num_beams, int max_length,
                          int top_k, float top_p, float temperature,
                          bool do_sample, bool use_graph_capture, bool has_guidance);

  // GenerateStart/End: Paired events tracing inference lifecycle.
  // input_modality is the modality actually used for this request (grouped):
  // text / vision / audio / multimodal.
  // Returns true only when the event was accepted by the live logger.
  bool LogGenerateStart(uint32_t session_id, uint32_t generator_id, int64_t prompt_tokens,
                        const std::string& input_modality);
  void LogGenerateEnd(uint32_t session_id, uint32_t generator_id,
                      const GenerateEndInfo& info);

  // AdapterActivated: Emitted when a LoRA adapter is activated for a generator.
  // The adapter name is NOT sent (it may be sensitive); only correlation ids, so
  // we can measure runtime adapter usage in aggregate.
  void LogAdapterActivated(uint32_t session_id, uint32_t generator_id);

  // RuntimeError: Emitted on failures.
  void LogRuntimeError(uint32_t session_id,
                       const std::string& error_type,
                       const std::string& error_message,
                       const std::string& context);

 private:
  GenAiTelemetry() = default;
  ~GenAiTelemetry();
  GenAiTelemetry(const GenAiTelemetry&) = delete;
  GenAiTelemetry& operator=(const GenAiTelemetry&) = delete;

  // Acquires a shared lock and returns it only when telemetry is live (initialized, logger valid,
  // and -- when require_enabled is true -- enabled); otherwise returns an empty lock. Initialize and
  // Shutdown take the same mutex exclusively, so the logger cannot be torn down during LogEvent while
  // independent logging threads may proceed concurrently, matching ORT and 1DS ILogger's concurrent
  // ActiveLoggerCall synchronization. ProcessInfo passes require_enabled=false.
  std::shared_lock<std::shared_mutex> LockForLogging(bool require_enabled = true);

  // Runs fn under LockForLogging(require_enabled) with a catch-all guard, so telemetry emission
  // can never throw into the caller (Log* run from destructors and extern-C
  // entry points where an escaping exception would call std::terminate). fn is a
  // no-op call when telemetry is not live. Templated to avoid std::function allocation before the
  // catch-all guard is active.
  template <typename Fn>
  void RunLocked(Fn&& fn, bool require_enabled = true) {
    auto lock = LockForLogging(require_enabled);
    if (!lock) return;
    try {
      std::forward<Fn>(fn)();
    } catch (...) {
      // Telemetry must never affect host control flow.
    }
  }

  struct Impl;
  std::unique_ptr<Impl> impl_;
  std::atomic<bool> enabled_{true};
  std::atomic<bool> env_disabled_{false};
  std::atomic<bool> initialized_{false};
  std::atomic<bool> process_info_logged_{false};
#if defined(__ANDROID__)
  std::atomic<bool> android_readiness_warning_logged_{false};
#endif
  std::atomic<uint32_t> next_session_id_{1};
  std::string app_session_guid_;  // Tier 1 identity: process-wide GUID (logger context)
  std::shared_mutex mutex_;
};

}  // namespace Generators
