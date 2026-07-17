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
//   2. Environment:   ORT_TELEMETRY_DISABLED=1 (shared with ONNX Runtime) or
//                     ORT_GENAI_TELEMETRY_DISABLED=1 (disables non-essential events)
//   3. Runtime API:   OgaSetTelemetryEnabled(false) (disables non-essential events dynamically)

#pragma once

#include <atomic>
#include <cstdint>
#include <memory>
#include <mutex>
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
  void LogModelLoadStart(uint32_t session_id);
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
  void LogGenerateStart(uint32_t session_id, uint32_t generator_id, int prompt_tokens,
                        const std::string& input_modality);
  void LogGenerateEnd(uint32_t session_id, uint32_t generator_id,
                      int total_tokens,
                      double time_to_first_token_ms, double total_time_ms,
                      double tokens_per_second);

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

  // Acquires init_mutex_ and returns an owned lock only when telemetry is live (initialized, logger
  // valid, and -- when require_enabled is true -- enabled); otherwise returns an empty lock. ProcessInfo
  // passes require_enabled=false so it emits whenever telemetry is initialized (i.e. outside CI/testing)
  // even when detailed lifecycle events are disabled. Held across event emission so Log* is serialized with Shutdown,
  // preventing a use-after-free on impl_ during teardown.
  std::unique_lock<std::mutex> LockForLogging(bool require_enabled = true);

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
  std::atomic<uint32_t> next_session_id_{1};
  std::string app_session_guid_;  // Tier 1 identity: process-wide GUID (logger context)
  std::mutex init_mutex_;
};

}  // namespace Generators
