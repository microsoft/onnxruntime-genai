// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#pragma once

#include <algorithm>
#include <array>
#include <assert.h>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstring>
#include "filesystem.h"
#include <functional>
#include <iostream>
#include "span.h"
#include <memory>
#include <mutex>
#include <numeric>
#include <optional>
#include <queue>
#include <set>
#include <stdexcept>
#include <string_view>
#include <unordered_map>
#include <unordered_set>
#include <variant>
#include <vector>

#include "leakcheck.h"
#include "make_string.h"
#include "models/onnxruntime_api.h"
#include "smartptrs.h"
#include "models/debugging.h"
#include "config.h"
#include "logging.h"
#include "runtime_settings.h"
#include "tensor.h"

void ThrowErrorIfSessionTerminated(bool is_session_terminated);

namespace Generators {
struct Model;
struct State;
struct TransducerState;
struct Search;
struct Tokenizer;
struct ConstrainedLogitsProcessor;
struct ExtraInput {  // Extra inputs provided via SetInputs()
  std::string name;
  std::shared_ptr<Tensor> tensor;
};

template <typename T, typename V>
bool contains(const T& t, V&& v) {
  return std::find(t.begin(), t.end(), v) != t.end();
}

template <typename T>
DeviceSpan<T> WrapTensor(DeviceInterface& device, OrtValue& value) {
  auto info = value.GetTensorTypeAndShapeInfo();
  assert(info->GetElementType() == Ort::TypeToTensorType<std::remove_const_t<T>>);
  return device.WrapMemory(std::span<T>{value.GetTensorMutableData<T>(), info->GetElementCount()});
}

DeviceSpan<uint8_t> ByteWrapTensor(DeviceInterface& device, OrtValue& value);

// OgaSequences are a vector of int32 vectors
using TokenSequences = std::vector<std::vector<int32_t>>;

std::string to_string(DeviceType device_type);
DeviceInterface* GetDeviceInterface(DeviceType type);

struct GeneratorParams : std::enable_shared_from_this<GeneratorParams>, LeakChecked<GeneratorParams>, ExternalRefCounted<GeneratorParams> {
  GeneratorParams(const Config& config);  // This constructor is only used for internal generator benchmarks
  GeneratorParams(const Model& model);

  // Co-owns the model so the aliased Config below cannot be freed while this
  // params object is alive. Null for the benchmark-only Config constructor.
  std::shared_ptr<const Model> model_;
  const Config& config;                  // Aliases model-owned Config; kept alive by model_
  Config::Search search{config.search};  // Copy of the search parameters from the config

  // Query the params to get the value set for a param
  double GetSearchNumber(std::string_view name) const;
  bool GetSearchBool(std::string_view name) const;

  int max_batch_size{0};
  bool use_graph_capture{};
  bool use_multi_profile{};
  int BatchBeamSize() const { return search.num_beams * search.batch_size; }

  DeviceInterface* p_device{};  // Scoring device (usually CPU, but can be CUDA)

  std::string guidance_type;               // e.g. json_schema or regex
  std::string guidance_data;               // e.g. rules data in json_schema or regex
  bool guidance_ff_tokens_enabled{false};  // Whether to enable ff_tokens during constrained decoding
  void SetGuidance(std::string_view type, std::string_view data, bool enable_ff_tokens);

  // Determines if past_present_share_buffer is actually enabled based on config and runtime conditions
  // Returns true only if config option is true AND (num_beams == 1 OR model is Whisper)
  bool IsPastPresentShareBufferEnabled(const std::string& model_type) const;
};

struct Generator : LeakChecked<Generator> {
  Generator(const Model& model, const GeneratorParams& params);
  ~Generator();

  bool IsDone();
  size_t TokenCount() const;
  void AppendTokens(cpu_span<const int32_t> input_ids);
  void GenerateNextToken();
  void RewindToLength(size_t new_length);  // Rewind state to new_length
  DeviceSpan<float> GetLogits();
  void SetLogits(DeviceSpan<float> logits);
  void SetRuntimeOption(const char* key, const char* value);
  bool IsSessionTerminated() const;

  DeviceSpan<int32_t> GetSequence(size_t index) const;

  // A list of extra model inputs that will be matched at runtime based on name
  std::vector<ExtraInput> extra_inputs_;
  void SetInputs(const NamedTensors& inputs);

  std::shared_ptr<const Model> model_;
  std::unique_ptr<State> state_;
  std::unique_ptr<Search> search_;
  std::unique_ptr<ConstrainedLogitsProcessor> guidance_logits_processor_;

  bool computed_logits_{};       // Set to true in ComputeLogits() and false after appending a token to ensure a 1 to 1 call ratio
  bool set_extra_inputs_{true};  // Set to false once SetExtraInputs() is called once

#if defined(ORTGENAI_ENABLE_TELEMETRY)
  // Telemetry tracking
  uint32_t telemetry_generator_id_{0};
  int telemetry_prompt_tokens_{0};
  int telemetry_generated_tokens_{0};
  std::chrono::steady_clock::time_point telemetry_start_time_;
  std::chrono::steady_clock::time_point telemetry_first_token_time_;
  std::chrono::steady_clock::time_point telemetry_last_token_time_;
  bool telemetry_first_token_logged_{false};
  bool telemetry_generate_start_logged_{false};
  bool telemetry_generation_abandoned_{false};
  std::string telemetry_input_modality_{"text"};
  bool telemetry_suppress_append_tracking_{false};
#endif

 private:
  DeviceSpan<int32_t> AllocateInputIdsOnDevice(cpu_span<const int32_t> input_ids);
  void ComputeLogits(DeviceSpan<int32_t> next_tokens);
#if defined(ORTGENAI_ENABLE_TELEMETRY)
  bool ShouldTrackTelemetry() const;
  void LogTelemetryGenerateEnd();
  void ResetTelemetryGeneration();
#endif
  enum Action { standard,   // Default, set in any other case
                generated,  // Set after GenerateNextToken
                rewound };  // Set after RewindToLength
  Action last_action_{standard};

  // Pre-computed per-token decisions: avoid repeated checks each token
  // Non-null when the model is a transducer (RNNT, TDT); points into state_.
  TransducerState* transducer_state_{nullptr};
  int phi3_rope_threshold_{};  // 0 means no ROPE rewind needed
  enum class SamplingMethod { kGreedy,
                              kTopK,
                              kTopP,
                              kTopKTopP };
  SamplingMethod sampling_method_{SamplingMethod::kGreedy};
  void InitializeSamplingMethod(const GeneratorParams& params);
  void InitializePhi3RopeThreshold(const GeneratorParams& params);
};

// Defined in generators.cpp; owned by OrtGlobals so genai add-on libraries (e.g. the CUDA
// add-on) are unloaded on teardown.
struct LibraryHandle;

struct OrtGlobals {
  OrtGlobals();
  ~OrtGlobals();

  std::unique_ptr<OrtEnv> env_;

  // Get-or-create the DeviceInterface for a device type. The interface is owned by this
  // OrtGlobals instance (in-process EPs) or by a genai add-on library it holds (CUDA), so every
  // interface is rebuilt on re-initialization after a shutdown. Thread-safe.
  DeviceInterface* GetDeviceInterface(DeviceType type);

  struct Allocator {
    // Field order matters here. The OrtAllocator returned by OrtApi::CreateAllocator (called via
    // Ort::Allocator::Create) "wraps the internal allocator from the OrtSession and becomes invalid when the session
    // does" -- see
    // https://github.com/microsoft/onnxruntime/blob/3c8c46029735a89c8d1ea0aa6c1812db5b78ad72/include/onnxruntime/core/session/onnxruntime_c_api.h#L2852-L2862
    // Members are destroyed in reverse declaration order, so session_ must be declared BEFORE allocator_ so that
    // ~allocator_ runs first.
    std::unique_ptr<OrtSession> session_;
    std::unique_ptr<Ort::Allocator> allocator_;
  };
  Allocator device_allocators_[static_cast<int>(DeviceType::MAX)];

  // Cache for dynamically built graph sessions (e.g., Cast, TopK operations)
  // Destroyed before env_ to ensure proper cleanup order
  struct SessionCache {
    std::unordered_map<uint64_t, std::unique_ptr<OrtSession>> sessions_;
    std::mutex mutex_;
  };
  SessionCache graph_session_cache_;

 private:
  OrtGlobals(const OrtGlobals&) = delete;
  void operator=(const OrtGlobals&) = delete;

  DeviceInterface* LoadCudaInterface(DeviceType type);

  std::mutex device_interfaces_mutex_;
  // Non-owning cache: values point into owned_interfaces_, the CUDA add-on library, or a
  // module-owned interface (DML). Rebuilt each env cycle.
  std::unordered_map<DeviceType, DeviceInterface*> device_interfaces_;
  // In-process interfaces owned directly by genai (CPU / WebGPU / QNN / OpenVINO / RyzenAI).
  std::vector<std::unique_ptr<DeviceInterface>> owned_interfaces_;
  // The genai CUDA add-on library (onnxruntime-genai-cuda). Holds the loaded library so that
  // unloading it (on teardown) runs the add-on's static destructors. The interface pointer it
  // provides is non-owning and lives in device_interfaces_.
  std::unique_ptr<LibraryHandle> cuda_library_;
};

std::unique_ptr<OrtGlobals>& GetOrtGlobals();
void Shutdown();  // Do this once at exit, Ort code will fail after this call
OrtEnv& GetOrtEnv();

std::shared_ptr<Model> CreateModel(OrtEnv& ort_env, const char* config_path, const RuntimeSettings* settings = nullptr);
std::shared_ptr<Model> CreateModel(OrtEnv& ort_env, std::unique_ptr<Config> config);

// Constructs a Config from `config_path`. For a model package, a variant is selected
// (auto-detected when `ep` is null/empty). For a flat directory `ep` must be null/empty.
std::unique_ptr<Config> CreateConfig(OrtEnv& ort_env, const char* config_path,
                                     const char* ep = nullptr,
                                     std::string_view json_overlay = {});

std::shared_ptr<GeneratorParams> CreateGeneratorParams(const Model& model);
std::shared_ptr<GeneratorParams> CreateGeneratorParams(const Config& config);  // For benchmarking purposes only
std::unique_ptr<Generator> CreateGenerator(const Model& model, const GeneratorParams& params);

// Fallback to copy between two separate device buffers by going through CPU memory (slow unless we're the CPU device)
void CopyThroughCpu(DeviceBuffer& dest, size_t begin_dest, DeviceBuffer& source, size_t begin_source, size_t size_in_bytes);

float Float16ToFloat32(uint16_t v);  // v is a IEEE 752-2008 binary16 format, 1 sign bit, 5 bit exponent, 10 bit fraction

std::unique_ptr<Search> CreateSearch(const GeneratorParams& params);

}  // namespace Generators
