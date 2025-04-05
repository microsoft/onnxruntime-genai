// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#pragma once

#include <algorithm>
#include <array>
#include <assert.h>
#include <atomic>
#include <cmath>
#include <cstring>
#include "filesystem.h"
#include <functional>
#include <iostream>
#include "span.h"
#include <memory>
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
struct Search;
struct Tokenizer;

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

  const Config& config;                  // The model outlives the GeneratorParams
  Config::Search search{config.search};  // Copy of the search parameters from the config

  int max_batch_size{0};
  bool use_graph_capture{};
  int BatchBeamSize() const { return search.num_beams * search.batch_size; }

  DeviceInterface* p_device{};  // Scoring device (usually CPU, but can be CUDA)

  cpu_span<int32_t> aux_input_ids{};  // Intermediate solution to be used with SetInputs function for multimodal and whisper models

  struct Whisper {
    std::shared_ptr<Tensor> input_features;   // float32 [batch_size, number_of_mels, number_of_frames]
    std::shared_ptr<Tensor> alignment_heads;  // int32 [num_alignment_heads, 2]
  };

  std::variant<Whisper> inputs;

  struct Input {
    std::string name;
    std::shared_ptr<Tensor> tensor;
  };

  // A list of extra model inputs that will be matched at runtime based on name
  std::vector<Input> extra_inputs;

  void SetInputs(const NamedTensors& inputs);
};

struct Generator : LeakChecked<Generator> {
  Generator(const Model& model, const GeneratorParams& params);

  bool IsDone() const;
  void AppendTokens(cpu_span<const int32_t> input_ids);
  void GenerateNextToken();
  void RewindToLength(size_t new_length);  // Rewind state to new_length
  DeviceSpan<float> GetLogits();
  void SetLogits(DeviceSpan<float> logits);
  void SetRuntimeOption(const char* key, const char* value);
  bool IsSessionTerminated() const;

  DeviceSpan<int32_t> GetSequence(size_t index) const;

  std::shared_ptr<const Model> model_;
  std::unique_ptr<State> state_;
  std::unique_ptr<Search> search_;
  bool computed_logits_{};  // Set to true in ComputeLogits() and false after appending a token to ensure a 1 to 1 call ratio

 private:
  DeviceSpan<int32_t> AllocateInputIdsOnDevice(cpu_span<const int32_t> input_ids);
  void AuxAppendTokens(cpu_span<const int32_t> input_ids);
  void ComputeLogits(DeviceSpan<int32_t> next_tokens);
  enum Action { standard,   // Default, set in any other case
                generated,  // Set after GenerateNextToken
                rewound };  // Set after RewindToLength
  Action last_action_{standard};
};

struct OrtGlobals {
  OrtGlobals();

  std::unique_ptr<OrtEnv> env_;

  struct Allocator {
    std::unique_ptr<Ort::Allocator> allocator_;
    std::unique_ptr<OrtSession> session_;
  };
  Allocator device_allocators_[static_cast<int>(DeviceType::MAX)];

 private:
  OrtGlobals(const OrtGlobals&) = delete;
  void operator=(const OrtGlobals&) = delete;
};

std::unique_ptr<OrtGlobals>& GetOrtGlobals();
void Shutdown();  // Do this once at exit, Ort code will fail after this call
OrtEnv& GetOrtEnv();

std::shared_ptr<Model> CreateModel(OrtEnv& ort_env, const char* config_path, const RuntimeSettings* settings = nullptr);
std::shared_ptr<Model> CreateModel(OrtEnv& ort_env, std::unique_ptr<Config> config);
std::shared_ptr<GeneratorParams> CreateGeneratorParams(const Model& model);
std::shared_ptr<GeneratorParams> CreateGeneratorParams(const Config& config);  // For benchmarking purposes only
std::unique_ptr<Generator> CreateGenerator(const Model& model, const GeneratorParams& params);

// Fallback to copy between two separate device buffers by going through CPU memory (slow unless we're the CPU device)
void CopyThroughCpu(DeviceBuffer& dest, size_t begin_dest, DeviceBuffer& source, size_t begin_source, size_t size_in_bytes);

float Float16ToFloat32(uint16_t v);  // v is a IEEE 752-2008 binary16 format, 1 sign bit, 5 bit exponent, 10 bit fraction

}  // namespace Generators
