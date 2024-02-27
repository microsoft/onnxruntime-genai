// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#include <algorithm>
#include <array>
#include <assert.h>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <functional>
#include "span.h"
#include <memory>
#include <numeric>
#include <queue>
#include <set>
#include <stdexcept>
#include <string_view>
#include <unordered_map>
#include <unordered_set>
#include <variant>
#include <vector>
#if USE_CUDA
#include <cuda_runtime.h>
#include "cuda_common.h"
#endif

#include "smartptrs.h"
#include "models/onnxruntime_api.h"
#include "config.h"

namespace Generators {
struct Model;
struct State;
struct Search;

// Token sequences are a vector of int32 vectors
using TokenSequences = std::vector<std::vector<int32_t>>;

// If we don't include cuda_runtime.h, we define this to avoid lots of extra #ifdefs
#ifndef USE_CUDA
using cudaStream_t = void*;
#endif

enum struct DeviceType {
  Auto,
  CPU,
  CUDA,
};

struct OrtCPUProviderOptions {};  // Stub so that ProviderOptions isn't empty without cuda

using ProviderOptions = std::variant<
    OrtCPUProviderOptions
#if USE_CUDA
    ,
    OrtCUDAProviderOptions
#endif
    >;

ProviderOptions GetDefaultProviderOptions(DeviceType device_type);

struct GeneratorParams {
  GeneratorParams() = default;  // This constructor is only used if doing a custom model handler vs built-in
  GeneratorParams(const Model& model);

  void SetInputSequences(const TokenSequences& sequences);

  // Values copied from config
  int pad_token_id{};
  int eos_token_id{};
  int vocab_size{};
  int max_length{};
  float length_penalty{};
  bool early_stopping{};

  int batch_size{};
  int sequence_length{};
  int num_beams{1};
  int BatchBeamSize() const { return num_beams * batch_size; }

  DeviceType device_type{DeviceType::CPU};
  cudaStream_t cuda_stream{};

#if 0
  struct Bert {
    std::span<const int32_t> input_ids;  // Array of [batchsize][sequence_length]
  };

  struct Gpt {
    using Gpt=Bert;
  };

  struct T5 {
    std::span<const int32_t> encoder_input_ids;  // Array of [batchsize][sequence_length]
    std::span<const int32_t> decoder_input_ids;  // Array of [batchsize][sequence_length]  
  };
  using Bart=T5;

#endif

  // TODO: Move this to a separate GPT struct
  std::span<const int32_t> input_ids;  // Array of [batchsize][sequence_length]

  struct Whisper {
    std::unique_ptr<OrtValue> input_features;  // float32 [batch_size, number_of_mels, something that is 3000]
    std::span<int32_t> decoder_input_ids;
  };

  std::variant<Whisper> inputs;

 private:
  std::unique_ptr<int32_t[]> input_ids_owner_;
};

struct Generator {
  Generator(const Model& model, const GeneratorParams& search_params);

  bool IsDone() const;
  void ComputeLogits();
  void GenerateNextToken_TopK_TopP(int top_k, float top_p, float temperature);
  void GenerateNextToken_TopP(float p, float temperature) { GenerateNextToken_TopK_TopP(0, p, temperature); }
  void GenerateNextToken_TopK(int k, float temperature) { GenerateNextToken_TopK_TopP(k, 1.0f, temperature); }
  void GenerateNextToken_Top() { GenerateNextToken_TopK_TopP(1, 1.0f, 0.0f); }
  void GenerateNextToken();

  RoamingArray<int32_t> GetSequence(int index) const;

  const Model& model_;
  std::unique_ptr<State> state_;
  std::unique_ptr<Search> search_;
  bool computed_logits_{};  // Set to true in ComputeLogits() and false after appending a token to ensure a 1 to 1 call ratio
};

std::unique_ptr<Model> CreateModel(OrtEnv& ort_env, const char* config_path, const ProviderOptions* provider_options = nullptr);
std::unique_ptr<Generator> CreateGenerator(const Model& model, const GeneratorParams& search_params);
std::vector<std::vector<int32_t>> Generate(const Model& model, const GeneratorParams& params);  // Uses CreateGenerator and a simple loop to return the entire sequence

float Float16ToFloat32(uint16_t v);  // v is a IEEE 752-2008 binary16 format, 1 sign bit, 5 bit exponent, 10 bit fraction
void top_k_indices(std::span<int32_t> top_k, std::span<const float> inputs);

}  // namespace Generators
