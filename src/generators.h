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

struct SearchParams {
  SearchParams() = default;  // This constructor is only used if doing a custom model handler vs built-in
  SearchParams(const Model& model);

  std::unique_ptr<Search> CreateSearch() const;

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
};

struct Search {
  Search(const SearchParams& params) : params_{params} {}
  virtual ~Search() = default;

  virtual RoamingArray<int32_t> GetNextTokens() = 0;
  virtual RoamingArray<int32_t> GetNextIndices() = 0;
  virtual RoamingArray<int32_t> GetSequenceLengths() = 0;
  virtual int GetSequenceLength() const = 0;
  virtual RoamingArray<int32_t> GetSequence(int index) = 0;

  virtual void SetLogits(RoamingArray<float> logits) = 0;
  virtual bool IsDone() const = 0;

  // TODO: Beam Search only, this should be removed and made automatic
  virtual void Finalize(size_t /*num_return_sequences*/, RoamingArray<int32_t> /*output*/, RoamingArray<float> /*sequence_scores*/) { assert(false); }

  virtual void SelectTop() = 0;
  virtual void SampleTopP(float /*p*/, float /*temperature*/) { assert(false); }
  virtual void SampleTopK(int /*k*/, float /*temperature*/) { assert(false); }

  const SearchParams& params_;
};

struct Generator {
  Generator(Model& model, const SearchParams& search_params);

  bool IsDone() const;
  void ComputeLogits();
  void AppendNextToken_TopK_TopP(int top_k, float top_p, float temperature);
  void AppendNextToken_TopP(float p, float temperature) { AppendNextToken_TopK_TopP(0, p, temperature); }
  void AppendNextToken_TopK(int k, float temperature) { AppendNextToken_TopK_TopP(k, 1.0f, temperature); }
  void AppendNextToken_Top() { AppendNextToken_TopK_TopP(1, 1.0f, 0.0f); }
  void AppendNextToken();

  RoamingArray<int32_t> GetSequence(int index) { return search_->GetSequence(index); }

  Model& model_;
  std::unique_ptr<State> state_;
  std::unique_ptr<Search> search_;
  bool computed_logits_{}; // Set to true in ComputeLogits() and false after appending a token to ensure a 1 to 1 call ratio
};

std::unique_ptr<Model> CreateModel(OrtEnv& ort_env, const char* config_path, const ProviderOptions* provider_options = nullptr);
std::unique_ptr<Generator> CreateGenerator(Model& model, const SearchParams& search_params);
std::vector<int32_t> Generate(Model& model, const SearchParams& params); // Uses CreateGenerator and a simple loop to return the entire sequence

float Float16ToFloat32(uint16_t v);  // v is a IEEE 752-2008 binary16 format, 1 sign bit, 5 bit exponent, 10 bit fraction
void top_k_indices(std::span<int32_t> top_k, std::span<const float> inputs);

}  // namespace Generators
