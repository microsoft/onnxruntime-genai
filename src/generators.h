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

struct Search {
  virtual ~Search() = default;

  virtual RoamingArray<int32_t> GetNextTokens() = 0;
  virtual RoamingArray<int32_t> GetNextIndices() {
    throw std::runtime_error("GetNextIndices() can only be called for beam search, num_beams must be >1");
  }
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
  virtual void GetTopKSubset(int* tokens_out, int k) { assert(false); }
};

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

float Float16ToFloat32(uint16_t v);  // v is a IEEE 752-2008 binary16 format, 1 sign bit, 5 bit exponent, 10 bit fraction
void top_k_indices(std::span<int32_t> top_k, std::span<const float> inputs);

}  // namespace Generators
