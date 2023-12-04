// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#pragma once

#include <algorithm>
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

namespace Generators {
struct Model;
using ScoreType = float;

enum struct DeviceType {
  Auto,
  CPU,
  CUDA,
};

struct OrtCPUProviderOptions { }; // Stub so that ProviderOptions isn't empty without cuda

using ProviderOptions=std::variant<
OrtCPUProviderOptions
#if USE_CUDA
,OrtCUDAProviderOptions
#endif
>;

ProviderOptions GetDefaultProviderOptions(DeviceType device_type);

struct Config {
  Config()=default;
  Config(const std::filesystem::path& path);

  std::filesystem::path config_path; // Path of the config directory

  // Sequence Generation
  int min_length {0};
  int max_length{20};
  bool early_stopping {false}; //  Whether to stop the beam search when at least num_beams sentences are finished per batch or not.
  int num_beams{1}; //  1 means no beam search.
  float temperature{1.0f};
  int top_k {50}; // Number of highest probability vocabulary tokens to keep for top-k-filtering that will be used by default in the generate method of the model.
  float top_p {1.0f}; // If set to float < 1, only the most probable tokens with probabilities that add up to top_p or higher are kept for generation.
  float repetition_penalty {1.0f}; // 1.0 means no penalty.
  float length_penalty {1.0f}; // Exponential penalty to the length that is used with beam-based generation. length_penalty > 0.0 promotes longer sequences, while length_penalty < 0.0 encourages shorter sequences.
  
  // Tokenizer Parameters
  std::string tokenizer_class;
  std::string prefix;
  int pad_token_id{}; // The id of the padding token.
  int eos_token_id{}; // The id of the end-of-stream token.
  int bos_token_id{}; // The id of the beginning-of-stream token.
  int decoder_start_token_id {}; //  If an encoder-decoder model starts decoding with a different token than bos, the id of that token.
  int sep_token_id{}; // The id of the separation token.

  // Model Class Attributes
  std::string model_decoder;
  std::string model_type;
  int vocab_size{};
  int hidden_size {};
  int n_embed{768};  // GPT version of hidden_size?
  int num_attention_heads{};
  int n_head{12}; // GPT version of num_attention_heads?
  int num_hidden_layers {};
  int n_layer {12}; // GPT version of num_hidden_layers?
};

struct SearchParams
{
  SearchParams()=default;
  SearchParams(const Model& model);

  // Values copied from config
  int pad_token_id{};
  int eos_token_id{};
  int vocab_size{};
  int max_length {};
  float length_penalty{};
  bool early_stopping {};

  int batch_size{};
  int sequence_length{};
  int num_beams{1};
  int BatchBeamSize() const { return num_beams * batch_size; }

  std::span<const int32_t> input_ids;  // Array of [batchsize][sequence_length]
#if USE_CUDA
  cudaStream_t cuda_stream{};
#endif
};

float Float16ToFloat32(uint16_t v);  // v is a IEEE 752-2008 binary16 format, 1 sign bit, 5 bit exponent, 10 bit fraction
void ParseConfig(const std::filesystem::path& filename, Config& params);
void top_k_indices(std::span<int32_t> top_k, std::span<const ScoreType> inputs);

}
