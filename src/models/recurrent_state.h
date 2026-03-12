// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "model.h"

namespace Generators {

// Manages recurrent (linear attention) state tensors for hybrid models (e.g., Qwen3.5 Gated DeltaNet).
// Auto-discovers which layers have recurrent states by probing session inputs.
// Each recurrent layer carries two fixed-size states across generation steps:
//   - conv_state: causal convolution buffer
//   - recurrent_state: compressed sequence history
struct RecurrentState {
  RecurrentState(State& state);

  void Add();
  void Update();
  void RewindTo(size_t index);

  bool IsEmpty() const { return layer_indices_.empty(); }

 private:
  State& state_;
  const Model& model_{state_.model_};

  std::vector<int> layer_indices_;

  // Interleaved as [conv_0, recurrent_0, conv_1, recurrent_1, ...]
  std::vector<std::unique_ptr<OrtValue>> pasts_;
  std::vector<std::unique_ptr<OrtValue>> presents_;

  // Kept alive for state_ const char* pointers
  std::vector<std::string> input_name_strings_;
  std::vector<std::string> output_name_strings_;

  size_t input_index_{~0U};
  size_t output_index_{~0U};

  ONNXTensorElementDataType conv_type_{};
  ONNXTensorElementDataType recurrent_type_{};

  std::vector<int64_t> conv_shape_;
  std::vector<int64_t> recurrent_shape_;

  // Precomputed for RewindTo
  size_t conv_bytes_{};
  size_t recurrent_bytes_{};
  size_t per_layer_bytes_{};

  // Two contiguous memory blocks laid out as [conv_0 | recurrent_0 | conv_1 | recurrent_1 | ...]
  // Self-owned (not from ORT arena) to guarantee zero-initialization.
  std::unique_ptr<OrtMemoryInfo> cpu_mem_info_;
  std::unique_ptr<float[]> past_block_;
  std::unique_ptr<float[]> present_block_;
};

// Factory: returns nullptr if no recurrent layers are found in the session.
std::unique_ptr<RecurrentState> CreateRecurrentState(State& state);

}  // namespace Generators
