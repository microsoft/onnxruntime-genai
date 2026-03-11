// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "model.h"

namespace Generators {

// Manages recurrent (linear attention) state tensors for hybrid models like Qwen3.5.
// Auto-discovers which layers have recurrent states by probing session inputs.
// Each recurrent layer has:
//   - conv_state: causal convolution state (fixed shape, e.g. [batch, conv_dim, kernel_size-1])
//   - recurrent_state: hidden state (fixed shape, e.g. [batch, num_heads, head_dim, head_dim])
// Both are carried across generation steps via pointer swaps (zero-copy).
struct RecurrentState {
  RecurrentState(State& state);

  void Add();             // Add tensors to state inputs/outputs
  void Update();          // Swap present->past after each step (pointer swap, no copy)
  void RewindTo(size_t index);  // Reset states to zeros

  bool IsEmpty() const { return layer_indices_.empty(); }

 private:
  State& state_;
  const Model& model_{state_.model_};

  std::vector<int> layer_indices_;  // Actual layer indices with recurrent states

  // Per-layer tensors: interleaved as [conv_0, recurrent_0, conv_1, recurrent_1, ...]
  std::vector<std::unique_ptr<OrtValue>> pasts_;
  std::vector<std::unique_ptr<OrtValue>> presents_;

  // Name strings (kept alive for state_.input_names_/output_names_ const char* pointers)
  std::vector<std::string> input_name_strings_;
  std::vector<std::string> output_name_strings_;

  // Indices into state_.inputs_/outputs_ for efficient pointer updates
  size_t input_index_{~0U};
  size_t output_index_{~0U};

  // Shapes and types discovered from session info
  ONNXTensorElementDataType conv_type_{};
  ONNXTensorElementDataType recurrent_type_{};

  // Shapes cached for tensor creation
  std::vector<int64_t> conv_shape_;
  std::vector<int64_t> recurrent_shape_;

  // Precomputed byte counts for fast zeroing in RewindTo
  size_t conv_bytes_{};       // bytes per single conv_state tensor
  size_t recurrent_bytes_{};  // bytes per single recurrent_state tensor
  size_t per_layer_bytes_{};  // conv_bytes_ + recurrent_bytes_

  // Two contiguous memory blocks: one for past, one for present.
  // Each block is laid out as: [conv_0 | recurrent_0 | conv_1 | recurrent_1 | ...]
  // Using 2 bulk allocations instead of 72 individual ones for better cache locality
  // and fewer heap operations. value-initialized to zero by std::make_unique<float[]>.
  std::unique_ptr<OrtMemoryInfo> cpu_mem_info_;
  std::unique_ptr<float[]> past_block_;     // past states — zero-initialized
  std::unique_ptr<float[]> present_block_;  // present states — output buffer
};

// Factory: returns nullptr if no recurrent layers are found in the session.
std::unique_ptr<RecurrentState> CreateRecurrentState(State& state);

}  // namespace Generators
