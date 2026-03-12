// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "model.h"

namespace Generators {

// Manages recurrent state tensors (conv_state + recurrent_state) for hybrid models.
// Auto-discovers recurrent layers by probing session inputs.
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

  // Precomputed for allocation and RewindTo
  size_t conv_bytes_{};
  size_t recurrent_bytes_{};
  size_t recurrent_offset_{};  // Aligned offset of recurrent tensor within each layer block
  size_t per_layer_stride_{};  // Aligned stride per layer

  // Self-owned contiguous memory blocks (not from ORT arena)
  std::unique_ptr<OrtMemoryInfo> cpu_mem_info_;
  std::unique_ptr<uint8_t[]> past_block_;
  std::unique_ptr<uint8_t[]> present_block_;
};

// Factory: returns nullptr if no recurrent layers are found in the session.
std::unique_ptr<RecurrentState> CreateRecurrentState(State& state);

}  // namespace Generators
