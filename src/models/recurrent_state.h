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
  void ZeroStates(std::vector<std::unique_ptr<OrtValue>>& states);

  State& state_;
  const Model& model_{state_.model_};

  std::vector<int> layer_indices_;

  // Interleaved as [conv_0, recurrent_0, conv_1, recurrent_1, ...]
  std::vector<std::unique_ptr<OrtValue>> pasts_;
  std::vector<std::unique_ptr<OrtValue>> presents_;

  // Cached byte spans for graph-capture copy path (avoids recomputing
  // tensor metadata on every decode step for fixed-shape tensors).
  std::vector<DeviceSpan<uint8_t>> past_byte_spans_;
  std::vector<DeviceSpan<uint8_t>> present_byte_spans_;

  // Kept alive for state_ const char* pointers
  std::vector<std::string> input_name_strings_;
  std::vector<std::string> output_name_strings_;

  size_t input_index_{~0U};
  size_t output_index_{~0U};

  ONNXTensorElementDataType conv_type_{};
  ONNXTensorElementDataType recurrent_type_{};

  std::vector<int64_t> conv_shape_;
  std::vector<int64_t> recurrent_shape_;
};

// Factory: returns nullptr if no recurrent layers are found in the session.
std::unique_ptr<RecurrentState> CreateRecurrentState(State& state);

}  // namespace Generators
