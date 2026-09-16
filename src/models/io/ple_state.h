// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "models/model.h"

namespace Generators {

struct PleState {
  explicit PleState(State& state);

  void Add();
  void Update();
  void RewindTo(size_t index);

  bool IsEmpty() const { return layer_indices_.empty(); }
  int GraphCaptureVariant() const { return graph_buffer_variant_; }

 private:
  void InitializeStates(std::vector<std::unique_ptr<OrtValue>>& states);

  State& state_;
  const Model& model_{state_.model_};
  std::vector<int> layer_indices_;
  std::vector<std::unique_ptr<OrtValue>> pasts_;
  std::vector<std::unique_ptr<OrtValue>> presents_;
  std::vector<std::string> input_name_strings_;
  std::vector<std::string> output_name_strings_;
  std::vector<int64_t> token_shape_;
  std::vector<int64_t> conv_shape_;
  ONNXTensorElementDataType conv_type_{};
  int graph_buffer_variant_{};
  size_t input_index_{~0U};
  size_t output_index_{~0U};
};

std::unique_ptr<PleState> CreatePleState(State& state);

}  // namespace Generators