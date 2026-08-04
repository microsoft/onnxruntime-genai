// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "model.h"

namespace Generators {

// Manages recurrent state tensors for hybrid and compressed-attention models.
// State names can be declared in config; legacy recurrent layers are auto-discovered.
struct RecurrentState {
  RecurrentState(State& state);
  ~RecurrentState();

  void Add();
  void Update();
  void RewindTo(size_t index);

  bool IsEmpty() const { return input_name_strings_.empty(); }

 private:
  void ZeroStates(std::vector<std::unique_ptr<OrtValue>>& states);

  State& state_;
  const Model& model_{state_.model_};

  std::vector<std::unique_ptr<OrtValue>> pasts_;
  std::vector<std::unique_ptr<OrtValue>> presents_;
  bool dynamic_states_{false};

  // Mirrors past_present_share_buffer config: true means inputs alias outputs (same allocation,
  // stable handles for graph capture). False uses separate past/present buffers with per-step swap.
  bool share_buffers_{false};
  size_t input_index_{~0U};
  size_t output_index_{~0U};

  // Kept alive for state_ const char* pointers
  std::vector<std::string> input_name_strings_;
  std::vector<std::string> output_name_strings_;

  std::vector<ONNXTensorElementDataType> types_;
  std::vector<std::vector<int64_t>> shapes_;
};

// Factory: returns nullptr if no recurrent layers are found in the session.
std::unique_ptr<RecurrentState> CreateRecurrentState(State& state);

}  // namespace Generators
