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

  // Snapshot/restore the recurrent (conv + recurrent) state buffers. Required for
  // speculative decoding (e.g. MTP): the recurrent state cannot be partially rewound
  // (unlike the attention KV cache), so a draft/verify step snapshots the state before
  // a speculative forward and restores it if the draft is rejected. Restore copies back
  // in place so buffer addresses stay stable (required by CUDA-graph replay).
  void Snapshot();
  void RestoreSnapshot();

  bool IsEmpty() const { return layer_indices_.empty(); }

 private:
  void ZeroStates(std::vector<std::unique_ptr<OrtValue>>& states);
  void CopyStates(const std::vector<std::unique_ptr<OrtValue>>& src, std::vector<std::unique_ptr<OrtValue>>& dst);

  State& state_;
  const Model& model_{state_.model_};

  std::vector<int> layer_indices_;

  // Interleaved as [conv_0, recurrent_0, conv_1, recurrent_1, ...]
  std::vector<std::unique_ptr<OrtValue>> pasts_;
  std::vector<std::unique_ptr<OrtValue>> presents_;
  std::vector<std::unique_ptr<OrtValue>> snapshot_;  // Lazily-allocated copy of the live state for speculative rollback.
  bool snapshot_valid_{false};                       // Whether snapshot_ holds a valid captured state.

  // WebGPU cannot alias input/output buffers, so it uses separate past/present\n  // with swap. All other EPs share buffers for stable addresses.
  bool share_buffers_{false};
  size_t input_index_{~0U};
  size_t output_index_{~0U};

  // Kept alive for state_ const char* pointers
  std::vector<std::string> input_name_strings_;
  std::vector<std::string> output_name_strings_;

  ONNXTensorElementDataType conv_type_{};
  ONNXTensorElementDataType recurrent_type_{};

  std::vector<int64_t> conv_shape_;
  std::vector<int64_t> recurrent_shape_;
};

// Factory: returns nullptr if no recurrent layers are found in the session.
std::unique_ptr<RecurrentState> CreateRecurrentState(State& state);

}  // namespace Generators
