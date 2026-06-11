// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#pragma once

namespace Generators {

// Feeds a `hidden_states` input ([batch, sequence_length, hidden_size]) to a model that
// consumes the main model's last hidden state (e.g. the Qwen3.6 MTP self-speculative head).
//
// Unlike a one-shot extra input, the value changes every step and its sequence length varies
// (prompt length, then 1 or 2 tokens per speculative step), so this feeder owns a resizable
// device tensor and is refreshed from a caller-provided OrtValue before each Run.
struct HiddenStatesInputs {
  HiddenStatesInputs(State& state);
  HiddenStatesInputs(const HiddenStatesInputs&) = delete;
  HiddenStatesInputs& operator=(const HiddenStatesInputs&) = delete;

  // Register the hidden_states input as an ORT session input (called once at init).
  void Add();

  // Stage the hidden-state values to feed on the next Run. `source` is [batch, seq, hidden]
  // of the model's io dtype (typically provided on CPU from the caller); it is copied into
  // the device tensor by Update().
  void SetValue(OrtValue* source) { pending_source_ = source; }

  // Resize the device tensor to match `sequence_length` and copy the pending source into it.
  void Update(int sequence_length);

 private:
  State& state_;
  const Model& model_{state_.model_};
  size_t input_index_{~0U};

  std::array<int64_t, 3> shape_{};
  ONNXTensorElementDataType type_;
  std::unique_ptr<Tensor> value_;
  OrtValue* pending_source_{};
};

}  // namespace Generators
