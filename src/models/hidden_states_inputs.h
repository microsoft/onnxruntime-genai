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

// Manages a `hidden_states` *output* ([batch, sequence_length, hidden_size]) as a first-class
// model output with a pre-allocated buffer. A plain ORT-allocated extra output cannot be used
// here because CUDA-graph capture binds outputs to static buffers; owning the tensor (and making
// it static when graph capture is active) lets the hidden state be exposed under CUDA graph.
// Created only for models whose graph emits a hidden_states output (e.g. exported with
// include_hidden_states, to feed the MTP head).
struct HiddenStatesOutputs {
  HiddenStatesOutputs(State& state);
  HiddenStatesOutputs(const HiddenStatesOutputs&) = delete;
  HiddenStatesOutputs& operator=(const HiddenStatesOutputs&) = delete;

  void Add();                       // Register the hidden_states output (called once at init).
  void Update(int sequence_length);  // Resize the output buffer to match the step's sequence length.

 private:
  State& state_;
  const Model& model_{state_.model_};
  size_t output_index_{~0U};

  std::array<int64_t, 3> shape_{};
  ONNXTensorElementDataType type_;
  std::unique_ptr<Tensor> value_;
};

}  // namespace Generators
