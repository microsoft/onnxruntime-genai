// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#pragma once

namespace Generators {

struct Logits {
  Logits(State& state);

  // Register input_ids as ORT session input.
  void Add();
  DeviceSpan<float> GetAll();
  // For first iteration, find last token of each beam and store it in output_last_tokens_.
  DeviceSpan<float> Get();

  // Resize logits to [bz, token_count, vocab_size] if necessary.
  void Update(const DeviceSpan<int32_t>& next_tokens, size_t new_kv_length);

 private:
  // dropped_rows is the count of input positions the model did not emit logits for.
  void UpdateInputSequenceLengths(const DeviceSpan<int32_t>& next_tokens, size_t input_kv_length, size_t dropped_rows);

  State& state_;
  const Model& model_{state_.model_};
  size_t output_index_{~0U};

  std::array<int64_t, 3> shape_{};
  ONNXTensorElementDataType type_;
  size_t max_output_sequence_length_{};

  // Tensor to keep the logits of the last tokens. It is used in the 2 cases below. Otherwhise, it is not used.
  // 1. prompt: store the last tokens logits from output_raw_
  // 2. token gen: store the converted fp32 logits if output_raw_ is fp16.
  std::unique_ptr<OrtValue> output_last_tokens_;
  std::unique_ptr<OrtValue> logits_of_last_token_fp32_;
  std::unique_ptr<OrtValue> all_logits_fp32_;

  std::unique_ptr<Tensor> output_raw_;  // Raw logits output from model

  std::vector<int> input_sequence_lengths;
  // OrtValue wrapped in a DeviceMemory object to make it universal
  DeviceSpan<float> logits_;
  DeviceSpan<float> all_logits_;

  // Set to true when prefill will generate the already 'trimmed' logits required for sampling.
  bool trimmed_prefill_logits_ = false;
};

}  // namespace Generators
