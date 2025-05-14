// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#pragma once

namespace Generators {

// Base class for shared functionality
struct BaseLogits {
  BaseLogits(State& state);
  virtual ~BaseLogits() = default;

  void Add();  // Shared implementation for Add

  // Virtual functions for differences in behavior
  virtual DeviceSpan<float> Get() = 0;
  virtual void Update(const DeviceSpan<int32_t>& next_tokens, size_t new_kv_length) = 0;

 protected:
  State& state_;
  const Model& model_{state_.model_};
  size_t output_index_{~0U};

  ONNXTensorElementDataType type_;
  std::unique_ptr<Tensor> output_raw_;
  std::unique_ptr<OrtValue> output_last_tokens_;
  std::unique_ptr<OrtValue> logits_of_last_token_fp32_;

  std::vector<int> input_sequence_lengths;
  DeviceSpan<float> logits_;
  bool trimmed_prefill_logits_ = false;
};

// Logits class
struct Logits : BaseLogits {
  Logits(State& state);

  DeviceSpan<float> Get() override;  // Specific implementation for Logits
  void Update(const DeviceSpan<int32_t>& next_tokens, size_t new_kv_length) override;  // Specific implementation for Logits

 private:
  std::array<int64_t, 3> shape_;
};

// RNNLogits class
struct RNNLogits : BaseLogits {
  RNNLogits(State& state);

  DeviceSpan<float> Get() override;  // Specific implementation for RNNLogits
  void Update(const DeviceSpan<int32_t>& next_tokens, size_t new_kv_length) override;  // Specific implementation for RNNLogits

 private:
  std::array<int64_t, 2> shape_;
};

}  // namespace Generators