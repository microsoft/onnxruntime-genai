// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <optional>

#include "position_inputs.h"

namespace Generators {

struct AttentionMask {
  AttentionMask(const Model& model, State& state, ONNXTensorElementDataType type);
  virtual ~AttentionMask() = default;

  virtual void Initialize(std::unique_ptr<OrtValue> cpu_attention_mask) = 0;
  virtual void Update(int total_length, int new_length) = 0;
  virtual void RewindTo(size_t index) = 0;

  OrtValue* GetOrtTensor() { return mask_.GetOrtTensor(); }
  const std::array<int64_t, 2>& GetShape() const { return shape_; }

 protected:
  void UpdateOnDevice(Tensor* next_mask, int total_length, int new_length, int mask_capacity);

  const Model& model_;
  State& state_;
  ONNXTensorElementDataType type_;
  std::array<int64_t, 2> shape_{};
  Tensor mask_;
};

// An explicit capacity selects a static mask independently of generation max_length.
std::unique_ptr<AttentionMask> CreateAttentionMask(const Model& model, State& state, ONNXTensorElementDataType type,
                                                   std::optional<int> static_mask_capacity = std::nullopt);

}  // namespace Generators
