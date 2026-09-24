// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "attention_mask.h"

namespace Generators {

struct StaticAttentionMask : AttentionMask {
  StaticAttentionMask(const Model& model, State& state, ONNXTensorElementDataType type, int mask_capacity);

  void Initialize(std::unique_ptr<OrtValue> cpu_attention_mask) override;
  void Update(int total_length, int new_length) override;
  void RewindTo(size_t index) override;

 private:
  template <typename T>
  void CopyInitialMask(OrtValue& cpu_attention_mask);

  const int mask_capacity_;
};

}  // namespace Generators
