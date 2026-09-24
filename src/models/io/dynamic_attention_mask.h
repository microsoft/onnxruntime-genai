// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "attention_mask.h"

namespace Generators {

struct DynamicAttentionMask : AttentionMask {
  using AttentionMask::AttentionMask;

  void Initialize(std::unique_ptr<OrtValue> cpu_attention_mask) override;
  void Update(int total_length, int new_length) override;
  void RewindTo(size_t index) override;

 private:
  Tensor next_mask_{model_.p_device_inputs_, type_};
};

}  // namespace Generators
