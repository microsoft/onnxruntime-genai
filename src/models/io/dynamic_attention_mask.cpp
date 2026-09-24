// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "dynamic_attention_mask.h"

namespace Generators {

void DynamicAttentionMask::Initialize(std::unique_ptr<OrtValue> cpu_attention_mask) {
  const auto input_shape = cpu_attention_mask->GetTensorTypeAndShapeInfo()->GetShape();
  shape_ = {input_shape[0] * state_.params_->search.num_beams, input_shape[1]};
  mask_.ort_tensor_ = model_.ExpandInputs(cpu_attention_mask, state_.params_->search.num_beams);
}

void DynamicAttentionMask::Update(int total_length, int new_length) {
  shape_[1] = total_length;
  next_mask_.CreateTensor(shape_);
  UpdateOnDevice(&next_mask_, total_length, new_length, state_.params_->search.max_length);
  mask_.ort_tensor_ = std::move(next_mask_.ort_tensor_);
}

void DynamicAttentionMask::RewindTo(size_t index) {
  // RewindTo only permits a single sequence; the next update fills its mask with ones.
  shape_[1] = static_cast<int64_t>(index);
}

}  // namespace Generators
