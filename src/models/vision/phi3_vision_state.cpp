// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "models/vision/phi3_vision_state.h"

#include <numeric>

namespace Generators {

int64_t Phi3VisionState::GetImageFeatureBatchSize(const std::vector<ExtraInput>& extra_inputs) const {
  for (const auto& input : extra_inputs) {
    if (input.name == Config::Defaults::PixelValuesName) {
      assert(input.tensor->ort_tensor_);
      const auto shape = input.tensor->ort_tensor_->GetTensorTypeAndShapeInfo()->GetShape();
      return shape.size() >= 3 ? shape.front() : 0;
    }
  }
  return 0;
}

int64_t Phi3VisionState::GetNumImageTokens(const std::vector<ExtraInput>& extra_inputs) const {
  for (const auto& input : extra_inputs) {
    if (input.name == Config::Defaults::NumImageTokens) {
      assert(input.tensor->ort_tensor_);
      const auto info = input.tensor->ort_tensor_->GetTensorTypeAndShapeInfo();
      const int64_t* data = input.tensor->ort_tensor_->GetTensorData<int64_t>();
      return std::accumulate(data, data + info->GetElementCount(), 0LL);
    }
  }
  return 0;
}

}  // namespace Generators
