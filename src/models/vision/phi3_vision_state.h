// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "models/vision/multi_modal_vision.h"

namespace Generators {

struct Phi3VisionState : VisionState {
  using VisionState::VisionState;

  int64_t GetImageFeatureBatchSize(const std::vector<ExtraInput>& extra_inputs) const override;
  int64_t GetNumImageTokens(const std::vector<ExtraInput>& extra_inputs) const override;
};

}  // namespace Generators
