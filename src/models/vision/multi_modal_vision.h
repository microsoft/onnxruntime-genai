// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <cstdint>
#include <vector>

#include "models/model.h"
#include "models/io/multi_modal_features.h"
#include "models/io/extra_inputs.h"

namespace Generators {

struct MultiModalLanguageModel;
struct MultiModalPipelineState;

// Base VisionState: runs vision.onnx with a single State::Run() call.
// Works for models whose vision encoder accepts batched input (Phi, Gemma).
struct VisionState : State {
  VisionState(const MultiModalLanguageModel& model, const GeneratorParams& params);
  VisionState(const VisionState&) = delete;
  VisionState& operator=(const VisionState&) = delete;

  virtual void SetExtraInputs(const std::vector<ExtraInput>& extra_inputs, const int64_t num_images, const int64_t num_image_tokens);
  virtual int64_t GetImageFeatureBatchSize(const std::vector<ExtraInput>& extra_inputs) const;
  virtual int64_t GetNumImageTokens(const std::vector<ExtraInput>& extra_inputs) const;
  DeviceSpan<float> Run(int current_length, DeviceSpan<int32_t>& next_tokens, DeviceSpan<int32_t> next_indices = {}) override;

 protected:
  friend struct MultiModalPipelineState;

  const MultiModalLanguageModel& model_;
  int64_t num_image_tokens_;
  int64_t num_images_{};
  ExtraInputs extra_inputs_{*this};  // Model inputs
  std::unique_ptr<MultiModalFeatures> image_features_;
};

// Factory: pick the right VisionState subclass based on model type.
std::unique_ptr<VisionState> CreateVisionState(const MultiModalLanguageModel& model, const GeneratorParams& params);

}  // namespace Generators
