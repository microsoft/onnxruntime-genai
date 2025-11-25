// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "model.h"
#include "extra_inputs.h"

namespace Generators {

struct VisionPipelineModel : Model {
  VisionPipelineModel(std::unique_ptr<Config> config, OrtEnv& ort_env);

  VisionPipelineModel(const VisionPipelineModel&) = delete;
  VisionPipelineModel& operator=(const VisionPipelineModel&) = delete;

  std::unique_ptr<State> CreateState(DeviceSpan<int32_t> sequence_lengths, const GeneratorParams& params) const override;

  std::vector<std::unique_ptr<OrtSession>> sessions_;
  std::vector<std::unique_ptr<OrtSessionOptions>> session_options_;
  OrtEnv& ort_env_;
  
  // Window indexing support
  std::vector<int64_t> window_indices_;
  int spatial_merge_size_{0};
};

struct VisionPipelineState : State {
  VisionPipelineState(const VisionPipelineModel& model, const GeneratorParams& params);

  VisionPipelineState(const VisionPipelineState&) = delete;
  VisionPipelineState& operator=(const VisionPipelineState&) = delete;

  void SetExtraInputs(const std::vector<ExtraInput>& extra_inputs, int64_t num_images, int64_t num_image_tokens);

  DeviceSpan<float> Run(int current_length, DeviceSpan<int32_t>& next_tokens,
                        DeviceSpan<int32_t> next_indices) override;

  OrtValue* GetOutput(const char* name) override;

  int64_t GetNumImageTokens() const { return num_image_tokens_; }
  int64_t GetNumImages() const { return num_images_; }

 private:
  OrtValue* ApplyWindowIndexingTransform(OrtValue* input_tensor, bool restore_order);

  const VisionPipelineModel& model_;
  int64_t num_image_tokens_{0};
  int64_t num_images_{0};
  
  ExtraInputs extra_inputs_{*this};
  
  // Stores intermediate outputs between pipeline stages
  std::unordered_map<std::string, std::unique_ptr<OrtValue>> ortvalue_store_;
  
  // For storing transformed tensors during window indexing
  std::unique_ptr<OrtValue> intermediate_tensor_;
  
  // Final output (image_features)
  std::unique_ptr<OrtValue> image_features_;
};

}  // namespace Generators
