// Copyright (c) Intel® Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <memory>
#include <string>
#include <unordered_map>

#include "multi_modal.h"

namespace Generators {

// Model for OpenVINO-partitioned VLMs. These models use a text-only embedding graph and
// merge image features on the host, so they delegate all vision work to a per-family
// OpenVINOVisionMerger (see openvino_vision_merger.h).
struct OpenVINOMultiModalModel : MultiModalLanguageModel {
  OpenVINOMultiModalModel(std::unique_ptr<Config> config, OrtEnv& ort_env);

  OpenVINOMultiModalModel(const OpenVINOMultiModalModel&) = delete;
  OpenVINOMultiModalModel& operator=(const OpenVINOMultiModalModel&) = delete;

  OrtSession& GetVisionPipelineSession(const std::string& model_id) const;

  // Session options for the host-side OpenVINO vision sub-model(s).
  std::unique_ptr<OrtSessionOptions> vision_session_options_ov_;

  // Single-session vision model. Registered so the image processor can resolve pixel_values type.
  std::unique_ptr<OrtSession> ov_vision_session_;

  // Shared across generators and created through Model::CreateSession to support in-memory model data.
  std::unordered_map<std::string, std::unique_ptr<OrtSession>> ov_vision_pipeline_sessions_;
};

inline const OpenVINOMultiModalModel* AsOpenVINOMultiModalModel(const Model& model) {
  return dynamic_cast<const OpenVINOMultiModalModel*>(&model);
}

}  // namespace Generators
