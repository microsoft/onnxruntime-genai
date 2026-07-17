// Copyright (c) Intel® Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "multi_modal.h"

namespace Generators {

// Model for OpenVINO-partitioned VLMs. These models use a text-only embedding graph and
// merge image features on the host, so they delegate all vision work to a per-family
// OpenVINOVisionMerger (see openvino_vision_merger.h).
struct OpenVINOMultiModalModel : MultiModalLanguageModel {
  OpenVINOMultiModalModel(std::unique_ptr<Config> config, OrtEnv& ort_env);

  OpenVINOMultiModalModel(const OpenVINOMultiModalModel&) = delete;
  OpenVINOMultiModalModel& operator=(const OpenVINOMultiModalModel&) = delete;

  OrtEnv& ort_env_;

  // Session options for the host-side OpenVINO vision sub-model(s), built from the config's vision
  // session options. Owned here so the vision merger can create its sessions.
  std::unique_ptr<OrtSessionOptions> vision_session_options_ov_;

  // Single-session vision model. Null for families that use a multi-stage vision pipeline
  // (config.model.vision.pipeline), which the merger builds instead. Registered in session_info_ so the
  // image processor can resolve the pixel_values input type.
  std::unique_ptr<OrtSession> ov_vision_session_;
};

}  // namespace Generators
