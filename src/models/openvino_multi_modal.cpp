// Copyright (c) Intel® Corporation. All rights reserved.
// Licensed under the MIT License.

#include "generator/generators.h"
#include "openvino_multi_modal.h"

namespace Generators {

OpenVINOMultiModalModel::OpenVINOMultiModalModel(std::unique_ptr<Config> config, OrtEnv& ort_env)
    // vision=false: the merger owns host-side vision, and this makes the embedding text-only.
    : MultiModalLanguageModel(std::move(config), ort_env, /*vision=*/false, /*speech=*/false),
      ort_env_{ort_env} {
  // Build session options for the host-side vision sub-model(s). Fall back to the decoder session
  // options when the config does not specify vision-specific ones, matching MultiModalLanguageModel.
  vision_session_options_ov_ = OrtSessionOptions::Create();
  CreateSessionOptionsFromConfig(
      config_->model.vision.session_options.has_value() ? config_->model.vision.session_options.value()
                                                        : config_->model.decoder.session_options,
      *vision_session_options_ov_, /*is_primary_session_options=*/true, /*disable_graph_capture=*/true);

  if (config_->model.vision.pipeline.empty()) {
    ov_vision_session_ = CreateSession(ort_env, config_->model.vision.filename, vision_session_options_ov_.get());
    session_info_.Add(*ov_vision_session_);
  }
}

}  // namespace Generators
