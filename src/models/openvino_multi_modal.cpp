// Copyright (c) Intel® Corporation. All rights reserved.
// Licensed under the MIT License.

#include "generator/generators.h"
#include "openvino_multi_modal.h"

namespace Generators {

const Config::SessionOptions& GetOpenVINOVisionStageSessionOptions(const Config& config,
                                                                   const Config::Model::Vision::PipelineModel& stage) {
  if (stage.session_options.has_value()) return *stage.session_options;
  if (config.model.vision.session_options.has_value()) return *config.model.vision.session_options;
  return config.model.decoder.session_options;
}

OpenVINOMultiModalModel::OpenVINOMultiModalModel(std::unique_ptr<Config> config, OrtEnv& ort_env)
    // vision=false: the merger owns host-side vision, and this makes the embedding text-only.
    : MultiModalLanguageModel(std::move(config), ort_env, /*vision=*/false, /*speech=*/false) {
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
  } else {
    for (const auto& stage : config_->model.vision.pipeline) {
      if (ov_vision_pipeline_sessions_.find(stage.model_id) != ov_vision_pipeline_sessions_.end()) {
        throw std::runtime_error("Duplicate OpenVINO vision pipeline stage: " + stage.model_id);
      }
      OrtSessionOptions* stage_options = vision_session_options_ov_.get();
      if (stage.session_options.has_value() || stage.run_on_cpu) {
        auto& options = ov_vision_stage_session_options_[stage.model_id];
        options = OrtSessionOptions::Create();
        // Non-primary, like decoder pipeline stages: Config fills `providers` only for component options, and
        // non-primary options take them from provider_options. run_on_cpu appends no provider (CPU EP).
        CreateSessionOptionsFromConfig(GetOpenVINOVisionStageSessionOptions(*config_, stage), *options,
                                       /*is_primary_session_options=*/false, /*disable_graph_capture=*/true,
                                       /*cloned_from_parent=*/false, /*append_providers=*/!stage.run_on_cpu);
        stage_options = options.get();
      }
      ov_vision_pipeline_sessions_.emplace(stage.model_id, CreateSession(ort_env, stage.filename, stage_options));
    }
  }
}

OrtSession& OpenVINOMultiModalModel::GetVisionPipelineSession(const std::string& model_id) const {
  auto it = ov_vision_pipeline_sessions_.find(model_id);
  if (it == ov_vision_pipeline_sessions_.end()) {
    throw std::runtime_error("OpenVINO vision pipeline stage not found: " + model_id);
  }
  return *it->second;
}

}  // namespace Generators
