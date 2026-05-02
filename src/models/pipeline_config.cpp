// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "../generators.h"
#include "pipeline_config.h"

namespace Generators {

PipelineConfigModel::PipelineConfigModel(
    std::unique_ptr<Config> config, OrtEnv& ort_env)
    : DecoderOnly_Model{std::move(config), ort_env} {
  // Delegates entirely to DecoderOnly_Model for decoder-only models.
  // Future: inspect config pipeline section to determine model category
  // and load additional sessions for multi-modal pipelines.
}

}  // namespace Generators
