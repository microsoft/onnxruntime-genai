// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "models/pipeline/model_pipeline.h"

#include "models/decoder_only_pipeline.h"
#include "models/model_type.h"
#include "models/pipeline/qwen_vl_pipeline.h"

namespace Generators {

std::shared_ptr<Model> CreatePipelineModel(std::unique_ptr<Config>& config, OrtEnv& ort_env) {
  const auto& model_type = config->model.type;
  if (!config->model.decoder.pipeline.empty() &&
      (model_type == "fara" || model_type == "qwen2_5_vl" || model_type == "qwen3_vl")) {
    return std::make_shared<Qwen2_5_VL_PipelineModel>(std::move(config), ort_env);
  }
  if (ModelType::IsPipe(model_type)) {
    return std::make_shared<DecoderOnlyPipelineModel>(std::move(config), ort_env);
  }
  return {};
}

}  // namespace Generators
