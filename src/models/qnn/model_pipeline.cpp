// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "models/qnn/model_pipeline.h"

#include "models/qnn/decoder_only_pipeline.h"
#include "models/model_type.h"
#include "models/qnn/qwen_vl_pipeline.h"

namespace Generators {

std::shared_ptr<Model> CreatePipelineModel(std::unique_ptr<Config>& config, OrtEnv& ort_env) {
  const auto& model_type = config->model.type;
  if (!config->model.decoder.pipeline.empty() &&
      (ModelType::IsVLM(model_type) || ModelType::IsMMM(model_type))) {
    if (!ModelType::IsPipe(model_type)) {
      throw std::runtime_error("Pipelined decoder is not supported for model type '" + model_type + "'");
    }
    return std::make_shared<Qwen2_5_VL_PipelineModel>(std::move(config), ort_env);
  }
  if (ModelType::IsPipe(model_type) &&
      !ModelType::IsVLM(model_type) &&
      !ModelType::IsMMM(model_type)) {
    return std::make_shared<DecoderOnlyPipelineModel>(std::move(config), ort_env);
  }
  return {};
}

}  // namespace Generators
