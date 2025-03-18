// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "../generators.h"
#include "model.h"

#include <regex>

namespace Generators {

namespace {

}  // namespace

GemmaImageProcessor::GemmaImageProcessor(Config& config, const SessionInfo& session_info)
    : pixel_values_type_{session_info.GetInputDataType(config.model.vision.inputs.pixel_values)},
      attention_mask_type_{session_info.GetInputDataType(config.model.vision.inputs.attention_mask)} {
  const auto image_processor_config = (config.config_path / fs::path(config.model.vision.config_filename)).string();
  CheckResult(OrtxCreateProcessor(image_processor_.ToBeAssigned(), image_processor_config.c_str()));

  config.AddMapping(std::string(Config::Defaults::InputIdsName), config.model.embedding.inputs.input_ids);

  config.AddMapping(std::string(Config::Defaults::PixelValuesName), config.model.vision.inputs.pixel_values);
  config.AddMapping(std::string(Config::Defaults::AttentionMaskName), config.model.vision.inputs.attention_mask);
  config.AddMapping(std::string(Config::Defaults::ImageSizesName), config.model.vision.inputs.image_sizes);
}

std::unique_ptr<NamedTensors> GemmaImageProcessor::Process(const Tokenizer& tokenizer, const Payload& payload) const {
  return nullptr;
}

}  // namespace Generators
