// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#pragma once

#include "processor.h"

namespace Generators {

struct Config;
struct SessionInfo;

struct PhiMultiModalProcessor : Processor {
  PhiMultiModalProcessor(Config& config, const SessionInfo& session_info);

  virtual std::unique_ptr<NamedTensors> Process(const Tokenizer& tokenizer, const Payload& payload) const override;

 private:
  ort_extensions::OrtxObjectPtr<OrtxProcessor> image_processor_;
  ort_extensions::OrtxObjectPtr<OrtxFeatureExtractor> audio_processor_;

  ONNXTensorElementDataType pixel_values_type_;
  ONNXTensorElementDataType attention_mask_type_;
  ONNXTensorElementDataType audio_features_type_;
  ONNXTensorElementDataType audio_sizes_type_;
};

}  // namespace Generators
