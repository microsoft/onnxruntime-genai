// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#pragma once

#include "processor.h"

namespace Generators {

struct Gemma4MultiModalProcessor : Processor {
  Gemma4MultiModalProcessor(Config& config, const SessionInfo& session_info);

  virtual std::unique_ptr<NamedTensors> Process(const Tokenizer& tokenizer, const Payload& payload) const override;

 private:
  ort_extensions::OrtxObjectPtr<OrtxProcessor> image_processor_;
  ort_extensions::OrtxObjectPtr<OrtxFeatureExtractor> audio_processor_;

  ONNXTensorElementDataType pixel_values_type_;
  ONNXTensorElementDataType pixel_position_ids_type_{ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64};
  ONNXTensorElementDataType audio_features_type_;

  bool has_speech_{false};
  bool unified_{false};  // gemma-4-12B encoder-free "unified" variant
  size_t vision_soft_tokens_per_image_{260};
};

}  // namespace Generators
