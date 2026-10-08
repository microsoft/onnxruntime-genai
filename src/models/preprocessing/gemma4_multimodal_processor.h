// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#pragma once

#include "models/preprocessing/processor.h"

namespace Generators {

std::unique_ptr<OrtValue> ConvertAndResizeGemma4PositionIds(const int64_t* data, std::span<const int64_t> shape,
                                                            int64_t target_patches, ONNXTensorElementDataType target_type,
                                                            Ort::Allocator& allocator);

struct Gemma4MultiModalProcessor : Processor {
  Gemma4MultiModalProcessor(Config& config, const SessionInfo& session_info);

  virtual std::unique_ptr<NamedTensors> Process(const Tokenizer& tokenizer, const Payload& payload) const override;

 private:
  ort_extensions::OrtxObjectPtr<OrtxProcessor> image_processor_;
  ort_extensions::OrtxObjectPtr<OrtxFeatureExtractor> audio_processor_;

  ONNXTensorElementDataType pixel_values_type_;
  ONNXTensorElementDataType pixel_position_ids_type_{ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64};
  ONNXTensorElementDataType audio_features_type_;

  int64_t vision_fixed_num_patches_{-1};
  bool has_speech_{false};
  size_t vision_soft_tokens_per_image_{260};
};

}  // namespace Generators
