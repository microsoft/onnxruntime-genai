// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#pragma once

#include "processor.h"

namespace Generators {

struct Gemma3nMultiModalProcessor : Processor {
  Gemma3nMultiModalProcessor(Config& config, const SessionInfo& session_info);

  virtual std::unique_ptr<NamedTensors> Process(const Tokenizer& tokenizer, const Payload& payload) const override;

 private:
  ort_extensions::OrtxObjectPtr<OrtxProcessor> image_processor_;
  ort_extensions::OrtxObjectPtr<OrtxFeatureExtractor> audio_processor_;

  ONNXTensorElementDataType pixel_values_type_;
  // Only read when has_speech_ is set, but default it anyway: a processor built
  // for a vision-only package would otherwise carry an indeterminate value.
  ONNXTensorElementDataType audio_features_type_{ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT};

  bool has_speech_{false};

  // vision_soft_tokens_per_image / audio_soft_tokens_per_image from the Gemma 3n
  // config. Both are fixed. The vision tower always sees one 768x768 image and
  // emits a 16x16 grid. Audio is fixed on the *model* side rather than the
  // feature-extractor side: the mel length varies with clip duration, and the
  // audio encoder pads short clips with the audio padding embedding and
  // truncates long ones so its output is always audio_seq_length_ rows
  // (HF Gemma3nModel.forward's extra_padding_features step).
  size_t image_seq_length_{256};
  size_t audio_seq_length_{188};
};

}  // namespace Generators
