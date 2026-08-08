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
  ONNXTensorElementDataType audio_features_type_;

  bool has_speech_{false};

  // vision_soft_tokens_per_image / audio_soft_tokens_per_image from the Gemma 3n
  // config. Both are fixed: the vision tower always sees one 768x768 image and
  // emits a 16x16 grid, and the feature extractor pads or truncates audio to the
  // length the conformer reduces to 188 frames.
  size_t image_seq_length_{256};
  size_t audio_seq_length_{188};
};

}  // namespace Generators
