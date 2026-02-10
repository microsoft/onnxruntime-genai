// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
//
// Nemotron Speech processor — handles audio input preprocessing for streaming ASR.
#pragma once

#include "processor.h"

namespace Generators {

struct NemotronSpeechProcessor : Processor {
  NemotronSpeechProcessor(Config& config, const SessionInfo& session_info);

  NemotronSpeechProcessor() = delete;
  NemotronSpeechProcessor(const NemotronSpeechProcessor&) = delete;
  NemotronSpeechProcessor& operator=(const NemotronSpeechProcessor&) = delete;

  std::unique_ptr<NamedTensors> Process(const Tokenizer& tokenizer, const Payload& payload) const override;

 private:
  ort_extensions::OrtxObjectPtr<OrtxFeatureExtractor> processor_;
  ONNXTensorElementDataType audio_features_type_;
};

}  // namespace Generators
