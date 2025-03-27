// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#pragma once

#include "processor.h"

namespace Generators {

struct Config;
struct SessionInfo;

struct WhisperProcessor : Processor {
  WhisperProcessor(Config& config, const SessionInfo& session_info);

  WhisperProcessor() = delete;
  WhisperProcessor(const WhisperProcessor&) = delete;
  WhisperProcessor& operator=(const WhisperProcessor&) = delete;

  virtual std::unique_ptr<NamedTensors> Process(const Tokenizer& tokenizer, const Payload& payload) const override;

 private:
  ort_extensions::OrtxObjectPtr<OrtxFeatureExtractor> processor_;
  ONNXTensorElementDataType input_features_type_;
};

}  // namespace Generators
