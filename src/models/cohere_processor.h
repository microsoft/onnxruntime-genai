// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#pragma once

#include "processor.h"
#include "nemo_mel_spectrogram.h"

namespace Generators {

struct CohereProcessor : Processor {
  CohereProcessor(Config& config, const SessionInfo& session_info);

  CohereProcessor() = delete;
  CohereProcessor(const CohereProcessor&) = delete;
  CohereProcessor& operator=(const CohereProcessor&) = delete;

  std::unique_ptr<NamedTensors> Process(const Tokenizer& tokenizer, const Payload& payload) const override;

 private:
  ONNXTensorElementDataType audio_features_type_;
  nemo_mel::NemoMelConfig mel_cfg_;
};

}  // namespace Generators
