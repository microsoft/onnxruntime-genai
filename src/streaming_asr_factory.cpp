// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
//
// CreateStreamingASR factory — routes to the correct StreamingASR implementation
// based on the model type (nemotron_speech → NemoStreamingASR, parakeet_tdt → ParakeetStreamingASR).

#include "generators.h"
#include "streaming_asr.h"
#include "nemo_streaming_asr.h"
#include "parakeet_streaming_asr.h"
#include "models/parakeet_speech.h"
#include "models/nemotron_speech.h"

namespace Generators {

std::unique_ptr<StreamingASR> CreateStreamingASR(Model& model) {
  if (model.config_->model.type == "parakeet_tdt") {
    return std::make_unique<ParakeetStreamingASR>(model);
  }
  // Default to NemoStreamingASR for nemotron_speech and any other streaming type
  return std::make_unique<NemoStreamingASR>(model);
}

}  // namespace Generators
