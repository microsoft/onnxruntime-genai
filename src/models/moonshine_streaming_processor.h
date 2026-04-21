// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
//
// MoonshineStreamingProcessor: accumulates raw PCM audio, then on flush()
// runs the encoder in overlapping chunks and returns concatenated
// encoder_hidden_states for the decoder.
#pragma once

#include "streaming_processor.h"
#include "moonshine_streaming.h"

namespace Generators {

struct MoonshineStreamingProcessor : StreamingProcessor {
  explicit MoonshineStreamingProcessor(Model& model);
  ~MoonshineStreamingProcessor() override;

  std::unique_ptr<NamedTensors> Process(const float* audio_data, size_t num_samples) override;
  std::unique_ptr<NamedTensors> Flush() override;

 private:
  Model& model_;
  MoonshineConfig config_;

  // Audio accumulation buffer
  std::vector<float> audio_buffer_;

  /// Run encoder on all buffered audio (in overlapping chunks) and return
  /// the concatenated encoder_hidden_states as NamedTensors.
  std::unique_ptr<NamedTensors> EncodeAllAudio();
};

}  // namespace Generators
