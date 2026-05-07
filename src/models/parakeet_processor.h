// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
//
// ParakeetTdtProcessor — feeds raw PCM samples decoded from the user-provided
// Audios object into the Generator pipeline. The actual mel spectrogram and
// chunked TDT decoding happen inside ParakeetTdtState::SetExtraInputs (driven
// by these tensors), so this processor's job is just to:
//   * decode the audio file(s) to float32 mono PCM at the model's sample rate
//   * package them as a NamedTensors map containing "audio_pcm" and an
//     "input_ids" entry seeded with the decoder SOS token.

#pragma once

#include "model.h"
#include "ortx_cpp_helper.h"
#include "processor.h"

namespace Generators {

struct ParakeetTdtProcessor : Processor {
  ParakeetTdtProcessor(Config& config, const SessionInfo& session_info);

  ParakeetTdtProcessor() = delete;
  ParakeetTdtProcessor(const ParakeetTdtProcessor&) = delete;
  ParakeetTdtProcessor& operator=(const ParakeetTdtProcessor&) = delete;

  std::unique_ptr<NamedTensors> Process(const Tokenizer& tokenizer,
                                        const Payload& payload) const override;

 private:
  int sample_rate_;
  int32_t decoder_start_token_id_;
};

}  // namespace Generators
