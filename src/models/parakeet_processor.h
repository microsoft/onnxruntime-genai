// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
//
// ParakeetTdtProcessor — decodes the user-provided audio and computes the
// full mel-spectrogram (with NeMo-style per-feature normalization).
//
// Outputs in the NamedTensors map:
//   * "mel_features" : float32 [1, num_mels, total_frames] — globally
//     mean/std normalized.
//   * "input_ids"    : int32   [1, 1] — decoder SOS token (drives the
//     standard genai Generator search loop).

#pragma once

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
  const Config& config_;
};

}  // namespace Generators
