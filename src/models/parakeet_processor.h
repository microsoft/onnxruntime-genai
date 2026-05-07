// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
//
// Parakeet-TDT processor: 128-mel log-spectrogram with per-feature normalization,
// plus tokens.txt-based detokenizer.

#pragma once

#include <optional>
#include <string>
#include <vector>

#include "processor.h"

namespace Generators {

struct ParakeetProcessor : Processor {
  ParakeetProcessor(Config& config, const SessionInfo& session_info);

  ParakeetProcessor() = delete;
  ParakeetProcessor(const ParakeetProcessor&) = delete;
  ParakeetProcessor& operator=(const ParakeetProcessor&) = delete;

  std::unique_ptr<NamedTensors> Process(const Tokenizer& tokenizer,
                                        const Payload& payload) const override;

  std::optional<std::string> Decode(std::span<const int32_t> tokens) const override;

 private:
  // Audio config
  int sample_rate_{16000};
  int num_mels_{128};
  int fft_size_{512};
  int hop_length_{160};
  int win_length_{400};
  float preemph_{0.0f};
  float log_eps_{1e-10f};

  int blank_id_{8192};

  // Loaded from tokens.txt
  std::vector<std::string> id_to_token_;
};

}  // namespace Generators
