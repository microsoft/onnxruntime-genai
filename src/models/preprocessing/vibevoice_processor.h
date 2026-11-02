// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#pragma once

#include <string>
#include <string_view>

#include "processor.h"

namespace Generators {

// The text that VibeVoice reads for `text`, as its demo and processor prepare it: typographic quotes
// become straight quotes, surrounding white space is removed and a line break ends the text. Empty
// when the text has nothing to speak.
std::string PrepareVibeVoiceText(std::string_view text);

// Text to OgaInputName_TextIds ([1, n] int64) for a VibeVoice generator. The tokenizer runs in literal mode, so
// text that looks like a special token (for example "<|endoftext|>") is read as plain text.
struct VibeVoiceProcessor : Processor {
  VibeVoiceProcessor(Config& config, const SessionInfo& session_info);

  void ConfigureTokenizer(Tokenizer& tokenizer) const override;
  std::unique_ptr<NamedTensors> Process(const Tokenizer& tokenizer, const Payload& payload) const override;
};

}  // namespace Generators
