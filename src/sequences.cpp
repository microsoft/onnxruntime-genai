// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "generator/generators.h"
#include "sequences.h"

namespace Generators {

void Sequences::AfterAppendNextTokens(DeviceSpan<int32_t>& next_tokens, size_t batch_beam_size) {
  if (g_log.enabled && g_log.append_next_tokens) {
    auto& stream = Log("append_next_tokens");
    DumpSpan(stream, next_tokens.CopyDeviceToCpu());
    stream << std::endl;
  }

  current_length_ += static_cast<int>(next_tokens.size() / batch_beam_size);

  // With beam search, we rotate the buffers each iteration
  if (!sequences_next_.empty())
    std::swap(sequences_, sequences_next_);
}

void Sequences::RewindTo(size_t index) {
  const bool keeps_encoder_prompt =
      encoder_prompt_length_ > 0 && index >= static_cast<size_t>(encoder_prompt_length_);
  current_length_ = static_cast<int>(index);
  if (encoder_prompt_length_ > current_length_)
    encoder_prompt_length_ = current_length_;
  if (encoder_prompt_length_ + decoder_prompt_length_ > current_length_)
    decoder_prompt_length_ = current_length_ - encoder_prompt_length_;
  if (!keeps_encoder_prompt && current_length_ <= encoder_prompt_length_ + decoder_prompt_length_)
    has_generated_tokens_ = false;
  assert(current_length_ >= 0);
}

}  // namespace Generators
