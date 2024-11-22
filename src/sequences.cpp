// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "generators.h"
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
  current_length_ = static_cast<int>(index);
  assert(current_length_ >= 0);
}

}  // namespace Generators
