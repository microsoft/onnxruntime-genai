// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "generators.h"
#include "sequences.h"

namespace Generators {

void Sequences::AfterAppendNextTokens(DeviceSpan<int32_t> next_tokens) {
  if (g_log.enabled && g_log.append_next_tokens) {
    auto& stream = Log("append_next_tokens");
    DumpSpan(stream, next_tokens.CopyDeviceToCpu());
    stream << std::endl;
  }

  ++current_length_;

  // With beam search, we rotate the buffers each iteration
  if (!sequences_next_.empty())
    std::swap(sequences_, sequences_next_);
}

void Sequences::GetLastTokens(cpu_span<int32_t>& last_tokens) {
  for (int i = 0; i < batch_beam_size_; i++) {
    last_tokens[i] = sequences_->CpuSpan()[i * max_length_ + current_length_ - 1];
  }
}

void Sequences::RewindTo(size_t index) {
  current_length_ = static_cast<int>(index);
  assert(current_length_ >= 0);
}

}  // namespace Generators
