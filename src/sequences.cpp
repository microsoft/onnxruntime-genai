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

}  // namespace Generators
