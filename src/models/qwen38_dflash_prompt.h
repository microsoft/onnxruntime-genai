// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <cstddef>
#include <optional>

namespace Generators {

// Which chunks of a prompt the DFlash drafter ingests, and when.
//
// The drafter attends to a sliding window of `context_window` positions (its sliding window plus a block) at the end of the
// prompt, so only the chunks that reach into that window carry features it can use; the chunks before it are skipped. The
// window is anchored at the END of the prompt: positions are absolute and measured against the total prompt length, which
// the executor announces before the first chunk (Qwen38DecodeDevice::ExpectPrefillTokens) and which is never less than
// what the turn has ingested so far. From the first chunk that is ingested on, every later one is too, as the context
// of the drafter stays contiguous.
//
// The features of a chunk sit in a buffer that the next forward overwrites, and the executor delivers a prompt in many
// calls, so the last chunk of a call is saved and decided when the next call begins; the last chunk of the prompt is
// handed to decoding whatever the window says. A decision uses the prompt length known when it is taken, a lower bound of
// the true length: a chunk dropped for being before the window stays outside it, and one kept may only reach back further
// than it has to.
class Qwen38DflashPromptFeatures {
 public:
  struct Chunk {
    size_t position{};  // absolute position of its first row
    size_t rows{};
  };

  // `context_window` is 0 when the drafter uses the whole prompt.
  explicit Qwen38DflashPromptFeatures(size_t context_window = 0) noexcept : context_window_{context_window} {}

  // A new turn: nothing is saved and ingestion starts over.
  void Reset() noexcept {
    saved_.reset();
    ingesting_ = false;
  }

  // A call starts: returns the chunk the previous call saved when its features are to be ingested now (they are still in
  // the buffer); a saved chunk before the window is dropped.
  std::optional<Chunk> BeginCall(size_t total_tokens) noexcept {
    std::optional<Chunk> saved = TakeSavedChunk();
    if (saved && !Admit(*saved, total_tokens)) return std::nullopt;
    return saved;
  }

  // A forward of the call produced the features of `chunk`: returns it when they are to be ingested now. The last chunk of
  // the call is saved instead, and nothing is returned for it.
  std::optional<Chunk> ChunkDone(Chunk chunk, bool last_of_call, size_t total_tokens) noexcept {
    if (last_of_call) {
      saved_ = chunk;
      return std::nullopt;
    }
    if (!Admit(chunk, total_tokens)) return std::nullopt;
    return chunk;
  }

  // The saved chunk, which is no longer saved: the last chunk of the prompt when decoding starts.
  std::optional<Chunk> TakeSavedChunk() noexcept {
    std::optional<Chunk> saved = saved_;
    saved_.reset();
    return saved;
  }

 private:
  bool Admit(Chunk chunk, size_t total_tokens) noexcept {
    const size_t end = chunk.position + chunk.rows;
    const bool before_the_window = context_window_ > 0 && total_tokens > end && total_tokens - end >= context_window_;
    if (!ingesting_ && before_the_window) return false;
    ingesting_ = true;
    return true;
  }

  size_t context_window_;
  std::optional<Chunk> saved_;
  bool ingesting_{false};
};

}  // namespace Generators
