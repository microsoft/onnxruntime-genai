// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
//
// This processor:
//   1. buffers incoming audio and drains it chunk_samples at a time
//      (default 500ms @ 16kHz),
//   2. runs the per-chunk VAD verdict (IsChunkSilent),
//   3. emits one NamedTensors per chunk with:
//        "audio_chunk" : float32 [1, num_samples] raw audio,
//        "is_silent"   : int64  [1] (1 iff VAD flagged this chunk silent),
//        "is_final"    : int64  [1] (1 only on the Flush() tail chunk).
//
// Everything stateful, the frontend causal buffers, accumulated features,
// encoder sliding window, adapter memory, incremental cross-KV cache, self-KV,
// segment resets, and re-decode-from-BOS — lives in MoonshineStreamingState,
// which owns all five ONNX sub-states.
#pragma once

#include "moonshine_streaming.h"
#include "streaming_processor.h"

namespace Generators {

struct MoonshineStreamingProcessor : StreamingProcessor {
  explicit MoonshineStreamingProcessor(Model& model);
  ~MoonshineStreamingProcessor() override;

  std::unique_ptr<NamedTensors> Process(const float* audio_data, size_t num_samples) override;
  std::unique_ptr<NamedTensors> Flush() override;

 private:
  Model& model_;
  MoonshineConfig config_;

  std::vector<float> audio_buffer_;

  // Build the {audio_chunk, is_silent, is_final} NamedTensors for one chunk.
  std::unique_ptr<NamedTensors> EmitChunk(const float* audio, size_t num,
                                          bool is_silent, bool is_final);
};

}  // namespace Generators
