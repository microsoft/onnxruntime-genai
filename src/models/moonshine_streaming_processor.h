// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
//
// MoonshineStreamingProcessor: real-time chunk-by-chunk streaming for the
// official UsefulSensors Moonshine streaming ONNX export.
//
// Pipeline per chunk:
//   1. frontend.onnx   : audio → new feature frames + updated state buffers.
//   2. (accumulate)    : append new features to a persistent buffer.
//   3. encoder.onnx    : run on a sliding window of accumulated features
//                        ([encoder_frames_emitted - left_context_frames :
//                          total_features]); slice the new-stable rows.
//   4. adapter.onnx    : run on the new-stable rows with a monotonically
//                        increasing pos_offset; append to accumulated memory.
//   5. cross_kv.onnx   : recompute from the full accumulated memory whenever
//                        memory grew; cache the {k_cross, v_cross} tensors
//                        and emit them via NamedTensors so the State can
//                        feed them to decoder_kv on every step.
//
// The State (MoonshineStreamingState) is responsible for re-decoding from
// BOS on every chunk with the current cross-KV.
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
  MoonshineStreamingModel* moonshine_model_;  // cached typed view (verified non-null in ctor)
  MoonshineConfig config_;

  // Audio not yet fed to the frontend (drained chunk_samples at a time).
  std::vector<float> audio_buffer_;

  // ---- Persistent frontend state ----------------------------------------
  // Lives for the life of the stream; reset by Flush() so the next
  // utterance starts clean.
  std::unique_ptr<OrtValue> sample_buffer_;
  std::unique_ptr<OrtValue> sample_len_;
  std::unique_ptr<OrtValue> conv1_buffer_;
  std::unique_ptr<OrtValue> conv2_buffer_;
  std::unique_ptr<OrtValue> frame_count_;

  // ---- Persistent encoder / adapter / memory state ----------------------
  // Accumulated encoder-input features ([total_features_, encoder_dim]).
  std::vector<float> accumulated_features_;
  int total_features_{0};
  // Number of features already adapted into memory.
  int encoder_frames_emitted_{0};
  // Running pos_offset fed to adapter.onnx (== total memory frames so far).
  int64_t adapter_pos_offset_{0};
  // Accumulated decoder memory ([memory_len_, decoder_dim]).
  std::vector<float> accumulated_memory_;
  int memory_len_{0};

  // Cached cross-KV tensors. Reused unless memory grew this chunk.
  std::shared_ptr<Tensor> cached_k_cross_;
  std::shared_ptr<Tensor> cached_v_cross_;
  bool cross_kv_valid_{false};

  // ---- Helpers -----------------------------------------------------------
  void ResetState();

  /// Run the frontend on `num` samples of audio, then accumulate features
  /// and (if any new stable frames have been produced) encoder + adapter +
  /// append-to-memory. Sets cross_kv_valid_ = false when memory grew.
  ///   is_final: when true, treat all accumulated features as stable (no
  ///             lookahead held back) — used by Flush().
  void RunFrontendAndAccumulate(const float* audio, size_t num, bool is_final);

  /// If !cross_kv_valid_, run cross_kv.onnx on the full accumulated memory
  /// and refresh cached_k_cross_ / cached_v_cross_. No-op otherwise.
  void RefreshCrossKv();

  /// Build a NamedTensors holding the cached {k_cross, v_cross}; nullptr if
  /// memory is empty. `is_final` is forwarded to the State via an int64[1]
  /// "is_final" tensor so it knows to commit all tokens (vs only the
  /// longest-common-prefix between consecutive passes).
  std::unique_ptr<NamedTensors> EmitCrossKv(bool is_final);
};

}  // namespace Generators
