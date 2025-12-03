// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
//
// Fara VLM Vision pipeline support (initial skeleton).
// Executes three ONNX models in sequence:
//   1) Patch Embedding  : pixel_values -> hidden
//   2) Vision Attention : hidden -> hidden
//   3) Patch Merger      : hidden -> merged embeddings
// Performs window expansion/reordering using wnd_idx, then final reverse ordering.
//
// This is a minimal starting point to integrate Fara VLM vision processing
// into onnxruntime-genai. Further work will: (a) connect to Config parsing,
// (b) expose via MultiModal pipeline, (c) add EP selection, (d) reuse buffers.

#pragma once

#include <string>
#include <vector>
#include <memory>
#include <cstdint>

#include "onnxruntime_api.h"

namespace Generators {

// Simple loader for a 1D numpy .npy file containing integer indices.
// Supports little-endian int32/int64 arrays of shape (N,).
std::vector<int64_t> Load1DNpyIndices(const std::string& file_path);

// Internal vision pipeline (no external DLL interface required after Python binding removal).
struct FaraVisionPipeline {
  FaraVisionPipeline(OrtEnv& env,
                     const std::string& patch_embed_model,
                     const std::string& vision_attn_model,
                     const std::string& patch_merger_model,
                     int64_t spatial_merge_size,
                     const std::string& wnd_idx_path,
                     bool use_qnn_attn = false,
                     const std::string& qnn_backend_path = "QnnHtp.dll");
  bool use_qnn_attn_{};
  std::string qnn_backend_path_{};

  FaraVisionPipeline(const FaraVisionPipeline&) = delete;
  FaraVisionPipeline& operator=(const FaraVisionPipeline&) = delete;

  // Run vision pipeline.
  // pixel_values: float32 tensor with shape [S, C] or [B, C, H, W] depending on export (caller provides shape).
  // The ONNX model is assumed to accept the provided shape directly as 'pixel_values'.
  // Returns final merged embeddings (shape: [num_image_tokens, hidden_size]).
  std::vector<float> Run(const float* pixel_data, const std::vector<int64_t>& pixel_shape);

  // Shape info from last Run (seq_len, hidden_size). Returns empty vector if Run not called yet.
  std::vector<int64_t> GetLastOutputShape() const {
    if (last_seq_len_ <= 0 || last_hidden_size_ <= 0) return {};
    return {last_seq_len_, last_hidden_size_};
  }

  // Accessors
  const std::vector<int64_t>& GetWndIdx() const { return wnd_idx_; }
  int64_t GetSpatialMergeSize() const { return spatial_merge_size_; }

 private:
  // Internal helpers
  std::unique_ptr<OrtValue> CreateTensor(const float* data, size_t count, const std::vector<int64_t>& shape) const;

  std::unique_ptr<OrtSession> patch_embed_session_;
  std::unique_ptr<OrtSession> vision_attn_session_;
  std::unique_ptr<OrtSession> patch_merger_session_;

  std::vector<int64_t> wnd_idx_;  // window reordering indices
  std::vector<int64_t> rev_idx_;  // reverse ordering indices (argsort of wnd_idx)
  int64_t spatial_merge_size_{};
  OrtEnv& env_;
  int64_t last_seq_len_{0};
  int64_t last_hidden_size_{0};
};

} // namespace Generators
