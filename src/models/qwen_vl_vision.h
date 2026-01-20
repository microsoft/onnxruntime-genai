// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <string>
#include <vector>
#include <memory>
#include <cstdint>

#include "onnxruntime_api.h"

namespace Generators {

// Internal vision pipeline (no external DLL interface required after Python binding removal).
struct QwenVisionPipeline {
  QwenVisionPipeline(OrtEnv& env,
                     const std::string& patch_embed_model,
                     const std::string& vision_attn_model,
                     const std::string& patch_merger_model,
                     int64_t spatial_merge_size,
                     bool use_qnn_attn = false,
                     const std::string& qnn_backend_path = "QnnHtp.dll",
                     int64_t patch_size = 14,
                     int64_t window_size = 56);
  bool use_qnn_attn_{};
  std::string qnn_backend_path_{};

  QwenVisionPipeline(const QwenVisionPipeline&) = delete;
  QwenVisionPipeline& operator=(const QwenVisionPipeline&) = delete;

  // Run vision pipeline.
  // pixel_values: float32 tensor with shape [S, C] or [B, C, H, W] depending on export (caller provides shape).
  // grid_thw: optional grid dimensions [temporal, height, width] for dynamic window indexing
  // The ONNX model is assumed to accept the provided shape directly as 'pixel_values'.
  // Returns final merged embeddings (shape: [num_image_tokens, hidden_size]).
  std::vector<float> Run(const float* pixel_data, const std::vector<int64_t>& pixel_shape,
                         const std::vector<int64_t>& grid_thw = {});

  // Shape info from last Run (seq_len, hidden_size). Returns empty vector if Run not called yet.
  std::vector<int64_t> GetLastOutputShape() const {
    if (last_seq_len_ <= 0 || last_hidden_size_ <= 0) return {};
    return {last_seq_len_, last_hidden_size_};
  }

 private:
  // Internal helpers
  std::unique_ptr<OrtValue> CreateTensor(const float* data, size_t count, const std::vector<int64_t>& shape) const;

  // Calculate window indices dynamically based on grid dimensions
  // Returns window_index (reordering indices for windowing)
  std::vector<int64_t> CalculateWindowIndex(int64_t grid_t, int64_t grid_h, int64_t grid_w);

  std::unique_ptr<OrtSession> patch_embed_session_;
  std::unique_ptr<OrtSession> vision_attn_session_;
  std::unique_ptr<OrtSession> patch_merger_session_;

  std::vector<int64_t> wnd_idx_;  // window reordering indices (computed dynamically)
  std::vector<int64_t> rev_idx_;  // reverse ordering indices (argsort of wnd_idx)
  int64_t spatial_merge_size_{};
  int64_t patch_size_{14};   // Vision patch size (typically 14)
  int64_t window_size_{56};  // Window size for attention (typically 56)
  OrtEnv& env_;
  int64_t last_seq_len_{0};
  int64_t last_hidden_size_{0};

  // Reusable buffers to avoid repeated allocation/deallocation
  mutable std::vector<float> pe_out_buf_;
  mutable std::vector<float> reordered_buf_;
  mutable std::vector<float> attn_out_buf_;
  mutable std::vector<float> merger_out_buf_;
  mutable std::vector<float> final_embeddings_buf_;
};

}  // namespace Generators
