// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <string>
#include <vector>
#include <memory>
#include <cstdint>
#include <stdexcept>

#include "onnxruntime_api.h"

namespace Generators {

// Validates grid dimensions and config parameters for the Qwen vision window indexing.
// Throws std::runtime_error on invalid inputs.
// Defined inline in the header so that unit tests can call it directly without requiring
// DLL export, since this is a free function not part of the public C API.
inline void ValidateWindowIndexParams(int64_t grid_t, int64_t grid_h, int64_t grid_w,
                                      int64_t spatial_merge_size, int64_t patch_size, int64_t window_size) {
  if (spatial_merge_size <= 0)
    throw std::runtime_error("CalculateWindowIndex: spatial_merge_size must be positive");
  if (patch_size <= 0)
    throw std::runtime_error("CalculateWindowIndex: patch_size must be positive");
  if (grid_t <= 0 || grid_h <= 0 || grid_w <= 0)
    throw std::runtime_error("CalculateWindowIndex: grid dimensions must be positive");
  if (grid_h % spatial_merge_size != 0 || grid_w % spatial_merge_size != 0)
    throw std::runtime_error("CalculateWindowIndex: grid_h and grid_w must be divisible by spatial_merge_size");

  int64_t vit_merger_window_size = window_size / spatial_merge_size / patch_size;
  if (vit_merger_window_size <= 0)
    throw std::runtime_error("CalculateWindowIndex: vit_merger_window_size must be positive (check window_size, spatial_merge_size, patch_size config)");

  constexpr int64_t kMaxElements = static_cast<int64_t>(1) << 30;
  int64_t llm_grid_h = grid_h / spatial_merge_size;
  int64_t llm_grid_w = grid_w / spatial_merge_size;
  if (llm_grid_h > kMaxElements || llm_grid_w > kMaxElements || grid_t > kMaxElements)
    throw std::runtime_error("CalculateWindowIndex: grid dimensions are too large");

  int64_t pad_h = (vit_merger_window_size - (llm_grid_h % vit_merger_window_size)) % vit_merger_window_size;
  int64_t pad_w = (vit_merger_window_size - (llm_grid_w % vit_merger_window_size)) % vit_merger_window_size;
  int64_t padded_h = llm_grid_h + pad_h;
  int64_t padded_w = llm_grid_w + pad_w;
  if (padded_h > 0 && padded_w > 0 && grid_t > kMaxElements / padded_h / padded_w)
    throw std::runtime_error("CalculateWindowIndex: total grid size exceeds maximum allowed");
}

// Internal vision pipeline (no external DLL interface required after Python binding removal).
struct QwenVisionPipeline {
  QwenVisionPipeline(OrtEnv& env,
                     const std::string& patch_embed_model,
                     const std::string& vision_attn_model,
                     const std::string& patch_merger_model,
                     int64_t spatial_merge_size,
                     int64_t patch_size = 14,
                     int64_t window_size = 0,
                     const OrtSessionOptions* vision_attn_session_options = nullptr);
  const OrtSessionOptions* vision_attn_session_options_{};

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

  // Hidden dimension sizes read from the ONNX model output shapes at construction time.
  // Populated in the constructor; -1 means the dimension is dynamic/unknown.
  int64_t hidden_dim_{-1};     // patch_embed output dim[1] (e.g. 1152 for 3B, 1280 for 7B)
  int64_t merged_hidden_{-1};  // patch_merger output dim[1] (e.g. 2048 for 3B, 3584 for 7B)

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
