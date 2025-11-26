// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "model.h"
#include "input_ids.h"
#include "logits.h"
#include "kv_cache.h"

namespace Generators {

// Manages execution of multi-stage vision pipeline for Qwen 2.5 VL
// Pipeline: patch_embed (CPU) -> vision_attn (NPU) -> patch_merger (CPU)
struct VisionPipelineState : State {
  VisionPipelineState(const Model& model, const GeneratorParams& params);
  
  // Process image through the 3-stage vision pipeline
  // Input: pixel_values tensor from image processor
  // Output: image embeddings ready for injection into text stream
  std::unique_ptr<OrtValue> ProcessImage(OrtValue* pixel_values, OrtValue* image_grid_thw);
  
  DeviceSpan<float> Run(int total_length, DeviceSpan<int32_t>& next_tokens, DeviceSpan<int32_t> next_indices) override;
  
 private:
  void RunPatchEmbed(OrtValue* pixel_values);
  void ApplyWindowIndexing();
  void RunVisionAttention();
  void ReverseWindowIndexing();
  void RunPatchMerger();
  
  // Load window indexing array from wnd_idx.npy
  void LoadWindowIndexing(const fs::path& wnd_idx_path);
  
  const Model& model_;
  
  // Pipeline stage sessions
  std::unique_ptr<OrtSession> patch_embed_session_;
  std::unique_ptr<OrtSession> vision_attn_session_;
  std::unique_ptr<OrtSession> patch_merger_session_;
  
  // Intermediate tensors between stages
  std::unique_ptr<OrtValue> patch_embed_output_;  // Output from patch_embed
  std::unique_ptr<OrtValue> reordered_patches_;   // After window indexing
  std::unique_ptr<OrtValue> vision_attn_output_;  // Output from vision_attn
  std::unique_ptr<OrtValue> restored_patches_;    // After reverse indexing
  std::unique_ptr<OrtValue> final_embeddings_;    // Output from patch_merger
  
  // Window indexing for spatial reordering
  std::vector<int64_t> window_indices_;
  std::vector<int64_t> reverse_indices_;
  int spatial_merge_size_{2};
  
  // Allocators
  Ort::Allocator* allocator_cpu_{};
  DeviceInterface* allocator_device_{};
};

}  // namespace Generators
