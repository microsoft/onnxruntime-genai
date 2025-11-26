// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "../generators.h"
#include "vision_pipeline.h"
#include "decoder_only_pipeline.h"
#include "model.h"
#include <fstream>
#include <cstring>

namespace Generators {

VisionPipelineState::VisionPipelineState(const Model& model, const GeneratorParams& params)
    : State{params, model},
      model_{model} {
  
  std::cout << "[VisionPipeline] Constructor started" << std::endl;
  
  allocator_cpu_ = &model.allocator_cpu_;
  allocator_device_ = model.p_device_;

  std::cout << "[VisionPipeline] Allocators set" << std::endl;

  // Load window indexing if configured
  if (model.config_->model.vision.window_indexing.has_value()) {
    std::cout << "[VisionPipeline] Loading window indexing..." << std::endl;
    spatial_merge_size_ = model.config_->model.vision.window_indexing.value().spatial_merge_size;
    
    // Load window index file
    auto wnd_idx_filename = model.config_->model.vision.window_indexing.value().filename;
    fs::path wnd_idx_path = model.config_->config_path / fs::path(wnd_idx_filename);
    std::cout << "[VisionPipeline] Window index path: " << wnd_idx_path.string() << std::endl;
    LoadWindowIndexing(wnd_idx_path);
    std::cout << "[VisionPipeline] Window indexing loaded successfully" << std::endl;
  }
  
  std::cout << "[VisionPipeline] Getting decoder model..." << std::endl;
  // Get the OrtEnv from the decoder model
  const auto* decoder_model = dynamic_cast<const DecoderOnlyPipelineModel*>(&model);
  if (!decoder_model) {
    throw std::runtime_error("Vision pipeline requires DecoderOnlyPipelineModel");
  }
  
  std::cout << "[VisionPipeline] Creating sessions for vision pipeline stages..." << std::endl;
  // Create sessions for each stage of the vision pipeline
  for (size_t stage_idx = 0; stage_idx < model.config_->model.vision.pipeline.size(); ++stage_idx) {
    const auto& pipeline_model = model.config_->model.vision.pipeline[stage_idx];
    std::cout << "[VisionPipeline] Creating session " << stage_idx << ": " << pipeline_model.filename << std::endl;
    // CreateSession expects just the filename, not the full path
    // It will prepend config_path internally
    const std::string& model_filename = pipeline_model.filename;
    
    std::cout << "[VisionPipeline] Getting session options..." << std::endl;
    // Get or create session options for this pipeline model
    OrtSessionOptions* session_options = nullptr;
    
    if (pipeline_model.run_on_cpu) {
      std::cout << "[VisionPipeline] Using CPU session options" << std::endl;
      // For CPU execution, create minimal session options with CPU provider
      session_options = model.session_options_.get();  // Use default session options which will fall back to CPU
    } else {
      std::cout << "[VisionPipeline] Using model-specific session options for: " << pipeline_model.model_id << std::endl;
      // For non-CPU execution (e.g., QNN), use the model-specific session options
      session_options = model.GetSessionOptions(pipeline_model.model_id);
    }
    
    std::cout << "[VisionPipeline] Creating ORT session..." << std::endl;
    // Create session - pass just the filename
    auto session = const_cast<DecoderOnlyPipelineModel*>(decoder_model)->CreateSession(
        const_cast<OrtEnv&>(decoder_model->ort_env_), 
        model_filename, 
        session_options);
    
    std::cout << "[VisionPipeline] Session created successfully" << std::endl;
    
    // Store session based on order: patch_embed, vision_attn, patch_merger
    if (!patch_embed_session_) {
      patch_embed_session_ = std::move(session);
      std::cout << "[VisionPipeline] Stored as patch_embed_session" << std::endl;
    } else if (!vision_attn_session_) {
      vision_attn_session_ = std::move(session);
      std::cout << "[VisionPipeline] Stored as vision_attn_session" << std::endl;
    } else if (!patch_merger_session_) {
      patch_merger_session_ = std::move(session);
      std::cout << "[VisionPipeline] Stored as patch_merger_session" << std::endl;
    }
  }
  
  std::cout << "[VisionPipeline] Constructor completed successfully" << std::endl;
}

void VisionPipelineState::LoadWindowIndexing(const fs::path& wnd_idx_path) {
  if (!fs::exists(wnd_idx_path)) {
    throw std::runtime_error("Window indexing file not found: " + wnd_idx_path.string());
  }
  
  // Load numpy file - simplified loader for .npy format
  // NPY format: magic (6 bytes) + version (2 bytes) + header length (2/4 bytes) + header (JSON-like dict) + data
  std::ifstream file(wnd_idx_path.string(), std::ios::binary);
  if (!file) {
    throw std::runtime_error("Failed to open window indexing file: " + wnd_idx_path.string());
  }
  
  // Read magic bytes
  char magic[6];
  file.read(magic, 6);
  if (std::string(magic, 6) != "\x93NUMPY") {
    throw std::runtime_error("Invalid NPY file format");
  }
  
  // Read version
  uint8_t major_version, minor_version;
  file.read(reinterpret_cast<char*>(&major_version), 1);
  file.read(reinterpret_cast<char*>(&minor_version), 1);
  
  // Read header length
  uint32_t header_len;
  if (major_version == 1) {
    uint16_t header_len_16;
    file.read(reinterpret_cast<char*>(&header_len_16), 2);
    header_len = header_len_16;
  } else {
    file.read(reinterpret_cast<char*>(&header_len), 4);
  }
  
  // Skip header (contains shape and dtype info)
  file.seekg(header_len, std::ios::cur);
  
  // Read data - assume int64 based on FARA implementation
  // Read until end of file to get all indices
  std::vector<int64_t> indices;
  int64_t value;
  while (file.read(reinterpret_cast<char*>(&value), sizeof(int64_t))) {
    indices.push_back(value);
  }
  
  window_indices_ = indices;
  
  // Compute reverse indices for restoring original order
  reverse_indices_.resize(window_indices_.size());
  for (size_t i = 0; i < window_indices_.size(); ++i) {
    reverse_indices_[window_indices_[i]] = static_cast<int64_t>(i);
  }
}

std::unique_ptr<OrtValue> VisionPipelineState::ProcessImage(OrtValue* pixel_values, OrtValue* image_grid_thw) {
  std::cout << "[ProcessImage] Stage 1: Running patch embed..." << std::endl;
  // Stage 1: Patch Embed
  RunPatchEmbed(pixel_values);
  std::cout << "[ProcessImage] Stage 1 completed" << std::endl;
  
  // Stage 2: Apply window indexing reordering  
  if (!window_indices_.empty()) {
    std::cout << "[ProcessImage] Stage 2: Applying window indexing..." << std::endl;
    
    // Get current shape
    auto shape_info = patch_embed_output_->GetTensorTypeAndShapeInfo();
    auto shape = shape_info->GetShape();
    int64_t batch_size = shape[0];
    int64_t seq_len = shape[1];
    int64_t hidden_dim = shape[2];
    
    std::cout << "[ProcessImage] Current seq_len=" << seq_len << ", window_indices size=" << window_indices_.size() << std::endl;
    
    // The window indices select patches and each selected patch is expanded by spatial_merge_size^2
    // window_indices_ has 493 elements, each expanded to 4 patches = 1972 total
    int64_t patches_per_index = spatial_merge_size_ * spatial_merge_size_;  // 2*2 = 4
    int64_t output_seq_len = window_indices_.size() * patches_per_index;    // 493 * 4 = 1972
    
    std::cout << "[ProcessImage] Expanding via window indices: " << window_indices_.size() 
              << " indices * " << patches_per_index << " patches/index = " << output_seq_len << " output patches" << std::endl;
    
    std::vector<int64_t> output_shape = {batch_size, output_seq_len, hidden_dim};
    reordered_patches_ = OrtValue::CreateTensor<float>(*allocator_cpu_, std::span<const int64_t>(output_shape));
    
    const float* src_data = patch_embed_output_->GetTensorData<float>();
    float* dst_data = reordered_patches_->GetTensorMutableData<float>();
    
    // For each window index, copy the corresponding source patch to 4 output positions
    for (int64_t b = 0; b < batch_size; ++b) {
      for (size_t i = 0; i < window_indices_.size(); ++i) {
        int64_t src_idx = window_indices_[i];
        
        // Bounds check
        if (src_idx < 0 || src_idx >= seq_len) {
          throw std::runtime_error("Window index " + std::to_string(src_idx) + " out of bounds [0, " + std::to_string(seq_len) + ")");
        }
        
        const float* src_patch = &src_data[b * seq_len * hidden_dim + src_idx * hidden_dim];
        
        // Duplicate this patch to 4 consecutive output positions
        for (int64_t rep = 0; rep < patches_per_index; ++rep) {
          int64_t dst_idx = i * patches_per_index + rep;
          float* dst_patch = &dst_data[b * output_seq_len * hidden_dim + dst_idx * hidden_dim];
          std::memcpy(dst_patch, src_patch, hidden_dim * sizeof(float));
        }
      }
    }
    
    std::cout << "[ProcessImage] Expanded sequence from " << seq_len << " to " << output_seq_len << std::endl;
    std::cout << "[ProcessImage] Stage 2 completed" << std::endl;
  } else {
    reordered_patches_ = std::move(patch_embed_output_);
  }
  
  // Stage 3: Vision Attention
  std::cout << "[ProcessImage] Stage 3: Running vision attention..." << std::endl;
  RunVisionAttention();
  std::cout << "[ProcessImage] Stage 3 completed" << std::endl;
  
  // Stage 4: Skip reverse window indexing - patch merger handles the reduction
  if (!window_indices_.empty()) {
    std::cout << "[ProcessImage] Stage 4: Skipping reverse window indexing (patch merger will handle reduction)..." << std::endl;
    // Pass the 1972-patch sequence directly to patch merger
    restored_patches_ = std::move(vision_attn_output_);
    std::cout << "[ProcessImage] Stage 4 completed" << std::endl;
  } else {
    restored_patches_ = std::move(vision_attn_output_);
  }
  
  // Stage 5: Patch Merger
  std::cout << "[ProcessImage] Stage 5: Running patch merger..." << std::endl;
  RunPatchMerger();
  std::cout << "[ProcessImage] Stage 5 completed" << std::endl;
  
  std::cout << "[ProcessImage] All stages completed successfully" << std::endl;
  return std::move(final_embeddings_);
}

void VisionPipelineState::RunPatchEmbed(OrtValue* pixel_values) {
  const auto& pipeline_config = model_.config_->model.vision.pipeline[0];
  
  std::vector<const char*> input_names;
  std::vector<OrtValue*> input_values;
  std::vector<const char*> output_names;
  
  for (const auto& input_name : pipeline_config.inputs) {
    input_names.push_back(input_name.c_str());
  }
  input_values.push_back(pixel_values);
  
  for (const auto& output_name : pipeline_config.outputs) {
    output_names.push_back(output_name.c_str());
  }
  
  std::vector<OrtValue*> outputs(output_names.size(), nullptr);
  patch_embed_session_->Run(nullptr, input_names.data(), input_values.data(), input_names.size(),
                            output_names.data(), outputs.data(), output_names.size());
  
  patch_embed_output_ = std::unique_ptr<OrtValue>(outputs[0]);
  
  // Check if we need to add batch dimension (model might output 2D tensor)
  auto shape_info = patch_embed_output_->GetTensorTypeAndShapeInfo();
  auto shape = shape_info->GetShape();
  
  if (shape.size() == 2) {
    // Need to reshape from [seq_len, hidden_dim] to [1, seq_len, hidden_dim]
    int64_t seq_len = shape[0];
    int64_t hidden_dim = shape[1];
    std::vector<int64_t> new_shape = {1, seq_len, hidden_dim};
    
    // Create new tensor with batch dimension
    auto reshaped = OrtValue::CreateTensor<float>(*allocator_cpu_, std::span<const int64_t>(new_shape));
    
    // Copy data
    const float* src_data = patch_embed_output_->GetTensorData<float>();
    float* dst_data = reshaped->GetTensorMutableData<float>();
    std::memcpy(dst_data, src_data, seq_len * hidden_dim * sizeof(float));
    
    patch_embed_output_ = std::move(reshaped);
  }
}

void VisionPipelineState::ApplyWindowIndexing() {
  std::cout << "[ApplyWindowIndexing] Starting..." << std::endl;
  // Get shape of patch_embed_output: [batch, seq_len, hidden_dim]
  auto shape_info = patch_embed_output_->GetTensorTypeAndShapeInfo();
  auto shape = shape_info->GetShape();
  std::cout << "[ApplyWindowIndexing] Shape: [" << shape[0] << ", " << shape[1] << ", " << shape[2] << "]" << std::endl;
  
  if (shape.size() != 3) {
    std::string shape_str = "[";
    for (size_t i = 0; i < shape.size(); ++i) {
      shape_str += std::to_string(shape[i]);
      if (i < shape.size() - 1) shape_str += ", ";
    }
    shape_str += "]";
    throw std::runtime_error("Expected 3D tensor for patch_embed_output, got shape: " + shape_str);
  }
  
  int64_t batch_size = shape[0];
  int64_t seq_len = shape[1];
  int64_t hidden_dim = shape[2];
  
  std::cout << "[ApplyWindowIndexing] batch_size=" << batch_size << ", seq_len=" << seq_len 
            << ", hidden_dim=" << hidden_dim << ", spatial_merge_size=" << spatial_merge_size_ << std::endl;
  std::cout << "[ApplyWindowIndexing] window_indices_.size()=" << window_indices_.size() << std::endl;
  
  // Window indices contain the reordering for the entire sequence, not per-block
  // The indices map from original position to new position
  
  std::cout << "[ApplyWindowIndexing] Getting input data..." << std::endl;
  const float* input_data = patch_embed_output_->GetTensorData<float>();
  
  std::cout << "[ApplyWindowIndexing] Creating output tensor..." << std::endl;
  // Create output tensor with same shape
  std::vector<int64_t> output_shape = {batch_size, seq_len, hidden_dim};
  reordered_patches_ = OrtValue::CreateTensor<float>(*allocator_cpu_, std::span<const int64_t>(output_shape));
  float* output_data = reordered_patches_->GetTensorMutableData<float>();
  
  std::cout << "[ApplyWindowIndexing] Starting reordering loop..." << std::endl;
  // Apply reordering: window_indices_[i] tells us where element i should go
  for (int64_t b = 0; b < batch_size; ++b) {
    for (int64_t i = 0; i < seq_len && i < static_cast<int64_t>(window_indices_.size()); ++i) {
      int64_t dst_pos = window_indices_[i];
      
      // Bounds check
      if (dst_pos < 0 || dst_pos >= seq_len) {
        std::cout << "[ApplyWindowIndexing] ERROR: Invalid dst_pos=" << dst_pos 
                  << " for i=" << i << ", seq_len=" << seq_len << std::endl;
        throw std::runtime_error("Window index out of bounds");
      }
      
      // Copy hidden_dim values from position i to dst_pos
      std::memcpy(&output_data[b * seq_len * hidden_dim + dst_pos * hidden_dim],
                  &input_data[b * seq_len * hidden_dim + i * hidden_dim],
                  hidden_dim * sizeof(float));
    }
    
    // If window_indices_ is shorter than seq_len, copy remaining elements unchanged
    for (int64_t i = window_indices_.size(); i < seq_len; ++i) {
      std::memcpy(&output_data[b * seq_len * hidden_dim + i * hidden_dim],
                  &input_data[b * seq_len * hidden_dim + i * hidden_dim],
                  hidden_dim * sizeof(float));
    }
  }
  std::cout << "[ApplyWindowIndexing] Completed successfully" << std::endl;
}

void VisionPipelineState::RunVisionAttention() {
  std::cout << "[RunVisionAttention] Starting..." << std::endl;
  const auto& pipeline_config = model_.config_->model.vision.pipeline[1];
  
  // Get shape of reordered_patches: [batch, seq_len, hidden_dim]
  auto shape_info = reordered_patches_->GetTensorTypeAndShapeInfo();
  auto shape = shape_info->GetShape();
  std::cout << "[RunVisionAttention] Input shape: [" << shape[0] << ", " << shape[1] << ", " << shape[2] << "]" << std::endl;
  
  // Vision attention model expects 2D input [seq_len, hidden_dim], so squeeze batch dimension
  int64_t batch_size = shape[0];
  int64_t seq_len = shape[1];
  int64_t hidden_dim = shape[2];
  
  std::unique_ptr<OrtValue> input_2d;
  if (batch_size == 1 && shape.size() == 3) {
    std::cout << "[RunVisionAttention] Reshaping from 3D [1, " << seq_len << ", " << hidden_dim 
              << "] to 2D [" << seq_len << ", " << hidden_dim << "]" << std::endl;
    // Reshape to 2D by removing batch dimension
    std::vector<int64_t> new_shape = {seq_len, hidden_dim};
    input_2d = OrtValue::CreateTensor<float>(*allocator_cpu_, std::span<const int64_t>(new_shape));
    
    const float* src_data = reordered_patches_->GetTensorData<float>();
    float* dst_data = input_2d->GetTensorMutableData<float>();
    std::memcpy(dst_data, src_data, seq_len * hidden_dim * sizeof(float));
  } else {
    input_2d = std::move(reordered_patches_);
  }
  
  std::vector<const char*> input_names;
  std::vector<OrtValue*> input_values;
  std::vector<const char*> output_names;
  
  for (const auto& input_name : pipeline_config.inputs) {
    input_names.push_back(input_name.c_str());
  }
  input_values.push_back(input_2d.get());
  
  // Get actual output names from the session instead of config
  auto session_output_names = vision_attn_session_->GetOutputNames();
  std::cout << "[RunVisionAttention] Model has " << session_output_names.size() << " outputs:" << std::endl;
  for (size_t i = 0; i < session_output_names.size(); ++i) {
    std::cout << "[RunVisionAttention]   Output " << i << ": " << session_output_names[i].c_str() << std::endl;
    output_names.push_back(session_output_names[i].c_str());
  }
  
  std::cout << "[RunVisionAttention] Running session..." << std::endl;
  std::vector<OrtValue*> outputs(output_names.size(), nullptr);
  vision_attn_session_->Run(nullptr, input_names.data(), input_values.data(), input_names.size(),
                            output_names.data(), outputs.data(), output_names.size());
  
  vision_attn_output_ = std::unique_ptr<OrtValue>(outputs[0]);
  
  // Check output shape and reshape back to 3D if needed
  auto output_shape_info = vision_attn_output_->GetTensorTypeAndShapeInfo();
  auto output_shape = output_shape_info->GetShape();
  std::cout << "[RunVisionAttention] Output shape: ";
  for (size_t i = 0; i < output_shape.size(); ++i) {
    std::cout << output_shape[i];
    if (i < output_shape.size() - 1) std::cout << ", ";
  }
  std::cout << std::endl;
  
  // If output is 2D [seq_len, hidden_dim], add batch dimension back
  if (output_shape.size() == 2) {
    std::cout << "[RunVisionAttention] Reshaping output from 2D to 3D" << std::endl;
    int64_t out_seq_len = output_shape[0];
    int64_t out_hidden_dim = output_shape[1];
    std::vector<int64_t> new_shape = {1, out_seq_len, out_hidden_dim};
    
    auto reshaped = OrtValue::CreateTensor<float>(*allocator_cpu_, std::span<const int64_t>(new_shape));
    const float* src_data = vision_attn_output_->GetTensorData<float>();
    float* dst_data = reshaped->GetTensorMutableData<float>();
    std::memcpy(dst_data, src_data, out_seq_len * out_hidden_dim * sizeof(float));
    
    vision_attn_output_ = std::move(reshaped);
  }
  
  std::cout << "[RunVisionAttention] Completed" << std::endl;
}

void VisionPipelineState::ReverseWindowIndexing() {
  // Get shape: [batch, seq_len, hidden_dim]
  auto shape_info = vision_attn_output_->GetTensorTypeAndShapeInfo();
  auto shape = shape_info->GetShape();
  
  int64_t batch_size = shape[0];
  int64_t seq_len = shape[1];
  int64_t hidden_dim = shape[2];
  
  const float* input_data = vision_attn_output_->GetTensorData<float>();
  
  // Create output tensor
  std::vector<int64_t> output_shape = {batch_size, seq_len, hidden_dim};
  restored_patches_ = OrtValue::CreateTensor<float>(*allocator_cpu_, std::span<const int64_t>(output_shape));
  float* output_data = restored_patches_->GetTensorMutableData<float>();
  
  // Apply reverse reordering: reverse_indices_[i] tells us where element at position i should go
  for (int64_t b = 0; b < batch_size; ++b) {
    for (int64_t i = 0; i < seq_len && i < static_cast<int64_t>(reverse_indices_.size()); ++i) {
      int64_t dst_pos = reverse_indices_[i];
      
      // Bounds check
      if (dst_pos < 0 || dst_pos >= seq_len) {
        throw std::runtime_error("Reverse window index out of bounds");
      }
      
      // Copy hidden_dim values from position i to dst_pos
      std::memcpy(&output_data[b * seq_len * hidden_dim + dst_pos * hidden_dim],
                  &input_data[b * seq_len * hidden_dim + i * hidden_dim],
                  hidden_dim * sizeof(float));
    }
    
    // If reverse_indices_ is shorter than seq_len, copy remaining elements unchanged
    for (int64_t i = reverse_indices_.size(); i < seq_len; ++i) {
      std::memcpy(&output_data[b * seq_len * hidden_dim + i * hidden_dim],
                  &input_data[b * seq_len * hidden_dim + i * hidden_dim],
                  hidden_dim * sizeof(float));
    }
  }
}

void VisionPipelineState::RunPatchMerger() {
  std::cout << "[RunPatchMerger] Starting..." << std::endl;
  const auto& pipeline_config = model_.config_->model.vision.pipeline[2];
  
  // Get shape of restored_patches: [batch, seq_len, hidden_dim]
  auto shape_info = restored_patches_->GetTensorTypeAndShapeInfo();
  auto shape = shape_info->GetShape();
  std::cout << "[RunPatchMerger] Input shape: [" << shape[0] << ", " << shape[1] << ", " << shape[2] << "]" << std::endl;
  
  // Patch merger model expects 2D input [seq_len, hidden_dim], so squeeze batch dimension
  int64_t batch_size = shape[0];
  int64_t seq_len = shape[1];
  int64_t hidden_dim = shape[2];
  
  std::unique_ptr<OrtValue> input_2d;
  if (batch_size == 1 && shape.size() == 3) {
    std::cout << "[RunPatchMerger] Reshaping from 3D [1, " << seq_len << ", " << hidden_dim 
              << "] to 2D [" << seq_len << ", " << hidden_dim << "]" << std::endl;
    // Reshape to 2D by removing batch dimension
    std::vector<int64_t> new_shape = {seq_len, hidden_dim};
    input_2d = OrtValue::CreateTensor<float>(*allocator_cpu_, std::span<const int64_t>(new_shape));
    
    const float* src_data = restored_patches_->GetTensorData<float>();
    float* dst_data = input_2d->GetTensorMutableData<float>();
    std::memcpy(dst_data, src_data, seq_len * hidden_dim * sizeof(float));
  } else {
    input_2d = std::move(restored_patches_);
  }
  
  std::vector<const char*> input_names;
  std::vector<OrtValue*> input_values;
  std::vector<const char*> output_names;
  
  for (const auto& input_name : pipeline_config.inputs) {
    input_names.push_back(input_name.c_str());
  }
  input_values.push_back(input_2d.get());
  
  for (const auto& output_name : pipeline_config.outputs) {
    output_names.push_back(output_name.c_str());
  }
  
  std::cout << "[RunPatchMerger] Running session..." << std::endl;
  std::vector<OrtValue*> outputs(output_names.size(), nullptr);
  patch_merger_session_->Run(nullptr, input_names.data(), input_values.data(), input_names.size(),
                             output_names.data(), outputs.data(), output_names.size());
  
  auto merger_output = std::unique_ptr<OrtValue>(outputs[0]);
  
  // Check output shape
  auto output_shape_info = merger_output->GetTensorTypeAndShapeInfo();
  auto output_shape = output_shape_info->GetShape();
  std::cout << "[RunPatchMerger] Output shape: ";
  for (size_t i = 0; i < output_shape.size(); ++i) {
    std::cout << output_shape[i];
    if (i < output_shape.size() - 1) std::cout << ", ";
  }
  std::cout << std::endl;
  
  // Apply reverse window indexing to restore original spatial order
  // The merger outputs embeddings in window-indexed order [wnd_idx[0], wnd_idx[1], ...]
  // We need to reorder them back to [0, 1, 2, ...] using reverse_indices_
  if (!window_indices_.empty() && reverse_indices_.size() == static_cast<size_t>(output_shape[0])) {
    std::cout << "[RunPatchMerger] Applying reverse window indexing to restore original order..." << std::endl;
    
    int64_t num_patches = output_shape[0];
    int64_t output_hidden_dim = output_shape[1];
    
    const float* src_data = merger_output->GetTensorData<float>();
    
    // Create reordered output
    std::vector<int64_t> reordered_shape = {num_patches, output_hidden_dim};
    final_embeddings_ = OrtValue::CreateTensor<float>(*allocator_cpu_, std::span<const int64_t>(reordered_shape));
    float* dst_data = final_embeddings_->GetTensorMutableData<float>();
    
    // Apply reverse reordering: element at position i goes to position reverse_indices_[i]
    for (int64_t i = 0; i < num_patches; ++i) {
      int64_t dst_pos = reverse_indices_[i];
      
      // Bounds check
      if (dst_pos < 0 || dst_pos >= num_patches) {
        throw std::runtime_error("Reverse window index out of bounds: " + std::to_string(dst_pos));
      }
      
      // Copy embedding from position i in src to position dst_pos in output
      std::memcpy(&dst_data[dst_pos * output_hidden_dim],
                  &src_data[i * output_hidden_dim],
                  output_hidden_dim * sizeof(float));
    }
    
    std::cout << "[RunPatchMerger] Reverse indexing completed, restored original spatial order" << std::endl;
  } else {
    final_embeddings_ = std::move(merger_output);
  }
  
  std::cout << "[RunPatchMerger] Completed" << std::endl;
}

DeviceSpan<float> VisionPipelineState::Run(int total_length, DeviceSpan<int32_t>& next_tokens, DeviceSpan<int32_t> next_indices) {
  // This method is not used for vision pipeline - ProcessImage is the main entry point
  throw std::runtime_error("VisionPipelineState::Run should not be called directly");
}

}  // namespace Generators
