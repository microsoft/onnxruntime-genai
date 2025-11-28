// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "../generators.h"
#include "vision_pipeline.h"
#include "../logging.h"
#include <iostream>

// Fallback logging macros to avoid build failures when LOGS_DEFAULT/VERBOSE are unavailable
#ifndef LOGS_DEFAULT
#define LOGS_DEFAULT(level) std::cout
#endif
#ifndef VERBOSE
#define VERBOSE 0
#endif
#include "decoder_only_pipeline.h"
#include "model.h"
#include <fstream>
#include <cstring>
#include <cmath>
#include <algorithm>

namespace Generators {

VisionPipelineState::VisionPipelineState(const Model& model, const GeneratorParams& params)
    : State{params, model},
      model_{model} {
  
  
  allocator_cpu_ = &model.allocator_cpu_;
  allocator_device_ = model.p_device_;


  // Load window indexing if configured
  if (model.config_->model.vision.window_indexing.has_value()) {
    spatial_merge_size_ = model.config_->model.vision.window_indexing.value().spatial_merge_size;
    
    // Load window index file
    auto wnd_idx_filename = model.config_->model.vision.window_indexing.value().filename;
    fs::path wnd_idx_path = model.config_->config_path / fs::path(wnd_idx_filename);
    LoadWindowIndexing(wnd_idx_path);
  }
  
  // Get the OrtEnv from the decoder model
  const auto* decoder_model = dynamic_cast<const DecoderOnlyPipelineModel*>(&model);
  if (!decoder_model) {
    throw std::runtime_error("Vision pipeline requires DecoderOnlyPipelineModel");
  }
  
  // Create sessions for each stage of the vision pipeline
  for (size_t stage_idx = 0; stage_idx < model.config_->model.vision.pipeline.size(); ++stage_idx) {
    const auto& pipeline_model = model.config_->model.vision.pipeline[stage_idx];
    // CreateSession expects just the filename, not the full path
    // It will prepend config_path internally
    const std::string& model_filename = pipeline_model.filename;
    
    // Get or create session options for this pipeline model
    OrtSessionOptions* session_options = nullptr;
    
    if (pipeline_model.run_on_cpu) {
      // For CPU execution, create minimal session options with CPU provider
      session_options = model.session_options_.get();  // Use default session options which will fall back to CPU
    } else {
      // For non-CPU execution (e.g., QNN), use the model-specific session options
      session_options = model.GetSessionOptions(pipeline_model.model_id);
    }
    
    // Create session - pass just the filename
    auto session = const_cast<DecoderOnlyPipelineModel*>(decoder_model)->CreateSession(
        const_cast<OrtEnv&>(decoder_model->ort_env_), 
        model_filename, 
        session_options);
    
    
    // Store session based on order: patch_embed, vision_attn, patch_merger
    if (!patch_embed_session_) {
      patch_embed_session_ = std::move(session);
    } else if (!vision_attn_session_) {
      vision_attn_session_ = std::move(session);
    } else if (!patch_merger_session_) {
      patch_merger_session_ = std::move(session);
    }
  }
  
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
  // Stage 1: Patch Embed
  RunPatchEmbed(pixel_values);
  
  // Stage 2: Apply window indexing reordering  
  if (!window_indices_.empty()) {
    ApplyWindowIndexing();
  } else {
    reordered_patches_ = std::move(patch_embed_output_);
  }
  
  // Stage 3: Vision Attention
  RunVisionAttention();
  
  // Stage 4: Skip reverse window indexing - patch merger handles the reduction
  if (!window_indices_.empty()) {
    // Pass the 1972-patch sequence directly to patch merger
    restored_patches_ = std::move(vision_attn_output_);
  } else {
    restored_patches_ = std::move(vision_attn_output_);
  }
  
  // Stage 5: Patch Merger
  RunPatchMerger();
  
  return std::move(final_embeddings_);
}

void VisionPipelineState::RunPatchEmbed(OrtValue* pixel_values) {
  const auto& pipeline_config = model_.config_->model.vision.pipeline[0];
  
  // Log pixel_values statistics BEFORE patch embed
  {
    auto input_shape_info = pixel_values->GetTensorTypeAndShapeInfo();
    auto input_shape = input_shape_info->GetShape();
    const float* input_data = pixel_values->GetTensorData<float>();
    size_t input_total = 1;
    for (auto dim : input_shape) input_total *= dim;
    
    float input_min = input_data[0], input_max = input_data[0], input_sum = 0.0f;
    for (size_t i = 0; i < input_total; ++i) {
      input_min = std::min(input_min, input_data[i]);
      input_max = std::max(input_max, input_data[i]);
      input_sum += input_data[i];
    }
    float input_mean = input_sum / input_total;
    (void)input_mean; // silence unused variable warning
  }
  
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
  
  for (size_t i = 0; i < shape.size(); ++i) {
  }
  
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
  
  // Log statistics
  auto final_shape_info = patch_embed_output_->GetTensorTypeAndShapeInfo();
  auto final_shape = final_shape_info->GetShape();
  const float* data = patch_embed_output_->GetTensorData<float>();
  size_t total_elements = 1;
  for (auto dim : final_shape) total_elements *= dim;
  
  float min_val = data[0], max_val = data[0], sum = 0.0f;
  for (size_t i = 0; i < total_elements; ++i) {
    min_val = std::min(min_val, data[i]);
    max_val = std::max(max_val, data[i]);
    sum += data[i];
  }
  float mean = sum / total_elements;
  (void)mean; // silence unused variable warning
}

void VisionPipelineState::ApplyWindowIndexing() {
  // Get shape of patch_embed_output: [batch, seq_len, hidden_dim]
  auto shape_info = patch_embed_output_->GetTensorTypeAndShapeInfo();
  auto shape = shape_info->GetShape();
  
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
    
  // CRITICAL: Window indices work on GROUPS of patches, not individual patches!
  // Python logic: hidden.reshape((493, 4, 1280))[wnd_idx].reshape((1972, 1280))
  // This selects which groups of 4 patches to use, then flattens back
  
  int64_t patches_per_group = spatial_merge_size_ * spatial_merge_size_;  // 4
  int64_t num_groups = seq_len / patches_per_group;  // 1972 / 4 = 493
    
  if (static_cast<int64_t>(window_indices_.size()) != num_groups) {
    throw std::runtime_error("window_indices size (" + std::to_string(window_indices_.size()) + 
                           ") must equal num_groups (" + std::to_string(num_groups) + ")");
  }
  
  const float* input_data = patch_embed_output_->GetTensorData<float>();
  
  // Create output tensor with same shape
  std::vector<int64_t> output_shape = {batch_size, seq_len, hidden_dim};
  reordered_patches_ = OrtValue::CreateTensor<float>(*allocator_cpu_, std::span<const int64_t>(output_shape));
  float* output_data = reordered_patches_->GetTensorMutableData<float>();
  
  // For each batch
  for (int64_t b = 0; b < batch_size; ++b) {
    // For each group in the OUTPUT
    for (int64_t out_group_idx = 0; out_group_idx < num_groups; ++out_group_idx) {
      // window_indices_[out_group_idx] tells us which INPUT group to copy from
      int64_t in_group_idx = window_indices_[out_group_idx];
      
      // Bounds check
      if (in_group_idx < 0 || in_group_idx >= num_groups) {
        throw std::runtime_error("Window index out of bounds: " + std::to_string(in_group_idx) + 
                               " for group " + std::to_string(out_group_idx));
      }
      
      // Copy all patches in this group (patches_per_group = 4 patches)
      for (int64_t patch_in_group = 0; patch_in_group < patches_per_group; ++patch_in_group) {
        int64_t in_patch_idx = in_group_idx * patches_per_group + patch_in_group;
        int64_t out_patch_idx = out_group_idx * patches_per_group + patch_in_group;
        
        // Copy hidden_dim values
        std::memcpy(&output_data[b * seq_len * hidden_dim + out_patch_idx * hidden_dim],
                    &input_data[b * seq_len * hidden_dim + in_patch_idx * hidden_dim],
                    hidden_dim * sizeof(float));
      }
    }
  }
  
  // Log statistics of reordered output
  const float* output = reordered_patches_->GetTensorData<float>();
  size_t total_elements = batch_size * seq_len * hidden_dim;
  float min_val = output[0], max_val = output[0], sum = 0.0f;
  for (size_t i = 0; i < total_elements; ++i) {
    min_val = std::min(min_val, output[i]);
    max_val = std::max(max_val, output[i]);
    sum += output[i];
  }
  float mean = sum / total_elements;
  (void)mean; // silence unused variable warning
}

void VisionPipelineState::RunVisionAttention() {
  const auto& pipeline_config = model_.config_->model.vision.pipeline[1];
  
  // Get shape of reordered_patches: [batch, seq_len, hidden_dim]
  auto shape_info = reordered_patches_->GetTensorTypeAndShapeInfo();
  auto shape = shape_info->GetShape();
  
  // Vision attention model expects 2D input [seq_len, hidden_dim], so squeeze batch dimension
  int64_t batch_size = shape[0];
  int64_t seq_len = shape[1];
  int64_t hidden_dim = shape[2];
  
  std::unique_ptr<OrtValue> input_2d;
  if (batch_size == 1 && shape.size() == 3) {
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
  for (size_t i = 0; i < session_output_names.size(); ++i) {
    output_names.push_back(session_output_names[i].c_str());
  }
  
  std::vector<OrtValue*> outputs(output_names.size(), nullptr);
  vision_attn_session_->Run(nullptr, input_names.data(), input_values.data(), input_names.size(),
                            output_names.data(), outputs.data(), output_names.size());
  
  vision_attn_output_ = std::unique_ptr<OrtValue>(outputs[0]);
  
  // Log vision attention output statistics BEFORE reshaping
  {
    auto temp_shape_info = vision_attn_output_->GetTensorTypeAndShapeInfo();
    auto temp_shape = temp_shape_info->GetShape();
    const float* attn_data = vision_attn_output_->GetTensorData<float>();
    size_t attn_total = 1;
    for (auto dim : temp_shape) attn_total *= dim;
    
    float attn_min = attn_data[0], attn_max = attn_data[0], attn_sum = 0.0f;
    for (size_t i = 0; i < attn_total; ++i) {
      attn_min = std::min(attn_min, attn_data[i]);
      attn_max = std::max(attn_max, attn_data[i]);
      attn_sum += attn_data[i];
    }
    float attn_mean = attn_sum / attn_total;
    (void)attn_mean; // silence unused variable warning
  }
  
  // Check output shape and reshape back to 3D if needed
  auto output_shape_info = vision_attn_output_->GetTensorTypeAndShapeInfo();
  auto output_shape = output_shape_info->GetShape();
  for (size_t i = 0; i < output_shape.size(); ++i) {
  }
  
  // If output is 2D [seq_len, hidden_dim], add batch dimension back
  if (output_shape.size() == 2) {
    int64_t out_seq_len = output_shape[0];
    int64_t out_hidden_dim = output_shape[1];
    std::vector<int64_t> new_shape = {1, out_seq_len, out_hidden_dim};
    
    auto reshaped = OrtValue::CreateTensor<float>(*allocator_cpu_, std::span<const int64_t>(new_shape));
    const float* src_data = vision_attn_output_->GetTensorData<float>();
    float* dst_data = reshaped->GetTensorMutableData<float>();
    std::memcpy(dst_data, src_data, out_seq_len * out_hidden_dim * sizeof(float));
    
    vision_attn_output_ = std::move(reshaped);
  }
  
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
  const auto& pipeline_config = model_.config_->model.vision.pipeline[2];
  
  // Get shape of restored_patches: [batch, seq_len, hidden_dim]
  auto shape_info = restored_patches_->GetTensorTypeAndShapeInfo();
  auto shape = shape_info->GetShape();
  
  // Patch merger model expects 2D input [seq_len, hidden_dim], so squeeze batch dimension
  int64_t batch_size = shape[0];
  int64_t seq_len = shape[1];
  int64_t hidden_dim = shape[2];
  
  std::unique_ptr<OrtValue> input_2d;
  if (batch_size == 1 && shape.size() == 3) {
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
  
  std::vector<OrtValue*> outputs(output_names.size(), nullptr);
  patch_merger_session_->Run(nullptr, input_names.data(), input_values.data(), input_names.size(),
                             output_names.data(), outputs.data(), output_names.size());
  
  auto merger_output = std::unique_ptr<OrtValue>(outputs[0]);
  
  // Check output shape
  auto output_shape_info = merger_output->GetTensorTypeAndShapeInfo();
  auto output_shape = output_shape_info->GetShape();
  for (size_t i = 0; i < output_shape.size(); ++i) {
  }
  
  // Log statistics before reverse indexing
  const float* merger_data = merger_output->GetTensorData<float>();
  size_t merger_total = 1;
  for (auto dim : output_shape) merger_total *= dim;
  
  float merger_min = merger_data[0], merger_max = merger_data[0], merger_sum = 0.0f;
  for (size_t i = 0; i < merger_total; ++i) {
    merger_min = std::min(merger_min, merger_data[i]);
    merger_max = std::max(merger_max, merger_data[i]);
    merger_sum += merger_data[i];
  }
  float merger_mean = merger_sum / merger_total;
  (void)merger_mean; // silence unused variable warning

  // RE-ENABLED: Reverse indexing after patch merger (baseline DOES use it: merged[rev])
  // Apply reverse window indexing to restore original spatial order
  
  size_t num_embeddings = static_cast<size_t>(output_shape[0]);
  size_t embedding_dim = static_cast<size_t>(output_shape[1]);
  
  // Create reversed embeddings tensor
  final_embeddings_ = OrtValue::CreateTensor<float>(*allocator_cpu_, 
                                                     std::span<const int64_t>(output_shape));
  
  const float* merger_data_src = merger_output->GetTensorData<float>();
  auto* final_data = final_embeddings_->GetTensorMutableData<float>();
  
  // Apply reverse indexing: rev = np.argsort(wnd_idx)
  // For each position in final output, copy from the position indicated by reverse_indices_
  for (size_t i = 0; i < num_embeddings && i < reverse_indices_.size(); ++i) {
    size_t src_idx = reverse_indices_[i];
    if (src_idx < num_embeddings) {
      std::memcpy(final_data + i * embedding_dim,
                  merger_data_src + src_idx * embedding_dim,
                  embedding_dim * sizeof(float));
    }
  }
  
  
  // Print statistics for comparison with baseline
  auto* data = final_embeddings_->GetTensorMutableData<float>();
  auto stats_shape_info = final_embeddings_->GetTensorTypeAndShapeInfo();
  auto stats_shape = stats_shape_info->GetShape();
  size_t total_elements = stats_shape[0] * stats_shape[1];
  size_t embed_dim = stats_shape[1];
  
  float min_val = data[0], max_val = data[0], sum = 0.0f;
  for (size_t i = 0; i < total_elements; ++i) {
    float val = data[i];
    min_val = std::min(min_val, val);
    max_val = std::max(max_val, val);
    sum += val;
  }
  float mean = sum / total_elements;
  
  float variance_sum = 0.0f;
  for (size_t i = 0; i < total_elements; ++i) {
    float diff = data[i] - mean;
    variance_sum += diff * diff;
  }
  float std_dev = std::sqrt(variance_sum / total_elements);
  (void)std_dev; // silence unused variable warning
  
  for (int i = 0; i < 10 && i < static_cast<int>(embed_dim); ++i) {
  }
  size_t last_offset = (stats_shape[0] - 1) * embed_dim;
  (void)last_offset; // silence unused variable warning
  for (int i = 0; i < 10 && i < static_cast<int>(embed_dim); ++i) {
  }
  
}

DeviceSpan<float> VisionPipelineState::Run(int total_length, DeviceSpan<int32_t>& next_tokens, DeviceSpan<int32_t> next_indices) {
  // This method is not used for vision pipeline - ProcessImage is the main entry point
  throw std::runtime_error("VisionPipelineState::Run should not be called directly");
}

}  // namespace Generators
