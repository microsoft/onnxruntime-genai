// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "../generators.h"
#include "../logging.h"
#include "vision_pipeline.h"

namespace Generators {

VisionPipelineModel::VisionPipelineModel(std::unique_ptr<Config> config, OrtEnv& ort_env)
    : Model{std::move(config)}, ort_env_{ort_env} {
  
  // Load window indexing if configured
  if (config_->model.vision.window_indexing.has_value()) {
    const auto& win_idx_config = config_->model.vision.window_indexing.value();
    auto win_idx_path = config_->config_path / fs::path(win_idx_config.filename);
    
    // Load window indexing array from .npy file
    auto file = win_idx_path.open(std::ios::binary);
    if (!file) {
      throw std::runtime_error("Failed to open window indexing file: " + win_idx_path.string());
    }
    
    // Simple NPY parser (assumes int64 dtype, little endian)
    // Skip magic string and version
    char header[10];
    file.read(header, 10);
    
    // Read header length
    uint16_t header_len;
    file.read(reinterpret_cast<char*>(&header_len), 2);
    
    // Read and parse header dict
    std::vector<char> header_dict(header_len);
    file.read(header_dict.data(), header_len);
    
    // Read data
    std::vector<int64_t> indices;
    int64_t value;
    while (file.read(reinterpret_cast<char*>(&value), sizeof(int64_t))) {
      indices.push_back(value);
    }
    
    window_indices_ = std::move(indices);
    spatial_merge_size_ = win_idx_config.spatial_merge_size;
  }
  
  // Load vision pipeline sessions
  for (const auto& model : config_->model.vision.pipeline) {
    auto session_options = OrtSessionOptions::Create();
    CreateSessionOptionsFromConfig(
        model.session_options.has_value() ? model.session_options.value() : config_->model.decoder.session_options,
        *session_options, true, true);
    
    sessions_.emplace_back(CreateSession(ort_env, model.filename, session_options.get()));
    session_options_.emplace_back(std::move(session_options));
  }

  for (auto& session : sessions_) {
    session_info_.Add(*session);
  }
}

std::unique_ptr<State> VisionPipelineModel::CreateState(DeviceSpan<int32_t> sequence_lengths, const GeneratorParams& params) const {
  return std::make_unique<VisionPipelineState>(*this, params);
}

VisionPipelineState::VisionPipelineState(const VisionPipelineModel& model, const GeneratorParams& params)
    : State{params, model},
      model_{model} {}

void VisionPipelineState::SetExtraInputs(const std::vector<ExtraInput>& extra_inputs, 
                                         int64_t num_images, int64_t num_image_tokens) {
  num_image_tokens_ = num_image_tokens;
  num_images_ = num_images;

  // Add inputs for first pipeline stage (pixel_values, etc.)
  extra_inputs_.Add(extra_inputs, model_.sessions_[0]->GetInputNames());
}

DeviceSpan<float> VisionPipelineState::Run(int current_length, DeviceSpan<int32_t>& next_tokens,
                                           DeviceSpan<int32_t> next_indices) {
  // Multi-stage vision pipeline execution
  OrtValue* intermediate_output = nullptr;
  
  for (size_t stage = 0; stage < model_.sessions_.size(); ++stage) {
    const auto& pipeline_config = model_.config_->model.vision.pipeline[stage];
    
    // Set stage-specific run options if configured
    if (pipeline_config.run_options.has_value()) {
      State::SetRunOptions(pipeline_config.run_options.value());
    }
    
    // For stages after the first, set up inputs from previous stage output
    if (stage > 0 && intermediate_output != nullptr) {
      // Get input name for this stage
      const char* input_name = pipeline_config.inputs.empty() 
                             ? "hidden" 
                             : pipeline_config.inputs[0].c_str();
      
      // Store in ortvalue_store_ for this stage
      auto key = std::string("stage_") + std::to_string(stage) + "_" + input_name;
      ortvalue_store_[key] = std::unique_ptr<OrtValue>(intermediate_output);
      
      // Find the input slot and set it
      for (size_t i = 0; i < input_names_.size(); ++i) {
        if (std::strcmp(input_names_[i], input_name) == 0) {
          inputs_[i] = ortvalue_store_[key].get();
          break;
        }
      }
    }
    
    // Run the stage
    State::Run(*model_.sessions_[stage]);
    
    // Get the output for next stage or final result
    const char* output_name = pipeline_config.outputs.empty() 
                            ? "hidden_states" 
                            : pipeline_config.outputs[0].c_str();
    
    for (size_t i = 0; i < output_names_.size(); ++i) {
      if (std::strcmp(output_names_[i], output_name) == 0) {
        intermediate_output = outputs_[i];
        break;
      }
    }
    
    // Apply window indexing transform if needed
    if (stage == 0 && !model_.window_indices_.empty()) {
      // Apply window indexing after patch_embed, before vision_attn
      intermediate_output = ApplyWindowIndexingTransform(intermediate_output, false);
    } else if (stage == model_.sessions_.size() - 1 && !model_.window_indices_.empty()) {
      // Restore original order after final stage
      intermediate_output = ApplyWindowIndexingTransform(intermediate_output, true);
    }
  }
  
  // Store final output as image_features
  if (intermediate_output) {
    image_features_ = std::unique_ptr<OrtValue>(intermediate_output);
  }
  
  return {};
}

OrtValue* VisionPipelineState::ApplyWindowIndexingTransform(OrtValue* input_tensor, bool restore_order) {
  // Window indexing transform: reshape and reorder patches
  
  if (!input_tensor) {
    throw std::runtime_error("Input tensor for window indexing is null");
  }
  
  // Get tensor info
  auto type_info = input_tensor->GetTensorTypeAndShapeInfo();
  auto shape = type_info->GetShape();
  
  // Expected shape: [seq_len, hidden_dim] for 2D or [batch, seq_len, hidden_dim] for 3D
  if (shape.size() < 2) {
    throw std::runtime_error("Window indexing expects at least 2D tensor, got " + std::to_string(shape.size()) + "D");
  }
  
  int64_t seq_len = shape.size() == 2 ? shape[0] : shape[1];
  int64_t hidden_dim = shape.size() == 2 ? shape[1] : shape[2];
  
  // Get CPU pointer to data (assuming CPU execution for this transform)
  const float* input_data = input_tensor->GetTensorData<float>();
  
  int64_t spatial_merge_area = model_.spatial_merge_size_ * model_.spatial_merge_size_;
  int64_t num_windows = seq_len / spatial_merge_area;
  
  // Create output tensor with same shape
  auto output_tensor = OrtValue::CreateTensor<float>(
      model_.allocator_cpu_,
      std::span<const int64_t>(shape)
  );
  float* output_data = output_tensor->GetTensorMutableData<float>();
  
  if (restore_order) {
    // Restore original order using argsort of window_indices_
    std::vector<int64_t> reverse_indices(model_.window_indices_.size());
    for (size_t i = 0; i < model_.window_indices_.size(); ++i) {
      reverse_indices[model_.window_indices_[i]] = i;
    }
    
    // Apply reverse indexing: output[reverse_indices[i]] = input[i]
    for (int64_t win = 0; win < num_windows; ++win) {
      int64_t src_win = reverse_indices[win];
      for (int64_t patch = 0; patch < spatial_merge_area; ++patch) {
        int64_t src_idx = (src_win * spatial_merge_area + patch) * hidden_dim;
        int64_t dst_idx = (win * spatial_merge_area + patch) * hidden_dim;
        std::copy_n(input_data + src_idx, hidden_dim, output_data + dst_idx);
      }
    }
  } else {
    // Apply forward window indexing
    // Reshape [seq_len, hidden] -> [num_windows, spatial_merge_area, hidden]
    // Then reorder: output[window_indices_[i]] = input[i]
    for (int64_t win = 0; win < num_windows; ++win) {
      int64_t dst_win = model_.window_indices_[win];
      for (int64_t patch = 0; patch < spatial_merge_area; ++patch) {
        int64_t src_idx = (win * spatial_merge_area + patch) * hidden_dim;
        int64_t dst_idx = (dst_win * spatial_merge_area + patch) * hidden_dim;
        std::copy_n(input_data + src_idx, hidden_dim, output_data + dst_idx);
      }
    }
  }
  
  // Store in intermediate buffer and return
  intermediate_tensor_ = std::move(output_tensor);
  return intermediate_tensor_.get();
}

OrtValue* VisionPipelineState::GetOutput(const char* name) {
  // Check if output is the final image_features
  if (std::strcmp(name, model_.config_->model.vision.outputs.image_features.c_str()) == 0) {
    return image_features_.get();
  }
  
  // Check if output is in ortvalue_store_
  auto it = ortvalue_store_.find(name);
  if (it != ortvalue_store_.end()) {
    return it->second.get();
  }
  
  return State::GetOutput(name);
}

}  // namespace Generators
