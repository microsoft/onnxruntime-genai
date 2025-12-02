#include "qwen_vl_model.h"
#include "model.h"
#include "onnxruntime_api.h"
#include "../logging.h"
#include <iostream>
#include <cstring>

namespace Generators {

Qwen2_5_VL_PipelineModel::Qwen2_5_VL_PipelineModel(std::unique_ptr<Config> config, OrtEnv& ort_env)
  : DecoderOnlyPipelineModel(std::move(config), ort_env) {  
  // Build vision pipeline if config provides vision pipeline stages
  if (!config_->model.vision.pipeline.empty() && config_->model.vision.window_indexing.has_value()) {
    // Expect identifiers patch_embed, vision_attn, patch_merger
    std::string patch_embed_path, vision_attn_path, patch_merger_path;
    for (const auto& stage : config_->model.vision.pipeline) {
      if (stage.model_id == "patch_embed") patch_embed_path = (config_->config_path / fs::path(stage.filename)).string();
      else if (stage.model_id == "vision_attn") vision_attn_path = (config_->config_path / fs::path(stage.filename)).string();
      else if (stage.model_id == "patch_merger") patch_merger_path = (config_->config_path / fs::path(stage.filename)).string();
    }
    if (!patch_embed_path.empty() && !vision_attn_path.empty() && !patch_merger_path.empty()) {
      auto wnd_idx_path = (config_->config_path / fs::path(config_->model.vision.window_indexing->filename)).string();
      int spatial_merge = config_->model.vision.window_indexing->spatial_merge_size;
      // For now, rely on run_on_cpu flag of vision_attn stage to decide QNN usage
      bool use_qnn_attn = false;
      for (const auto& stage : config_->model.vision.pipeline) {
        if (stage.model_id == "vision_attn" && !stage.run_on_cpu) {
          use_qnn_attn = true; break;
        }
      }      
      vision_pipeline_ = std::make_unique<QwenVisionPipeline>(ort_env,
                                                              patch_embed_path,
                                                              vision_attn_path,
                                                              patch_merger_path,
                                                              spatial_merge,
                                                              wnd_idx_path,
                                                              use_qnn_attn);
    } else {
      std::cout << "[GENAI VISION] WARNING: Missing vision model paths!" << std::endl;
    }
  } else {
    std::cout << "[GENAI VISION] No vision pipeline config found" << std::endl;
  }
}

std::unique_ptr<State> Qwen2_5_VL_PipelineModel::CreateState(DeviceSpan<int32_t> sequence_lengths,
                                                             const GeneratorParams& params) const {
  return std::make_unique<Qwen2_5_VL_PipelineState>(*this, sequence_lengths, params);
}

Qwen2_5_VL_PipelineState::Qwen2_5_VL_PipelineState(const Qwen2_5_VL_PipelineModel& model,
                                                   DeviceSpan<int32_t> sequence_lengths,
                                                   const GeneratorParams& params)
  : DecoderOnlyPipelineState(model, sequence_lengths, params), vl_model_{model} {
}

void Qwen2_5_VL_PipelineState::SetExtraInputs(const std::vector<ExtraInput>& extra_inputs) {  
  // Let base register provided extra inputs first
  DecoderOnlyPipelineState::SetExtraInputs(extra_inputs);
  
  if (vision_ran_) {
    return;
  }
  
  if (!vl_model_.vision_pipeline_) {
    return;
  }

  // Find pixel_values input among the extra inputs passed to this function
  OrtValue* pixel_values_val = nullptr;
  const std::string pixel_name = vl_model_.config_->model.vision.inputs.pixel_values;
  
  for (size_t i = 0; i < extra_inputs.size(); ++i) {
    if (extra_inputs[i].name == pixel_name) {
      pixel_values_val = extra_inputs[i].tensor->GetOrtTensor();
      break;
    }
  }
  
  if (!pixel_values_val) {
    return;
  }

  auto pixel_info = pixel_values_val->GetTensorTypeAndShapeInfo();
  auto pixel_shape = pixel_info->GetShape();
  std::vector<int64_t> pixel_shape_vec(pixel_shape.begin(), pixel_shape.end());
  
  size_t pixel_count = 1;
  for (auto d : pixel_shape_vec) pixel_count *= static_cast<size_t>(d);
  // Get pointer - but don't access it yet to avoid crash
  const float* pixel_data = nullptr;
  try {
    pixel_data = pixel_values_val->GetTensorMutableData<float>();
  } catch (const std::exception& ex) {
    std::cout << "[GENAI VISION] ERROR: Failed to get pixel data pointer: " << ex.what() << std::endl;
    return;
  }
  
  if (!pixel_data) {
    return;
  }

  // Run vision pipeline
  try {
    image_features_buffer_ = vl_model_.vision_pipeline_->Run(pixel_data, pixel_shape_vec);
  } catch (const std::exception& ex) {
    std::cout << "[GENAI VISION] ERROR: Vision pipeline run failed: " << ex.what() << std::endl;
    return;
  }

  auto out_shape = vl_model_.vision_pipeline_->GetLastOutputShape(); // [seq_len, hidden]
  if (out_shape.size() != 2) {
    return;
  }
  
  // Debug: Log vision embeddings statistics
  float min_feat = image_features_buffer_[0], max_feat = image_features_buffer_[0], sum_feat = 0.0f;
  for (const auto& val : image_features_buffer_) {
    min_feat = std::min(min_feat, val);
    max_feat = std::max(max_feat, val);
    sum_feat += val;
  }
  float mean_feat = sum_feat / image_features_buffer_.size();
  float sum_sq_diff = 0.0f;
  for (const auto& val : image_features_buffer_) {
    float diff = val - mean_feat;
    sum_sq_diff += diff * diff;
  }
  size_t count = static_cast<size_t>(out_shape[0]) * static_cast<size_t>(out_shape[1]);
  auto mem_info = OrtMemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);
  // API expects spans (data, shape)
  std::span<float> data_span(image_features_buffer_.data(), count);
  std::span<const int64_t> shape_span(out_shape.data(), out_shape.size());
  image_features_value_ = OrtValue::CreateTensor<float>(*mem_info, data_span, shape_span);

  vision_ran_ = true;
}

void Qwen2_5_VL_PipelineState::OnStageComplete(size_t stage_id, DeviceSpan<int32_t>& next_tokens) {
  // After embeddings stage (stage 0) completes, inject vision embeddings at image token positions
  if (stage_id == 0 && vision_ran_) {    
    // Find embeddings output name from config
    const auto& embeddings_config = vl_model_.config_->model.decoder.pipeline[0];
    if (!embeddings_config.outputs.empty()) {
      const std::string& embeddings_output_name = embeddings_config.outputs[0];      
      InjectVisionEmbeddings(embeddings_output_name, next_tokens);
    }
  }
}

void Qwen2_5_VL_PipelineState::InjectVisionEmbeddings(const std::string& embeddings_output_name,
                                                     DeviceSpan<int32_t>& input_token_ids) {  
  // Get image_token_id from config
  const int32_t image_token_id = vl_model_.config_->model.image_token_id;
  
  // Get embeddings output from ortvalue_store_
  auto it = ortvalue_store_.find(embeddings_output_name);
  if (it == ortvalue_store_.end()) {
    return;
  }
  
  OrtValue* embeddings_ortvalue = it->second.get();
  if (!embeddings_ortvalue) {
    return;
  }
  
  //Get tensor info
  auto type_info = embeddings_ortvalue->GetTensorTypeAndShapeInfo();
  auto shape = type_info->GetShape();
  float* embeddings_data = embeddings_ortvalue->GetTensorMutableData<float>();
  
  const int64_t embedding_dim = shape[2];
  
  // Get vision embeddings info
  auto vision_type_info = image_features_value_->GetTensorTypeAndShapeInfo();
  auto vision_shape = vision_type_info->GetShape();
  const float* vision_data = image_features_value_->GetTensorData<float>();
  
  const int64_t num_vision_tokens = vision_shape[0];
  const int64_t vision_dim = vision_shape[1];
  
  if (vision_dim != embedding_dim) {
    return;
  }
  
  // Get input_ids from the base class member
  if (!input_ids_ || !input_ids_->Get()) {
    return;
  }
  
  OrtValue* input_ids_ortvalue = input_ids_->Get();
  auto input_ids_type_info = input_ids_ortvalue->GetTensorTypeAndShapeInfo();
  auto input_ids_shape = input_ids_type_info->GetShape();
  const int32_t* token_ids_cpu = input_ids_ortvalue->GetTensorData<int32_t>();
  
  // Log input token IDs
  int64_t total_tokens = 1;
  for (auto dim : input_ids_shape) total_tokens *= dim;
  // std::cout << "[GENAI EMB INPUT] Token IDs count: " << total_tokens << std::endl;
  // std::cout << "[GENAI EMB INPUT] First 20 token IDs: ";
  // for (int i = 0; i < std::min(20LL, total_tokens); ++i) {
  //   std::cout << token_ids_cpu[i] << " ";
  // }
  // std::cout << std::endl;
  // std::cout << "[GENAI EMB INPUT] Last 20 token IDs: ";
  // int64_t start_idx = total_tokens > 20 ? total_tokens - 20 : 0;
  // for (int64_t i = start_idx; i < total_tokens; ++i) {
  //   std::cout << token_ids_cpu[i] << " ";
  // }
  // std::cout << std::endl;
  
//   std::cout << "[GENAI INJECT] input_ids shape: [";
  // for (size_t i = 0; i < input_ids_shape.size(); ++i) {
  //   std::cout << input_ids_shape[i];
  //   if (i < input_ids_shape.size() - 1) std::cout << ", ";
  // }
//   std::cout << "]" << std::endl;
  
  // Print first few token IDs for debugging
//   std::cout << "[GENAI INJECT] First 20 token IDs: ";
  // for (int i = 0; i < std::min(20LL, total_tokens); ++i) {
  //   std::cout << token_ids_cpu[i] << " ";
  // }
  // std::cout << std::endl;
  
  size_t num_image_tokens_in_chunk = 0;
  
  // // Iterate through input_ids to find image token positions
  // size_t num_image_tokens_found = 0;
  // for (int64_t i = 0; i < total_tokens; ++i) {
  //   if (token_ids_cpu[i] == image_token_id) {
  //     num_image_tokens_found++;
  //   }
  // }
  
  for (int64_t i = 0; i < total_tokens; ++i) {
    if (token_ids_cpu[i] == image_token_id) {
      // Found image token position - replace with vision embedding
      if (image_embed_consumed_ < static_cast<size_t>(num_vision_tokens)) {
        const float* src_vision_embedding = vision_data + (image_embed_consumed_ * vision_dim);
        
        // Map from input_ids position to embeddings position
        // Embeddings shape is [batch, seq_len, embedding_dim]
        // input_ids shape could be [batch, seq_len] or [seq_len]
        int64_t embed_idx = i;
        if (shape.size() == 3 && input_ids_shape.size() == 2) {
          // If embeddings has batch dimension but we're in flattened input_ids
          embed_idx = i;  // Assume batch=1, just use linear index
        }
        float* dst_text_embedding = embeddings_data + (embed_idx * embedding_dim);
        
        // Debug: Print first injection
        // if (num_image_tokens_in_chunk == 0) {
        // //   std::cout << "[GENAI INJECT] First injection: position " << i << " in input_ids, embedding index " << embed_idx << std::endl;
        // //   std::cout << "[GENAI INJECT] Vision embedding [0-5]: ";
        //   for (int k = 0; k < 5; ++k) std::cout << src_vision_embedding[k] << " ";
        //   std::cout << std::endl;
        // //   std::cout << "[GENAI INJECT] Original text embedding [0-5]: ";
        //   for (int k = 0; k < 5; ++k) std::cout << dst_text_embedding[k] << " ";
        //   std::cout << std::endl;
        // }
        
        // Copy vision embedding to this position
        std::memcpy(dst_text_embedding, src_vision_embedding, vision_dim * sizeof(float));
        
        // Verify the write
        // if (num_image_tokens_in_chunk == 0) {
        // //   std::cout << "[GENAI INJECT] After copy, embedding [0-5]: ";
        //   for (int k = 0; k < 5; ++k) std::cout << dst_text_embedding[k] << " ";
        //   std::cout << std::endl;
        // }
        
        num_image_tokens_in_chunk++;
        image_embed_consumed_++;
      } else {
        std::cout << "[GENAI INJECT] WARNING: More image tokens than vision embeddings!" << std::endl;
      }
    }
  }
  
//   std::cout << "[GENAI INJECT] Injected " << num_image_tokens_in_chunk << " vision embeddings at image token positions" << std::endl;
//   std::cout << "[GENAI INJECT] Total consumed: " << image_embed_consumed_ << " / " << num_vision_tokens << std::endl;
  
  // Verify embeddings after injection
  float min_after = std::numeric_limits<float>::max();
  float max_after = std::numeric_limits<float>::lowest();
  float sum_after = 0.0f;
  int64_t total_elems = 1;
  for (auto dim : shape) total_elems *= dim;
  for (int64_t i = 0; i < total_elems; ++i) {
    float val = embeddings_data[i];
    if (val < min_after) min_after = val;
    if (val > max_after) max_after = val;
    sum_after += val;
  }
  std::cout << "[GENAI INJECT] Embeddings AFTER injection: min=" << min_after << ", max=" << max_after << ", mean=" << (sum_after / total_elems) << std::endl;
  
  // Log embeddings AFTER injection with first 10 values
  std::cout << "[GENAI EMB AFTER INJECTION] Shape: [";
  for (size_t i = 0; i < shape.size(); ++i) {
    std::cout << shape[i];
    if (i < shape.size() - 1) std::cout << ", ";
  }
  std::cout << "]" << std::endl;
  std::cout << "[GENAI EMB AFTER INJECTION] Statistics: min=" << min_after << ", max=" << max_after << ", mean=" << (sum_after / total_elems) << std::endl;
  std::cout << "[GENAI EMB AFTER INJECTION] First 10 values: ";
  for (int i = 0; i < 10 && i < total_elems; ++i) {
    std::cout << embeddings_data[i] << " ";
  }
  std::cout << std::endl;
}

} // namespace Generators

