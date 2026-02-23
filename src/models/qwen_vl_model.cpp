#include "qwen_vl_model.h"
#include "model.h"
#include "onnxruntime_api.h"
#include "../logging.h"
#include <iostream>
#include <cstring>
#include <algorithm>

namespace Generators {

Qwen2_5_VL_PipelineModel::Qwen2_5_VL_PipelineModel(std::unique_ptr<Config> config, OrtEnv& ort_env)
    : DecoderOnlyPipelineModel(std::move(config), ort_env) {
  if (config_->model.vision.pipeline.empty()) return;

  // Find vision pipeline stage paths
  auto find_stage = [&](const std::string& id) -> std::string {
    for (const auto& stage : config_->model.vision.pipeline) {
      if (stage.model_id == id) return (config_->config_path / fs::path(stage.filename)).string();
    }
    return "";
  };

  auto patch_embed_path = find_stage("patch_embed");
  auto vision_attn_path = find_stage("vision_attn");
  auto patch_merger_path = find_stage("patch_merger");

  if (patch_embed_path.empty() || vision_attn_path.empty() || patch_merger_path.empty()) return;

  // Check if QNN should be used for vision attention
  bool use_qnn_attn = std::any_of(config_->model.vision.pipeline.begin(),
                                  config_->model.vision.pipeline.end(),
                                  [](const auto& stage) {
                                    return stage.model_id == "vision_attn" && !stage.run_on_cpu;
                                  });

  // Default spatial merge size
  constexpr int spatial_merge = 2;

  vision_pipeline_ = std::make_unique<QwenVisionPipeline>(
      ort_env, patch_embed_path, vision_attn_path, patch_merger_path,
      spatial_merge, use_qnn_attn);
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
  DecoderOnlyPipelineState::SetExtraInputs(extra_inputs);

  if (vision_ran_ || !vl_model_.vision_pipeline_) return;

  OrtValue* pixel_values_val = nullptr;
  OrtValue* image_grid_thw_val = nullptr;
  const auto& pixel_name = vl_model_.config_->model.vision.inputs.pixel_values;
  const auto& grid_thw_name = vl_model_.config_->model.vision.inputs.image_grid_thw;

  for (const auto& input : extra_inputs) {
    if (input.name == pixel_name) {
      pixel_values_val = input.tensor->GetOrtTensor();
    } else if (input.name == grid_thw_name) {
      image_grid_thw_val = input.tensor->GetOrtTensor();
    }
  }
  if (!pixel_values_val) {
    return;
  }

  auto pixel_type_info = pixel_values_val->GetTensorTypeAndShapeInfo();
  auto pixel_shape = pixel_type_info->GetShape();
  auto pixel_type = pixel_type_info->GetElementType();

  std::vector<int64_t> pixel_shape_vec(pixel_shape.begin(), pixel_shape.end());
  const float* pixel_data = nullptr;
  // Convert pixel values to float32 if needed (handles float16, bfloat16, float32)
  std::unique_ptr<OrtValue> pixel_values_fp32;

  if (pixel_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
    pixel_data = pixel_values_val->GetTensorData<float>();
  } else {
    // Use existing Cast() function to convert to float32
    Cast(*pixel_values_val, pixel_values_fp32, *vl_model_.p_device_inputs_, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT);
    pixel_data = pixel_values_fp32->GetTensorData<float>();
  }

  if (!pixel_data) {
    throw std::runtime_error("Vision pipeline: failed to access pixel_values tensor data");
  }

  // Extract grid_thw if provided - handle multiple images
  std::vector<std::vector<int64_t>> all_grid_thws;
  if (image_grid_thw_val) {
    auto grid_shape = image_grid_thw_val->GetTensorTypeAndShapeInfo()->GetShape();
    auto element_type = image_grid_thw_val->GetTensorTypeAndShapeInfo()->GetElementType();

    // grid_thw shape is [num_images, 3] where each row is [t, h, w]
    int64_t num_grid_images = 1;
    if (grid_shape.size() == 2) {
      num_grid_images = grid_shape[0];
    }

    if (element_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64) {
      const int64_t* grid_data = image_grid_thw_val->GetTensorData<int64_t>();
      for (int64_t i = 0; i < num_grid_images; ++i) {
        all_grid_thws.push_back({grid_data[i * 3], grid_data[i * 3 + 1], grid_data[i * 3 + 2]});
      }
    } else if (element_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32) {
      const int32_t* grid_data = image_grid_thw_val->GetTensorData<int32_t>();
      for (int64_t i = 0; i < num_grid_images; ++i) {
        all_grid_thws.push_back({static_cast<int64_t>(grid_data[i * 3]),
                                 static_cast<int64_t>(grid_data[i * 3 + 1]),
                                 static_cast<int64_t>(grid_data[i * 3 + 2])});
      }
    }
  }

  try {
    if (all_grid_thws.size() <= 1) {
      // Single image (or no grid info) - run vision pipeline once
      std::vector<int64_t> grid_thw;
      if (!all_grid_thws.empty()) {
        grid_thw = all_grid_thws[0];
      }
      image_features_buffer_ = vl_model_.vision_pipeline_->Run(pixel_data, pixel_shape_vec, grid_thw);
    } else {
      // Multiple images - run vision pipeline for each image and concatenate features
      image_features_buffer_.clear();
      int64_t total_merged_tokens = 0;
      int64_t merged_hidden = 0;
      size_t pixel_offset = 0;

      for (size_t img_idx = 0; img_idx < all_grid_thws.size(); ++img_idx) {
        const auto& grid_thw = all_grid_thws[img_idx];
        int64_t t = grid_thw[0];
        int64_t h = grid_thw[1];
        int64_t w = grid_thw[2];
        int64_t num_patches = t * h * w;

        // Extract this image's pixel data slice
        // pixel_values shape: [total_patches, patch_dim] or [1, total_patches, patch_dim]
        int64_t patch_dim = pixel_shape_vec.back();
        std::vector<int64_t> img_pixel_shape;
        if (pixel_shape_vec.size() == 3) {
          img_pixel_shape = {1, num_patches, patch_dim};
        } else {
          img_pixel_shape = {num_patches, patch_dim};
        }

        const float* img_pixel_data = pixel_data + pixel_offset;
        pixel_offset += num_patches * patch_dim;

        auto img_features = vl_model_.vision_pipeline_->Run(img_pixel_data, img_pixel_shape, grid_thw);
        auto img_out_shape = vl_model_.vision_pipeline_->GetLastOutputShape();

        if (img_out_shape.size() == 2) {
          total_merged_tokens += img_out_shape[0];
          merged_hidden = img_out_shape[1];
        }

        image_features_buffer_.insert(image_features_buffer_.end(), img_features.begin(), img_features.end());
      }
    }
  } catch (const std::exception& e) {
    throw std::runtime_error(std::string("Vision pipeline failed: ") + e.what());
  }

  // Compute overall output shape
  // For single image: use GetLastOutputShape directly
  // For multi-image: total_features = sum of per-image seq_lens, hidden_size stays the same
  int64_t total_features_tokens = static_cast<int64_t>(image_features_buffer_.size());
  auto last_out_shape = vl_model_.vision_pipeline_->GetLastOutputShape();
  if (last_out_shape.size() != 2) {
    throw std::runtime_error("Vision pipeline: expected output shape rank 2, got " + std::to_string(last_out_shape.size()));
  }
  int64_t hidden_size = last_out_shape[1];
  int64_t total_seq_len = total_features_tokens / hidden_size;
  std::vector<int64_t> out_shape = {total_seq_len, hidden_size};

  auto mem_info = OrtMemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);
  std::span<float> data_span(image_features_buffer_.data(), image_features_buffer_.size());
  std::span<const int64_t> shape_span(out_shape.data(), out_shape.size());
  image_features_value_ = OrtValue::CreateTensor<float>(*mem_info, data_span, shape_span);

  vision_ran_ = true;
}

void Qwen2_5_VL_PipelineState::OnStageComplete(size_t stage_id, DeviceSpan<int32_t>& next_tokens) {
  if (stage_id != 0 || !vision_ran_) return;

  const auto& embeddings_config = vl_model_.config_->model.decoder.pipeline[0];
  if (!embeddings_config.outputs.empty()) {
    InjectVisionEmbeddings(embeddings_config.outputs[0], next_tokens);
  }
}

void Qwen2_5_VL_PipelineState::InjectVisionEmbeddings(const std::string& embeddings_output_name,
                                                      DeviceSpan<int32_t>& input_token_ids) {
  auto it = ortvalue_store_.find(embeddings_output_name);
  if (it == ortvalue_store_.end() || !it->second) {
    throw std::runtime_error("Vision embedding injection: embeddings output '" + embeddings_output_name + "' not found in ortvalue_store");
  }

  OrtValue* embeddings_ortvalue = it->second.get();
  auto shape = embeddings_ortvalue->GetTensorTypeAndShapeInfo()->GetShape();
  float* embeddings_data = embeddings_ortvalue->GetTensorMutableData<float>();

  auto vision_shape = image_features_value_->GetTensorTypeAndShapeInfo()->GetShape();
  const float* vision_data = image_features_value_->GetTensorData<float>();

  const int64_t embedding_dim = shape[2];
  const int64_t num_vision_tokens = vision_shape[0];
  const int64_t vision_dim = vision_shape[1];
  if (vision_dim != embedding_dim) {
    throw std::runtime_error("Vision embedding injection: dimension mismatch - vision_dim=" + std::to_string(vision_dim) +
                             ", embedding_dim=" + std::to_string(embedding_dim));
  }

  constexpr int32_t image_token_id = 151655;

  if (!input_ids_ || !input_ids_->Get()) {
    throw std::runtime_error("Vision embedding injection: input_ids not available");
  }

  OrtValue* input_ids_ortvalue = input_ids_->Get();
  auto input_ids_shape = input_ids_ortvalue->GetTensorTypeAndShapeInfo()->GetShape();
  const int32_t* token_ids_cpu = input_ids_ortvalue->GetTensorData<int32_t>();

  int64_t total_tokens = 1;
  for (auto dim : input_ids_shape) total_tokens *= dim;

  for (int64_t i = 0; i < total_tokens; ++i) {
    if (token_ids_cpu[i] == image_token_id && image_embed_consumed_ < static_cast<size_t>(num_vision_tokens)) {
      std::memcpy(embeddings_data + (i * embedding_dim),
                  vision_data + (image_embed_consumed_ * vision_dim),
                  vision_dim * sizeof(float));
      image_embed_consumed_++;
    }
  }

  // Warn if there's a mismatch between image tokens and vision features
  if (image_embed_consumed_ != static_cast<size_t>(num_vision_tokens)) {
    Log("warning", "Vision embedding mismatch: consumed " + std::to_string(image_embed_consumed_) +
                       " of " + std::to_string(num_vision_tokens) + " available vision tokens. " +
                       "This may indicate a mismatch between the number of image placeholders in the prompt " +
                       "and the number of images provided.");
  }
}

}  // namespace Generators
