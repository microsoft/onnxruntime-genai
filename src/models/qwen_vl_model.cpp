#include "qwen_vl_model.h"
#include "model.h"
#include "onnxruntime_api.h"
#include "../logging.h"
#include <iostream>
#include <cstring>

namespace Generators {

Qwen2_5_VL_PipelineModel::Qwen2_5_VL_PipelineModel(std::unique_ptr<Config> config, OrtEnv& ort_env)
  : DecoderOnlyPipelineModel(std::move(config), ort_env) {  
  if (config_->model.vision.pipeline.empty() || !config_->model.vision.window_indexing.has_value()) return;

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
  bool use_qnn_attn = false;
  for (const auto& stage : config_->model.vision.pipeline) {
    if (stage.model_id == "vision_attn" && !stage.run_on_cpu) {
      use_qnn_attn = true;
      break;
    }
  }

  auto wnd_idx_path = (config_->config_path / fs::path(config_->model.vision.window_indexing->filename)).string();
  int spatial_merge = config_->model.vision.window_indexing->spatial_merge_size;
  
  vision_pipeline_ = std::make_unique<FaraVisionPipeline>(
    ort_env, patch_embed_path, vision_attn_path, patch_merger_path,
    spatial_merge, wnd_idx_path, use_qnn_attn);
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
  const auto& pixel_name = vl_model_.config_->model.vision.inputs.pixel_values;
  
  for (const auto& input : extra_inputs) {
    if (input.name == pixel_name) {
      pixel_values_val = input.tensor->GetOrtTensor();
      break;
    }
  }
  if (!pixel_values_val) return;

  auto pixel_shape = pixel_values_val->GetTensorTypeAndShapeInfo()->GetShape();
  std::vector<int64_t> pixel_shape_vec(pixel_shape.begin(), pixel_shape.end());
  const float* pixel_data = pixel_values_val->GetTensorMutableData<float>();
  if (!pixel_data) return;

  try {
    image_features_buffer_ = vl_model_.vision_pipeline_->Run(pixel_data, pixel_shape_vec);
  } catch (const std::exception&) {
    return;  // Silent failure - pipeline already logs errors
  }

  auto out_shape = vl_model_.vision_pipeline_->GetLastOutputShape();
  if (out_shape.size() != 2) return;
  
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
  if (it == ortvalue_store_.end() || !it->second) return;
  
  OrtValue* embeddings_ortvalue = it->second.get();
  auto shape = embeddings_ortvalue->GetTensorTypeAndShapeInfo()->GetShape();
  float* embeddings_data = embeddings_ortvalue->GetTensorMutableData<float>();
  
  auto vision_shape = image_features_value_->GetTensorTypeAndShapeInfo()->GetShape();
  const float* vision_data = image_features_value_->GetTensorData<float>();
  
  const int64_t embedding_dim = shape[2];
  const int64_t num_vision_tokens = vision_shape[0];
  const int64_t vision_dim = vision_shape[1];
  if (vision_dim != embedding_dim) return;
  
  const int32_t image_token_id = vl_model_.config_->model.image_token_id;
  
  if (!input_ids_ || !input_ids_->Get()) return;
  
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
}

} // namespace Generators

