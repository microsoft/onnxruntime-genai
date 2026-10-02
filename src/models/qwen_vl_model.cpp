#include "qwen_vl_model.h"
#include "model.h"
#include "onnxruntime_api.h"
#include "../logging.h"
#include <iostream>
#include <cstring>
#include <algorithm>

namespace Generators {

namespace {

// Creates a view of one image's slice of a tensor whose leading dimension is the image count.
// The result borrows source's buffer and must not outlive it.
std::unique_ptr<OrtValue> SliceLeadingImage(OrtValue& source, int64_t index) {
  auto info = source.GetTensorTypeAndShapeInfo();
  auto shape = info->GetShape();
  const auto type = info->GetElementType();
  int64_t per_image = 1;
  for (size_t i = 1; i < shape.size(); ++i) per_image *= shape[i];
  const size_t slice_bytes = static_cast<size_t>(per_image) * Ort::SizeOf(type);
  shape[0] = 1;
  auto* data = static_cast<uint8_t*>(source.GetTensorMutableRawData());
  return OrtValue::CreateTensor(source.GetTensorMemoryInfo(), data + static_cast<size_t>(index) * slice_bytes,
                                slice_bytes, shape, type);
}

// InjectVisionEmbeddings expects [num_image_tokens, hidden_size]. Encoders commonly emit a
// leading batch dimension, so collapse every leading dimension of size 1.
std::vector<int64_t> SqueezeToRank2(const std::vector<int64_t>& shape) {
  std::vector<int64_t> squeezed = shape;
  while (squeezed.size() > 2 && squeezed.front() == 1) {
    squeezed.erase(squeezed.begin());
  }
  if (squeezed.size() != 2) {
    std::string printable;
    for (auto dim : shape) {
      printable += (printable.empty() ? "" : ", ") + std::to_string(dim);
    }
    throw std::runtime_error("Vision encoder: expected image features of rank 2, got [" + printable + "]");
  }
  return squeezed;
}

}  // namespace

Qwen2_5_VL_PipelineModel::Qwen2_5_VL_PipelineModel(std::unique_ptr<Config> config, OrtEnv& ort_env)
    : DecoderOnlyPipelineModel(std::move(config), ort_env) {
  // This path runs a vision encoder in front of a pipelined decoder and creates no speech
  // session. A config that declares one would still load, because AppendEmbeddingFeatureInputs
  // binds an empty audio tensor for any unfilled modality, and would then answer as if the
  // audio were silent. Reject it rather than return a confidently wrong result.
  if (!config_->model.speech.filename.empty()) {
    throw std::runtime_error(
        "A pipelined decoder (decoder.pipeline) combined with a speech encoder "
        "(speech.filename) is not supported: this path runs vision only. Remove "
        "speech.filename, or export the decoder as a single graph so the multimodal "
        "model is used instead.");
  }

  if (config_->model.vision.pipeline.empty()) {
    // No three-stage vision pipeline configured. Models such as Gemma-4 export the
    // vision encoder as a single graph, so run it as one session.
    if (!config_->model.vision.filename.empty()) {
      vision_session_options_ = OrtSessionOptions::Create();
      // Deliberately does not fall back to the decoder's session options: in a pipelined
      // decoder those select the NPU, which cannot run an unquantized vision encoder.
      // Absent vision session options therefore mean default (CPU) placement.
      static const Config::SessionOptions kDefaultSessionOptions;
      CreateSessionOptionsFromConfig(
          config_->model.vision.session_options.has_value() ? *config_->model.vision.session_options
                                                            : kDefaultSessionOptions,
          *vision_session_options_, /*is_primary_session_options=*/false,
          /*disable_graph_capture=*/true);
      vision_session_ = CreateSession(ort_env, config_->model.vision.filename, vision_session_options_.get());
      // The multimodal processor resolves pixel_values / pixel_position_ids types through
      // session_info_, which the decoder pipeline populates with decoder sessions only.
      session_info_.Add(*vision_session_);
    }
    return;
  }

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

  OrtSessionOptions* vision_attn_so = nullptr;
  for (auto& stage : config_->model.vision.pipeline) {
    if (stage.model_id == "vision_attn" && !stage.run_on_cpu) {
      if (stage.session_options.has_value()) {
        auto emplaced = pipeline_session_options_.emplace("vision_attn", OrtSessionOptions::Create());
        CreateSessionOptionsFromConfig(*stage.session_options, *emplaced.first->second, false);
        vision_attn_so = emplaced.first->second.get();
      } else {
        // Fall back to primary session options when run_on_cpu=false but no stage-specific options
        vision_attn_so = session_options_.get();
      }
      break;
    }
  }

  int64_t spatial_merge = config_->model.vision.spatial_merge_size;
  int64_t patch_size = config_->model.vision.patch_size;
  int64_t window_size = config_->model.vision.window_size;

  vision_pipeline_ = std::make_unique<QwenVisionPipeline>(
      ort_env, patch_embed_path, vision_attn_path, patch_merger_path,
      spatial_merge, patch_size, window_size, vision_attn_so);
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
  // Vision must run before binding, because the embedding graph may declare
  // image_features as a real input and binding happens by name at Add() time.
  RunVision(extra_inputs);

  if (features_bound_) {
    // Already bound once. Mutating owned_extra_inputs_ now would move the strings whose
    // c_str() the state already holds, so forward the caller's vector untouched instead.
    DecoderOnlyPipelineState::SetExtraInputs(extra_inputs);
    return;
  }

  // ExtraInputs::Add stores name.c_str() and the raw tensor pointer, so the vector it reads
  // from must outlive this state and must not grow afterwards. Build it fully, bind once,
  // then leave it alone.
  owned_extra_inputs_ = extra_inputs;
  AppendEmbeddingFeatureInputs(owned_extra_inputs_);
  DecoderOnlyPipelineState::SetExtraInputs(owned_extra_inputs_);
}

DeviceSpan<float> Qwen2_5_VL_PipelineState::Run(int total_length, DeviceSpan<int32_t>& next_tokens,
                                                DeviceSpan<int32_t> next_indices) {
  EnsureFeatureInputsBound();
  return DecoderOnlyPipelineState::Run(total_length, next_tokens, next_indices);
}

// Text only requests never call SetExtraInputs, so bind the empty features here instead.
void Qwen2_5_VL_PipelineState::EnsureFeatureInputsBound() {
  if (features_bound_) return;

  AppendEmbeddingFeatureInputs(owned_extra_inputs_);
  if (!owned_extra_inputs_.empty()) {
    DecoderOnlyPipelineState::SetExtraInputs(owned_extra_inputs_);
  }
}

void Qwen2_5_VL_PipelineState::AppendEmbeddingFeatureInputs(std::vector<ExtraInput>& inputs) {
  if (features_bound_) return;
  const auto& image_name = vl_model_.config_->model.embedding.inputs.image_features;
  const auto& audio_name = vl_model_.config_->model.embedding.inputs.audio_features;

  auto already_bound = [&inputs](const std::string& name) {
    return std::any_of(inputs.begin(), inputs.end(),
                       [&name](const ExtraInput& input) { return input.name == name; });
  };

  auto mem_info = OrtMemoryInfo::Create("Cpu", OrtDeviceAllocator, 0, OrtMemTypeDefault);

  auto add = [&](const std::string& name, bool use_vision_output) {
    if (name.empty() || already_bound(name) || !vl_model_.session_info_.HasInput(name)) return;

    std::unique_ptr<OrtValue> value;
    if (use_vision_output && image_features_value_) {
      // Wrap the encoder output in place; the backing storage is owned by this state.
      const auto info = image_features_value_->GetTensorTypeAndShapeInfo();
      const auto shape = info->GetShape();
      std::span<float> data(image_features_value_->GetTensorMutableData<float>(), info->GetElementCount());
      value = OrtValue::CreateTensor<float>(*mem_info, data, std::span<const int64_t>(shape));
    } else {
      // No features for this modality: hand the graph an empty [0, hidden] tensor so the
      // in-graph merge is a no-op, matching what MultiModalLanguageModel does.
      auto declared = vl_model_.session_info_.GetInputShape(name);
      const int64_t hidden = declared.empty() ? 0 : declared.back();
      const std::vector<int64_t> shape{0, hidden};
      value = OrtValue::CreateTensor(vl_model_.allocator_cpu_, shape,
                                     ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT);
    }

    auto tensor = std::make_shared<Tensor>(std::move(value));
    inputs.push_back(ExtraInput{name, tensor});
    embedding_merges_features_ = true;
  };

  add(image_name, /*use_vision_output=*/true);
  add(audio_name, /*use_vision_output=*/false);
  features_bound_ = true;
}

void Qwen2_5_VL_PipelineState::RunVision(const std::vector<ExtraInput>& extra_inputs) {
  if (vision_ran_) return;

  if (vl_model_.vision_session_) {
    RunSingleSessionVision(extra_inputs);
    return;
  }

  if (!vl_model_.vision_pipeline_) return;

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

  // Extract grid_thw if provided
  std::vector<int64_t> grid_thw;
  if (image_grid_thw_val) {
    auto grid_shape = image_grid_thw_val->GetTensorTypeAndShapeInfo()->GetShape();
    auto element_type = image_grid_thw_val->GetTensorTypeAndShapeInfo()->GetElementType();

    if (element_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64) {
      const int64_t* grid_data = image_grid_thw_val->GetTensorData<int64_t>();
      size_t grid_count = 1;
      for (auto dim : grid_shape) grid_count *= dim;

      // Expect [batch, 3] or [3] shape - take last 3 values as [t, h, w]
      if (grid_count >= 3) {
        grid_thw = {grid_data[grid_count - 3], grid_data[grid_count - 2], grid_data[grid_count - 1]};
      }
    } else if (element_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32) {
      const int32_t* grid_data = image_grid_thw_val->GetTensorData<int32_t>();
      size_t grid_count = 1;
      for (auto dim : grid_shape) grid_count *= dim;

      if (grid_count >= 3) {
        grid_thw = {static_cast<int64_t>(grid_data[grid_count - 3]),
                    static_cast<int64_t>(grid_data[grid_count - 2]),
                    static_cast<int64_t>(grid_data[grid_count - 1])};
      }
    }
  }

  try {
    image_features_buffer_ = vl_model_.vision_pipeline_->Run(pixel_data, pixel_shape_vec, grid_thw);
  } catch (const std::exception& e) {
    throw std::runtime_error(std::string("Vision pipeline failed: ") + e.what());
  }

  auto out_shape = vl_model_.vision_pipeline_->GetLastOutputShape();
  if (out_shape.size() != 2) {
    throw std::runtime_error("Vision pipeline: expected output shape rank 2, got " + std::to_string(out_shape.size()));
  }

  auto mem_info = OrtMemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);
  std::span<float> data_span(image_features_buffer_.data(), image_features_buffer_.size());
  std::span<const int64_t> shape_span(out_shape.data(), out_shape.size());
  image_features_value_ = OrtValue::CreateTensor<float>(*mem_info, data_span, shape_span);

  vision_ran_ = true;
}

void Qwen2_5_VL_PipelineState::RunSingleSessionVision(const std::vector<ExtraInput>& extra_inputs) {
  const auto& pixel_name = vl_model_.config_->model.vision.inputs.pixel_values;

  auto find_extra_input = [&extra_inputs](const std::string& name) -> OrtValue* {
    for (const auto& input : extra_inputs) {
      if (input.name == name) return input.tensor->GetOrtTensor();
    }
    return nullptr;
  };

  // A text-only request supplies no pixel_values; leave the embeddings untouched.
  if (!find_extra_input(pixel_name)) return;

  const auto input_names = vl_model_.vision_session_->GetInputNames();
  std::vector<const char*> input_name_ptrs;
  std::vector<const OrtValue*> input_values;
  input_name_ptrs.reserve(input_names.size());
  input_values.reserve(input_names.size());
  for (const auto& name : input_names) {
    OrtValue* value = find_extra_input(name);
    if (!value) {
      throw std::runtime_error("Vision encoder: required input '" + name +
                               "' was not produced by the processor");
    }
    input_name_ptrs.push_back(name.c_str());
    input_values.push_back(value);
  }

  // Gemma-4's vision graph has a static batch of 1, so several images must be encoded one at a
  // time and their features concatenated. This mirrors Gemma4VisionState on the multi-modal path.
  const auto& position_name = vl_model_.config_->model.vision.inputs.pixel_position_ids;
  size_t pixel_index = SIZE_MAX;
  size_t position_index = SIZE_MAX;
  for (size_t i = 0; i < input_names.size(); ++i) {
    if (input_names[i] == pixel_name) pixel_index = i;
    if (input_names[i] == position_name) position_index = i;
  }
  OrtValue* pixel_values = find_extra_input(pixel_name);
  OrtValue* position_values = position_index == SIZE_MAX ? nullptr : find_extra_input(position_name);
  const auto pixel_shape = pixel_values->GetTensorTypeAndShapeInfo()->GetShape();
  const int64_t num_images = pixel_shape.size() == 3 ? pixel_shape[0] : 1;

  const auto output_names = vl_model_.vision_session_->GetOutputNames();
  if (output_names.empty()) {
    throw std::runtime_error("Vision encoder: model has no outputs");
  }
  // Fall back to the first output only when the config names no output at all. A name that
  // is set but absent from the model is a misconfiguration: running output 0 instead would
  // inject whatever that output happens to be as if it were image features.
  const auto& features_name = vl_model_.config_->model.vision.outputs.image_features;
  size_t output_index = 0;
  if (!features_name.empty()) {
    const auto found = std::find(output_names.begin(), output_names.end(), features_name);
    if (found == output_names.end()) {
      std::string available;
      for (const auto& name : output_names) {
        available += (available.empty() ? "" : ", ") + name;
      }
      throw std::runtime_error("Vision encoder: configured vision.outputs.image_features '" +
                               features_name + "' is not an output of the vision model. Available outputs: " +
                               available);
    }
    output_index = static_cast<size_t>(std::distance(output_names.begin(), found));
  }
  const char* output_name_ptrs[] = {output_names[output_index].c_str()};

  auto run_encoder = [&]() {
    OrtValue* raw_output = nullptr;
    vl_model_.vision_session_->Run(nullptr, input_name_ptrs.data(), input_values.data(),
                                   input_name_ptrs.size(), output_name_ptrs, &raw_output, 1);
    std::unique_ptr<OrtValue> owned(raw_output);
    if (owned->GetTensorTypeAndShapeInfo()->GetElementType() != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
      std::unique_ptr<OrtValue> cast_output;
      Cast(*owned, cast_output, *vl_model_.p_device_inputs_, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT);
      owned = std::move(cast_output);
    }
    return owned;
  };

  auto mem_info = OrtMemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);

  if (num_images > 1) {
    if (pixel_index == SIZE_MAX) {
      throw std::runtime_error("Vision encoder: multi-image input requires a '" + pixel_name + "' session input");
    }
    image_features_buffer_.clear();
    int64_t hidden_size = 0;
    for (int64_t image = 0; image < num_images; ++image) {
      auto pixel_slice = SliceLeadingImage(*pixel_values, image);
      input_values[pixel_index] = pixel_slice.get();
      std::unique_ptr<OrtValue> position_slice;
      if (position_values) {
        position_slice = SliceLeadingImage(*position_values, image);
        input_values[position_index] = position_slice.get();
      }

      auto features = run_encoder();
      const auto features_info = features->GetTensorTypeAndShapeInfo();
      const auto features_shape = SqueezeToRank2(features_info->GetShape());
      if (hidden_size == 0) {
        hidden_size = features_shape[1];
      } else if (hidden_size != features_shape[1]) {
        throw std::runtime_error("Vision encoder: image features changed hidden size between images");
      }
      const float* data = features->GetTensorMutableData<float>();
      image_features_buffer_.insert(image_features_buffer_.end(), data, data + features_info->GetElementCount());
    }

    // The per-image outputs were copied out, so nothing from the session needs to stay alive.
    vision_output_owner_.reset();
    const std::vector<int64_t> combined_shape{static_cast<int64_t>(image_features_buffer_.size()) / hidden_size,
                                              hidden_size};
    image_features_value_ = OrtValue::CreateTensor<float>(*mem_info, std::span<float>(image_features_buffer_),
                                                          std::span<const int64_t>(combined_shape));
    vision_ran_ = true;
    return;
  }

  vision_output_owner_ = run_encoder();
  const auto output_info = vision_output_owner_->GetTensorTypeAndShapeInfo();
  const auto shape = SqueezeToRank2(output_info->GetShape());

  std::span<float> data_span(vision_output_owner_->GetTensorMutableData<float>(),
                             output_info->GetElementCount());
  std::span<const int64_t> shape_span(shape.data(), shape.size());
  image_features_value_ = OrtValue::CreateTensor<float>(*mem_info, data_span, shape_span);

  vision_ran_ = true;
}

void Qwen2_5_VL_PipelineState::OnStageComplete(size_t stage_id) {
  if (stage_id != 0 || !vision_ran_) return;

  // When the embedding graph takes image_features directly it has already placed the rows,
  // so injecting them again would be redundant work over identical values.
  if (embedding_merges_features_) return;

  const auto& embeddings_config = vl_model_.config_->model.decoder.pipeline[0];
  if (!embeddings_config.outputs.empty()) {
    InjectVisionEmbeddings(embeddings_config.outputs[0]);
  }
}

void Qwen2_5_VL_PipelineState::InjectVisionEmbeddings(const std::string& embeddings_output_name) {
  auto it = ortvalue_store_.find(embeddings_output_name);
  if (it == ortvalue_store_.end() || !it->second) {
    throw std::runtime_error("Vision embedding injection: embeddings output '" + embeddings_output_name + "' not found in ortvalue_store");
  }

  OrtValue* embeddings_ortvalue = it->second.get();
  auto embeddings_info = embeddings_ortvalue->GetTensorTypeAndShapeInfo();
  if (embeddings_info->GetElementType() != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
    throw std::runtime_error("Vision embedding injection: embeddings output must have float elements");
  }
  auto shape = embeddings_info->GetShape();

  auto vision_shape = image_features_value_->GetTensorTypeAndShapeInfo()->GetShape();

  // model.image_token_id is the general contract, but no builder emitted it until recently:
  // only builders/mistral.py assigns it. Configs for the three model types that reached this
  // code before relied on a hardcoded 151655, so keep that value for exactly those types
  // rather than fail a config that used to work. Everything else must set the field.
  int32_t image_token_id = static_cast<int32_t>(vl_model_.config_->model.image_token_id);
  if (image_token_id == 0) {
    const auto& model_type = vl_model_.config_->model.type;
    if (model_type == "fara" || model_type == "qwen2_5_vl" || model_type == "qwen3_vl") {
      constexpr int32_t kQwenVLImageTokenId = 151655;
      image_token_id = kQwenVLImageTokenId;
    } else {
      throw std::runtime_error("Vision embedding injection: model.image_token_id is not set in genai_config.json");
    }
  }

  if (!input_ids_ || !input_ids_->Get()) {
    throw std::runtime_error("Vision embedding injection: input_ids not available");
  }

  OrtValue* input_ids_ortvalue = input_ids_->Get();
  auto input_ids_info = input_ids_ortvalue->GetTensorTypeAndShapeInfo();
  const int32_t* token_ids_cpu = input_ids_ortvalue->GetTensorData<int32_t>();

  const size_t total_tokens = input_ids_info->GetElementCount();
  ValidateVisionEmbeddingShapes(shape, embeddings_info->GetElementCount(), vision_shape, total_tokens);
  float* embeddings_data = embeddings_ortvalue->GetTensorMutableData<float>();
  const float* vision_data = image_features_value_->GetTensorData<float>();
  const int64_t num_vision_tokens = vision_shape[0];
  const int64_t embedding_dim = shape.back();
  const int64_t vision_dim = vision_shape[1];

  for (size_t i = 0; i < total_tokens; ++i) {
    if (token_ids_cpu[i] == image_token_id && image_embed_consumed_ < static_cast<size_t>(num_vision_tokens)) {
      std::memcpy(embeddings_data + (i * embedding_dim),
                  vision_data + (image_embed_consumed_ * vision_dim),
                  vision_dim * sizeof(float));
      image_embed_consumed_++;
    }
  }

  // Warn if there's a mismatch between image tokens and vision features
  if (image_embed_consumed_ != static_cast<size_t>(num_vision_tokens)) {
    if (g_log.enabled)
      Log("warning", "Vision embedding mismatch: consumed " + std::to_string(image_embed_consumed_) +
                         " of " + std::to_string(num_vision_tokens) + " available vision tokens. " +
                         "This may indicate a mismatch between the number of image placeholders in the prompt " +
                         "and the number of images provided.");
  }
}

}  // namespace Generators
