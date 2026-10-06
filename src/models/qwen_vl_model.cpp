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
  // session. Exports commonly ship an audio encoder alongside the vision one, so a config
  // reaching here may well declare a speech model that this path cannot host.
  //
  // Binding is not the hazard: AppendEmbeddingFeatureInputs hands any unfilled modality an
  // empty [0, hidden] tensor, so the in-graph merge is a no-op and the declared-but-unused
  // audio encoder changes no result. The hazard is the multimodal processor, which enables
  // audio whenever speech.config_filename and speech.filename are both set, then resolves
  // speech.inputs.audio_embeds through session_info. No speech session was created, so that
  // lookup throws.
  //
  // Clear the speech config so audio is cleanly unavailable rather than half-configured.
  // Rejecting the model instead would make the common vision-plus-audio export unusable on
  // this path for text and image prompts, which are the only things a chunked NPU decoder
  // serves today.
  if (!config_->model.speech.filename.empty()) {
    config_->model.speech.filename.clear();
    config_->model.speech.config_filename.clear();
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
  InitializeFeatureInputs();
}

void Qwen2_5_VL_PipelineState::SetExtraInputs(const std::vector<ExtraInput>& extra_inputs) {
  RunVision(extra_inputs);
  DecoderOnlyPipelineState::SetExtraInputs(extra_inputs);
}

DeviceSpan<float> Qwen2_5_VL_PipelineState::Run(int total_length, DeviceSpan<int32_t>& next_tokens,
                                                DeviceSpan<int32_t> next_indices) {
  if (next_tokens.size() > 1) image_feature_offset_ = 0;
  return DecoderOnlyPipelineState::Run(total_length, next_tokens, next_indices);
}

void Qwen2_5_VL_PipelineState::InitializeFeatureInputs() {
  const auto& image_name = vl_model_.config_->model.embedding.inputs.image_features;
  const auto& audio_name = vl_model_.config_->model.embedding.inputs.audio_features;

  auto add = [&](const std::string& name, size_t& index, std::unique_ptr<OrtValue>& value) {
    if (name.empty() || !vl_model_.session_info_.HasInput(name)) return;
    value = CreateEmptyFeatureInput(name);
    index = inputs_.size();
    input_names_.push_back(name.c_str());
    inputs_.push_back(value.get());
  };

  add(image_name, image_feature_input_index_, image_feature_input_);
  embedding_merges_features_ = image_feature_input_index_ != SIZE_MAX;
  add(audio_name, audio_feature_input_index_, audio_feature_input_);
}

std::unique_ptr<OrtValue> Qwen2_5_VL_PipelineState::CreateEmptyFeatureInput(const std::string& name) const {
  const auto declared = vl_model_.session_info_.GetInputShape(name);
  if (declared.size() != 2 && declared.size() != 3) {
    throw std::runtime_error("Embedding feature input '" + name + "' must have rank 2 or 3");
  }
  const int64_t hidden = declared.back() > 0 ? declared.back() : vl_model_.config_->model.decoder.hidden_size;
  std::vector<int64_t> shape;
  if (declared.size() == 3) {
    shape.push_back(declared[0] > 0 ? declared[0] : 1);
  }
  shape.push_back(0);
  shape.push_back(hidden);
  return OrtValue::CreateTensor(vl_model_.allocator_cpu_, shape,
                                vl_model_.session_info_.GetInputDataType(name));
}

void Qwen2_5_VL_PipelineState::OnStageStart(size_t stage_id) {
  if (!embedding_merges_features_) return;
  const auto& image_name = vl_model_.config_->model.embedding.inputs.image_features;
  const auto& stage_inputs = vl_model_.config_->model.decoder.pipeline[stage_id].inputs;
  if (std::find(stage_inputs.begin(), stage_inputs.end(), image_name) == stage_inputs.end()) return;
  UpdateImageFeatureInput();
}

void Qwen2_5_VL_PipelineState::UpdateImageFeatureInput() {
  const auto& image_name = vl_model_.config_->model.embedding.inputs.image_features;
  if (!first_run_ || !image_features_value_) {
    image_feature_cast_.reset();
    image_feature_input_ = CreateEmptyFeatureInput(image_name);
    inputs_[image_feature_input_index_] = image_feature_input_.get();
    return;
  }
  auto input_ids_info = input_ids_->Get()->GetTensorTypeAndShapeInfo();
  const auto input_ids_count = input_ids_info->GetElementCount();
  const int32_t* input_ids = input_ids_->Get()->GetTensorData<int32_t>();
  const int32_t image_token_id = GetImageTokenId();

  size_t feature_count = 0;
  for (size_t i = 0; i < input_ids_count; ++i) {
    if (input_ids[i] == image_token_id) ++feature_count;
  }

  if (feature_count == 0) {
    image_feature_cast_.reset();
    image_feature_input_ = CreateEmptyFeatureInput(image_name);
    inputs_[image_feature_input_index_] = image_feature_input_.get();
    return;
  }

  const auto source_info = image_features_value_->GetTensorTypeAndShapeInfo();
  const auto source_shape = SqueezeToRank2(source_info->GetShape());
  const size_t available_features = static_cast<size_t>(source_shape[0]);
  if (image_feature_offset_ + feature_count > available_features) {
    throw std::runtime_error("Embedding feature input requires more image rows than the vision encoder produced");
  }

  const int64_t hidden = source_shape[1];
  const auto declared = vl_model_.session_info_.GetInputShape(image_name);
  std::vector<int64_t> shape;
  if (declared.size() == 3) shape.push_back(declared[0] > 0 ? declared[0] : 1);
  shape.push_back(static_cast<int64_t>(feature_count));
  shape.push_back(hidden);

  auto mem_info = OrtMemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeDefault);
  float* source = image_features_value_->GetTensorMutableData<float>() + image_feature_offset_ * hidden;
  const std::vector<int64_t> rank2_shape{static_cast<int64_t>(feature_count), hidden};
  auto source_view = OrtValue::CreateTensor<float>(
      *mem_info, std::span<float>(source, feature_count * hidden), rank2_shape);
  const auto target_type = vl_model_.session_info_.GetInputDataType(image_name);
  if (target_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
    image_feature_cast_.reset();
    image_feature_input_ = OrtValue::CreateTensor(
        *mem_info, source, feature_count * hidden * sizeof(float), shape, target_type);
  } else {
    Cast(*source_view, image_feature_cast_, *GetDeviceInterface(DeviceType::CPU), target_type);
    image_feature_input_ = OrtValue::CreateTensor(
        *mem_info, image_feature_cast_->GetTensorMutableRawData(),
        feature_count * hidden * Ort::SizeOf(target_type), shape, target_type);
  }
  inputs_[image_feature_input_index_] = image_feature_input_.get();
  image_feature_offset_ += feature_count;
}

int32_t Qwen2_5_VL_PipelineState::GetImageTokenId() const {
  const int32_t image_token_id = static_cast<int32_t>(vl_model_.config_->model.image_token_id);
  if (image_token_id != 0) return image_token_id;

  const auto& model_type = vl_model_.config_->model.type;
  // Preserve the token ID used by legacy Qwen configs that omit this field.
  if (model_type == "fara" || model_type == "qwen2_5_vl" || model_type == "qwen3_vl") {
    constexpr int32_t kQwenVLImageTokenId = 151655;
    return kQwenVLImageTokenId;
  }

  throw std::runtime_error("model.image_token_id is not set in genai_config.json");
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
  std::vector<OrtValue*> source_values;
  input_name_ptrs.reserve(input_names.size());
  input_values.reserve(input_names.size());
  source_values.reserve(input_names.size());
  for (const auto& name : input_names) {
    OrtValue* value = find_extra_input(name);
    if (!value) {
      throw std::runtime_error("Vision encoder: required input '" + name +
                               "' was not produced by the processor");
    }
    input_name_ptrs.push_back(name.c_str());
    input_values.push_back(value);
    source_values.push_back(value);
  }

  // Gemma-4's vision graph has a static batch of 1, so several images must be encoded one at a
  // time and their features concatenated. This mirrors Gemma4VisionState on the multi-modal path.
  size_t pixel_index = SIZE_MAX;
  for (size_t i = 0; i < input_names.size(); ++i) {
    if (input_names[i] == pixel_name) pixel_index = i;
  }
  OrtValue* pixel_values = find_extra_input(pixel_name);
  const auto pixel_shape = pixel_values->GetTensorTypeAndShapeInfo()->GetShape();
  const int64_t num_images = pixel_shape.size() == 3 ? pixel_shape[0] : 1;

  // Every input the processor produced per image has to be sliced in step with pixel_values,
  // not just the ones this model family happens to name. Routing here is by capability, so
  // the session may declare inputs beyond pixel_values and pixel_position_ids, such as an
  // attention mask or spatial shapes. Leaving those at the full batch while pixel_values is
  // sliced to one image would feed the encoder mismatched batches. Select by leading
  // dimension, which is what makes an input per image.
  std::vector<size_t> batched_indices;
  if (num_images > 1) {
    for (size_t i = 0; i < source_values.size(); ++i) {
      const auto shape = source_values[i]->GetTensorTypeAndShapeInfo()->GetShape();
      if (shape.size() >= 2 && shape[0] == num_images) batched_indices.push_back(i);
    }
  }

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
    if (vl_model_.config_->model.vision.run_options.has_value()) {
      State::SetRunOptions(vl_model_.config_->model.vision.run_options.value());
    }
    OrtValue* raw_output = nullptr;
    vl_model_.vision_session_->Run(run_options_.get(), input_name_ptrs.data(), input_values.data(),
                                   input_name_ptrs.size(), output_name_ptrs, &raw_output, 1);
    std::unique_ptr<OrtValue> owned(raw_output);
    if (owned->GetTensorTypeAndShapeInfo()->GetElementType() != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
      std::unique_ptr<OrtValue> cast_output;
      // Cast allocates on the device it is handed, and p_device_inputs_ follows the decoder,
      // which may be an accelerator. Everything downstream of this point is host code: the
      // features are memcpy'd into image_features_buffer_ and wrapped in a CPU tagged OrtValue.
      // Cast on CPU so that stays true no matter what device the decoder runs on.
      Cast(*owned, cast_output, *GetDeviceInterface(DeviceType::CPU), ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT);
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
      // Slices must outlive the Run call below, so hold them for the whole iteration.
      std::vector<std::unique_ptr<OrtValue>> slices;
      slices.reserve(batched_indices.size());
      for (size_t index : batched_indices) {
        slices.push_back(SliceLeadingImage(*source_values[index], image));
        input_values[index] = slices.back().get();
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

  const int32_t image_token_id = GetImageTokenId();

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
