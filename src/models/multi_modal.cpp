// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "../generators.h"
#include "multi_modal.h"
#include <numeric>

namespace Generators {

namespace {

int64_t GetNumImageTokens(const std::vector<ExtraInput>& extra_inputs) {
  for (size_t i = 0; i < extra_inputs.size(); ++i) {
    if (extra_inputs[i].name == Config::Defaults::NumImageTokens) {
      assert(extra_inputs[i].tensor->ort_tensor_);
      const int64_t* num_image_tokens_data = extra_inputs[i].tensor->ort_tensor_->GetTensorData<int64_t>();
      return std::accumulate(num_image_tokens_data,
                             num_image_tokens_data + extra_inputs[i].tensor->ort_tensor_->GetTensorTypeAndShapeInfo()->GetElementCount(),
                             0LL);
    }
  }

  return 0;
}

int64_t GetNumAudioTokens(const std::vector<ExtraInput>& extra_inputs,
                          const std::string& audio_sizes_name) {
  for (size_t i = 0; i < extra_inputs.size(); ++i) {
    if (extra_inputs[i].name == audio_sizes_name) {
      assert(extra_inputs[i].tensor->ort_tensor_);
      auto type_and_shape_info = extra_inputs[i].tensor->ort_tensor_->GetTensorTypeAndShapeInfo();
      const auto element_count = type_and_shape_info->GetElementCount();
      if (type_and_shape_info->GetElementType() == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64) {
        const int64_t* audio_sizes_data = extra_inputs[i].tensor->ort_tensor_->GetTensorData<int64_t>();
        return std::accumulate(audio_sizes_data, audio_sizes_data + element_count, 0LL);
      } else {
        throw std::runtime_error("Unsupported data type " + std::to_string(static_cast<int64_t>(type_and_shape_info->GetElementType())) + " for audio_sizes tensor. Only int64 is supported.");
      }
    }
  }

  return 0;
}

// Returns the number of images in the current batch.
//
// Two strategies are tried in order:
//
// 1. pixel_values rank-3 path (Phi, Gemma, and legacy Qwen processors):
//    The extension returns pixel_values as [N, patches_per_image, patch_dim], so
//    the batch size is simply shape[0].
//
// 2. image_grid_thw fallback (Qwen2.5-VL / Qwen3-VL after the multi-image flatten fix):
//    QwenImageProcessor flattens pixel_values to rank 2 [total_patches, patch_dim]
//    so that vision.onnx—which is exported for a single image and loops per-image in
//    VisionState::Run—always receives a 2-D input regardless of image count.
//    Rank-2 pixel_values carries no image-count information, so we fall through and
//    read num_images from image_grid_thw.shape[0] ([num_images, 3]).
int64_t GetImageFeatureBatchSize(const std::vector<ExtraInput>& extra_inputs) {
  for (size_t i = 0; i < extra_inputs.size(); ++i) {
    if (extra_inputs[i].name == Config::Defaults::PixelValuesName) {
      assert(extra_inputs[i].tensor->ort_tensor_);
      const auto num_dims = extra_inputs[i].tensor->ort_tensor_->GetTensorTypeAndShapeInfo()->GetShape().size();
      if (num_dims < 3) {
        // Qwen flattens pixel_values to [total_patches, patch_dim] (rank 2) so that
        // vision.onnx always receives a single-image-shaped input; num_images cannot
        // be inferred from pixel_values alone — fall through to image_grid_thw.
        break;
      }
      // Rank ≥ 3: batch size is the leading dimension (Phi, Gemma, legacy Qwen).
      return extra_inputs[i].tensor->ort_tensor_->GetTensorTypeAndShapeInfo()->GetShape().front();
    }
  }

  // Fallback for Qwen2.5-VL / Qwen3-VL: image_grid_thw has shape [num_images, 3]
  // so its leading dimension directly gives the image count.
  // This tensor is Qwen-specific; for Phi and Gemma it is absent and we return 0.
  for (size_t i = 0; i < extra_inputs.size(); ++i) {
    if (extra_inputs[i].name == Config::Defaults::ImageGridThwName) {
      assert(extra_inputs[i].tensor->ort_tensor_);
      const auto shape = extra_inputs[i].tensor->ort_tensor_->GetTensorTypeAndShapeInfo()->GetShape();
      if (!shape.empty()) {
        return shape[0];  // num_images
      }
    }
  }

  return 0;
}

}  // namespace

MultiModalLanguageModel::MultiModalLanguageModel(std::unique_ptr<Config> config, OrtEnv& ort_env, bool vision, bool speech)
    : Model(std::move(config)) {
  // The non-decoder models don't support graph capture because of control flow nodes, so disable graph capture for them
  if (vision) {
    vision_session_options_ = OrtSessionOptions::Create();
    CreateSessionOptionsFromConfig(config_->model.vision.session_options.has_value() ? config_->model.vision.session_options.value() : config_->model.decoder.session_options, *vision_session_options_, true, true);
    vision_session_ = CreateSession(ort_env, config_->model.vision.filename, vision_session_options_.get());
  }

  if (speech) {
    speech_session_options_ = OrtSessionOptions::Create();
    CreateSessionOptionsFromConfig(config_->model.speech.session_options.has_value() ? config_->model.speech.session_options.value() : config_->model.decoder.session_options, *speech_session_options_, true, true);
    speech_session_ = CreateSession(ort_env, config_->model.speech.filename, speech_session_options_.get());
  }

  embedding_session_options_ = OrtSessionOptions::Create();
  CreateSessionOptionsFromConfig(config_->model.embedding.session_options.has_value() ? config_->model.embedding.session_options.value() : config_->model.decoder.session_options, *embedding_session_options_, true, true);

  embedding_session_ = CreateSession(ort_env, config_->model.embedding.filename, embedding_session_options_.get());
  decoder_session_ = CreateSession(ort_env, config_->model.decoder.filename, session_options_.get());

  session_info_.Add(*decoder_session_);
  session_info_.Add(*embedding_session_);
  if (speech) {
    session_info_.Add(*speech_session_);
  }
  if (vision) {
    session_info_.Add(*vision_session_);
  }
}

std::unique_ptr<State> MultiModalLanguageModel::CreateState(DeviceSpan<int32_t> sequence_lengths, const GeneratorParams& params) const {
  return std::make_unique<MultiModalPipelineState>(*this, sequence_lengths, params);
}

VisionState::VisionState(const MultiModalLanguageModel& model, const GeneratorParams& params)
    : State{params, model},
      model_{model} {}

void VisionState::SetExtraInputs(const std::vector<ExtraInput>& extra_inputs, const int64_t num_images, const int64_t num_image_tokens) {
  num_image_tokens_ = num_image_tokens;
  num_images_ = num_images;

  image_features_ = std::make_unique<MultiModalFeatures>(*this, MultiModalFeatures::Mode::Output,  // Optional model input
                                                         model_.config_->model.vision.outputs.image_features,
                                                         num_images_, num_image_tokens_);
  image_features_->Add();
  extra_inputs_.Add(extra_inputs, model_.vision_session_->GetInputNames());
}

DeviceSpan<float> VisionState::Run(int current_length, DeviceSpan<int32_t>& next_tokens, DeviceSpan<int32_t> next_indices) {
  if (model_.config_->model.vision.run_options.has_value()) {
    State::SetRunOptions(model_.config_->model.vision.run_options.value());
  }

  State::Run(*model_.vision_session_);
  return {};
}

// ---------------------------------------------------------------------------
// QwenVisionState: batched single-call or per-image loop
// ---------------------------------------------------------------------------

DeviceSpan<float> QwenVisionState::Run(int current_length, DeviceSpan<int32_t>& next_tokens, DeviceSpan<int32_t> next_indices) {
  if (model_.config_->model.vision.run_options.has_value()) {
    State::SetRunOptions(model_.config_->model.vision.run_options.value());
  }

  // Single image (or no image data): always a direct call.
  if (num_images_ <= 1) {
    State::Run(*model_.vision_session_);
    return {};
  }

  // --- Multi-image: decide between batched call vs per-image loop ---
  //
  // Strategy:
  //   1. Check if the ONNX model's image_grid_thw input has a DYNAMIC dim-0
  //      (i.e. supports variable num_images in one call).
  //   2. Check if all images share the same (t, h, w) grid — the vectorized
  //      model code uses a uniform-grid assumption so batching only works
  //      when grids are identical.
  //   3. If both conditions are met → single State::Run (like HuggingFace).
  //      Otherwise → per-image loop (handles any mix of image sizes, and
  //      works with static/QNN models that only accept N=1).

  const std::string& pv_name = model_.config_->model.vision.inputs.pixel_values;
  const std::string& grid_name = model_.config_->model.vision.inputs.image_grid_thw;

  size_t pv_idx = SIZE_MAX;
  size_t grid_idx = SIZE_MAX;
  for (size_t i = 0; i < input_names_.size(); ++i) {
    if (input_names_[i] == pv_name) {
      pv_idx = i;
    }
    if (input_names_[i] == grid_name) {
      grid_idx = i;
    }
  }

  if (pv_idx == SIZE_MAX || grid_idx == SIZE_MAX) {
    // Couldn't find expected inputs – fall back to single Run.
    State::Run(*model_.vision_session_);
    return {};
  }

  OrtValue* grid_full = inputs_[grid_idx];
  const int64_t* grid_data = grid_full->GetTensorData<int64_t>();

  // Check if the ONNX model accepts dynamic num_images.
  // A non-positive dim-0 (0 or -1) in the model's input shape = dynamic/symbolic.
  bool model_supports_batch = false;
  {
    // Find image_grid_thw's ordinal index in the ONNX session's inputs.
    auto session_input_names = model_.vision_session_->GetInputNames();
    for (size_t si = 0; si < session_input_names.size(); ++si) {
      if (session_input_names[si] == grid_name) {
        auto grid_input_info = model_.vision_session_->GetInputTypeInfo(si);
        auto grid_expected_shape = grid_input_info->GetTensorTypeAndShapeInfo().GetShape();
        if (!grid_expected_shape.empty() && grid_expected_shape[0] <= 0) {
          model_supports_batch = true;  // dim-0 is symbolic — accepts any N
        }
        break;
      }
    }
  }

  // Check if all images share the same (t, h, w) grid.
  bool uniform_grid = true;
  if (num_images_ > 1) {
    int64_t t0 = grid_data[0], h0 = grid_data[1], w0 = grid_data[2];
    for (int64_t img = 1; img < num_images_; ++img) {
      if (grid_data[img * 3] != t0 || grid_data[img * 3 + 1] != h0 || grid_data[img * 3 + 2] != w0) {
        uniform_grid = false;
        break;
      }
    }
  }

  // --- Batched single-call path (like HuggingFace) ---
  if (model_supports_batch && uniform_grid) {
    // The model has dynamic image_grid_thw dim-0 and all images share the
    // same grid.  Pass all N images' pixel_values and the full [N, 3]
    // grid_thw in one call — the ONNX graph was vectorized to handle this.
    State::Run(*model_.vision_session_);
    return {};
  }

  // --- Per-image loop path (fallback for different-sized images or static models) ---
  OrtValue* pv_full = inputs_[pv_idx];
  OrtValue* feat_full = outputs_[0];  // pre-allocated image_features output

  // Shapes: pixel_values[total_patches, patch_dim], image_features[total_logical_patches, hidden_size]
  auto pv_info = pv_full->GetTensorTypeAndShapeInfo();
  auto feat_info = feat_full->GetTensorTypeAndShapeInfo();
  auto pv_shape = pv_info->GetShape();
  auto feat_shape = feat_info->GetShape();
  auto pv_type = pv_info->GetElementType();
  auto feat_type = feat_info->GetElementType();
  int64_t patch_dim = pv_shape[1];
  int64_t hidden_size = feat_shape[1];

  // Map ONNX element type to byte size.
  auto element_size = [](ONNXTensorElementDataType type) -> size_t {
    switch (type) {
      case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT:
        return 4;
      case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16:
        return 2;
      case ONNX_TENSOR_ELEMENT_DATA_TYPE_BFLOAT16:
        return 2;
      case ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE:
        return 8;
      default:
        throw std::runtime_error("Unsupported pixel_values element type in multi-image vision loop");
    }
  };
  size_t pv_element_size = element_size(pv_type);
  size_t feat_element_size = element_size(feat_type);

  // grid_data already obtained above for the uniform-grid check.
  void* pv_raw = pv_full->GetTensorMutableRawData();
  void* feat_raw = feat_full->GetTensorMutableRawData();
  int64_t spatial_merge_size = model_.config_->model.vision.spatial_merge_size;

  auto cpu_mem = OrtMemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);

  int64_t total_patches = pv_shape[0];
  int64_t total_feats = feat_shape[0];
  int64_t merge_sq = spatial_merge_size * spatial_merge_size;

  int64_t patch_offset = 0;
  int64_t feat_offset = 0;
  for (int64_t img = 0; img < num_images_; ++img) {
    int64_t t = grid_data[img * 3];
    int64_t h = grid_data[img * 3 + 1];
    int64_t w = grid_data[img * 3 + 2];
    int64_t num_patches = t * h * w;

    if (num_patches % merge_sq != 0)
      throw std::runtime_error("num_patches (" + std::to_string(num_patches) +
                               ") is not divisible by spatial_merge_size^2 (" +
                               std::to_string(merge_sq) + ") for image " + std::to_string(img));
    if (patch_offset + num_patches > total_patches)
      throw std::runtime_error("patch_offset (" + std::to_string(patch_offset) + ") + num_patches (" +
                               std::to_string(num_patches) + ") exceeds pixel_values dim 0 (" +
                               std::to_string(total_patches) + ")");

    int64_t num_feats = num_patches / merge_sq;
    if (feat_offset + num_feats > total_feats)
      throw std::runtime_error("feat_offset (" + std::to_string(feat_offset) + ") + num_feats (" +
                               std::to_string(num_feats) + ") exceeds image_features dim 0 (" +
                               std::to_string(total_feats) + ")");

    // Create non-owning sub-tensors (zero-copy views into the original buffers).
    std::vector<int64_t> sub_pv_shape = {num_patches, patch_dim};
    std::vector<int64_t> sub_grid_shape = {1LL, 3LL};  // vision.onnx expects [1, 3] per image
    std::vector<int64_t> sub_feat_shape = {num_feats, hidden_size};

    auto sub_pv = OrtValue::CreateTensor(
        *cpu_mem,
        static_cast<uint8_t*>(pv_raw) + static_cast<size_t>(patch_offset * patch_dim) * pv_element_size,
        static_cast<size_t>(num_patches * patch_dim) * pv_element_size,
        std::span<const int64_t>(sub_pv_shape), pv_type);

    auto sub_grid = OrtValue::CreateTensor(
        *cpu_mem,
        const_cast<void*>(static_cast<const void*>(grid_data + img * 3)),
        3 * sizeof(int64_t),
        std::span<const int64_t>(sub_grid_shape),
        ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64);

    auto sub_feat = OrtValue::CreateTensor(
        *cpu_mem,
        static_cast<uint8_t*>(feat_raw) + static_cast<size_t>(feat_offset * hidden_size) * feat_element_size,
        static_cast<size_t>(num_feats * hidden_size) * feat_element_size,
        std::span<const int64_t>(sub_feat_shape), feat_type);

    // Temporarily point the State's inputs/output to the per-image slices,
    // run the session, then advance offsets.
    inputs_[pv_idx] = sub_pv.get();
    inputs_[grid_idx] = sub_grid.get();
    outputs_[0] = sub_feat.get();

    State::Run(*model_.vision_session_);

    patch_offset += num_patches;
    feat_offset += num_feats;
  }

  if (patch_offset != total_patches)
    throw std::runtime_error("Final patch_offset (" + std::to_string(patch_offset) +
                             ") != total patches (" + std::to_string(total_patches) + ")");
  if (feat_offset != total_feats)
    throw std::runtime_error("Final feat_offset (" + std::to_string(feat_offset) +
                             ") != total features (" + std::to_string(total_feats) + ")");

  // Restore original pointers so the State remains valid after this call.
  inputs_[pv_idx] = pv_full;
  inputs_[grid_idx] = grid_full;
  outputs_[0] = feat_full;

  return {};
}

// ---------------------------------------------------------------------------
// Factory
// ---------------------------------------------------------------------------

std::unique_ptr<VisionState> CreateVisionState(const MultiModalLanguageModel& model, const GeneratorParams& params) {
  if (ModelType::IsQwenVLFamily(model.config_->model.type)) {
    return std::make_unique<QwenVisionState>(model, params);
  }
  return std::make_unique<VisionState>(model, params);
}

SpeechState::SpeechState(const MultiModalLanguageModel& model, const GeneratorParams& params)
    : State{params, model},
      model_{model} {}

void SpeechState::SetExtraInputs(const std::vector<ExtraInput>& extra_inputs, const int64_t num_audio_tokens) {
  num_audio_tokens_ = num_audio_tokens;

  audio_features_ = std::make_unique<MultiModalFeatures>(*this, MultiModalFeatures::Mode::Output,  // Model output
                                                         model_.config_->model.speech.outputs.audio_features,
                                                         -1, num_audio_tokens_);
  audio_features_->Add();
  extra_inputs_.Add(extra_inputs, model_.speech_session_->GetInputNames());
}

DeviceSpan<float> SpeechState::Run(int current_length, DeviceSpan<int32_t>& next_tokens, DeviceSpan<int32_t> next_indices) {
  if (model_.config_->model.speech.run_options.has_value()) {
    State::SetRunOptions(model_.config_->model.speech.run_options.value());
  }
  State::Run(*model_.speech_session_);
  return {};
}

EmbeddingState::EmbeddingState(const MultiModalLanguageModel& model, const GeneratorParams& params)
    : State{params, model},
      model_{model} {
  input_ids_.Add();
  inputs_embeds_.Add();
}

void EmbeddingState::SetExtraInputs(const int64_t num_images, const int64_t num_image_tokens, const int64_t num_audio_tokens) {
  num_image_tokens_ = num_image_tokens;
  num_audio_tokens_ = num_audio_tokens;

  if (model_.vision_session_) {
    image_features_ = std::make_unique<MultiModalFeatures>(*this, MultiModalFeatures::Mode::Input,  // Optional model input
                                                           model_.config_->model.embedding.inputs.image_features,
                                                           num_images, num_image_tokens_);
    image_features_->Add();
  }
  if (model_.speech_session_) {
    audio_features_ = std::make_unique<MultiModalFeatures>(*this, MultiModalFeatures::Mode::Input,  // Optional model input
                                                           model_.config_->model.embedding.inputs.audio_features,
                                                           -1, num_audio_tokens_);
    audio_features_->Add();
  }
}

void EmbeddingState::UpdateInputsOutputs(DeviceSpan<int32_t>& next_tokens, bool is_prompt) {
  input_ids_.Update(next_tokens);
  if (model_.vision_session_) image_features_->Update(is_prompt);
  if (model_.speech_session_) audio_features_->Update(is_prompt);
}

DeviceSpan<float> EmbeddingState::Run(int current_length, DeviceSpan<int32_t>& next_tokens, DeviceSpan<int32_t> next_indices) {
  if (model_.config_->model.embedding.run_options.has_value()) {
    State::SetRunOptions(model_.config_->model.embedding.run_options.value());
  }
  State::Run(*model_.embedding_session_);
  return {};
}

DecoderState::DecoderState(const MultiModalLanguageModel& model, DeviceSpan<int32_t> sequence_lengths, const GeneratorParams& params)
    : State{params, model},
      model_{model},
      position_inputs_{CreatePositionInputs(*this, sequence_lengths, model_.config_->model.decoder.inputs.attention_mask)} {
  inputs_embeds_.Add();
  position_inputs_->Add();
  logits_.Add();
  kv_cache_.Add();
}

DeviceSpan<float> DecoderState::Run(int current_length, DeviceSpan<int32_t>& next_tokens, DeviceSpan<int32_t> next_indices) {
  if (model_.config_->model.decoder.run_options.has_value()) {
    State::SetRunOptions(model_.config_->model.decoder.run_options.value());
  }

  bool graph_capture_this_run = params_->use_graph_capture && inputs_embeds_.GetShape()[1] == 1;
  State::Run(*model_.decoder_session_, graph_capture_this_run);
  return logits_.Get();
}

void DecoderState::UpdateInputsOutputs(DeviceSpan<int32_t>& next_tokens, int total_length, DeviceSpan<int32_t> beam_indices) {
  int batch_size = static_cast<int>(inputs_embeds_.GetShape()[0]);
  size_t new_length = next_tokens.size() / batch_size;
  position_inputs_->Update(next_tokens, total_length, static_cast<int>(new_length));
  kv_cache_.Update(beam_indices, total_length);
  logits_.Update(next_tokens, new_length);
  inputs_embeds_.UpdateSequenceLength(new_length);
}

// Overload for pipeline to call
void DecoderState::UpdateInputsOutputs(DeviceSpan<int32_t>& next_tokens, int total_length, DeviceSpan<int32_t> beam_indices, size_t new_length) {
  kv_cache_.Update(beam_indices, total_length);
  logits_.Update(next_tokens, new_length);
  inputs_embeds_.UpdateSequenceLength(new_length);
}

MultiModalPipelineState::MultiModalPipelineState(const MultiModalLanguageModel& model, DeviceSpan<int32_t> sequence_lengths, const GeneratorParams& params)
    : State{params, model},
      model_{model},
      adapters_{std::make_shared<Adapters>(&model_)} {
  if (model_.vision_session_) {
    vision_state_ = CreateVisionState(model_, params);
  }
  if (model_.speech_session_) {
    speech_state_ = std::make_unique<SpeechState>(model_, params);
  }
  embedding_state_ = std::make_unique<EmbeddingState>(model, params);
  decoder_state_ = std::make_unique<DecoderState>(model_, sequence_lengths, params);

  if (vision_state_ != nullptr && model_.config_->model.vision.adapter_filename.has_value() && num_image_tokens_ > 0) {
    const auto lora_adapter = (model_.config_->config_path / fs::path(*model_.config_->model.vision.adapter_filename)).string();
    adapters_->LoadAdapter(lora_adapter.c_str(), vision_adapter_name_);
    decoder_state_->SetActiveAdapter(adapters_.get(), vision_adapter_name_);
  } else if (speech_state_ != nullptr && model_.config_->model.speech.adapter_filename.has_value() && num_audio_tokens_ > 0) {
    const auto lora_adapter = (model_.config_->config_path / fs::path(*model_.config_->model.speech.adapter_filename)).string();
    adapters_->LoadAdapter(lora_adapter.c_str(), speech_adapter_name_);
    decoder_state_->SetActiveAdapter(adapters_.get(), speech_adapter_name_);
  }
}

void MultiModalPipelineState::SetExtraInputs(const std::vector<ExtraInput>& extra_inputs) {
  num_image_tokens_ = GetNumImageTokens(extra_inputs);
  num_audio_tokens_ = GetNumAudioTokens(extra_inputs, model_.config_->model.speech.inputs.audio_sizes);
  num_images_ = GetImageFeatureBatchSize(extra_inputs);

  if (model_.vision_session_) {
    vision_state_->SetExtraInputs(extra_inputs, num_images_, num_image_tokens_);
  }
  if (model_.speech_session_) {
    speech_state_->SetExtraInputs(extra_inputs, num_audio_tokens_);
  }
  embedding_state_->SetExtraInputs(num_images_, num_image_tokens_, num_audio_tokens_);
  // Set the grid tensors for Qwen2-VL if present
  if (auto* qwen_pos_inputs = dynamic_cast<Qwen2VLPositionInputs*>(decoder_state_->position_inputs_.get())) {
    std::shared_ptr<Tensor> img_grid, vid_grid, sec_grid;

    for (const auto& input : extra_inputs) {
      if (input.name == Config::Defaults::ImageGridThwName) {
        img_grid = input.tensor;
      } else if (input.name == "video_grid_thw") {
        vid_grid = input.tensor;
      } else if (input.name == "second_per_grid_ts") {
        sec_grid = input.tensor;
      }
    }

    if (img_grid || vid_grid) {
      qwen_pos_inputs->SetGridTensors(img_grid, vid_grid, sec_grid);
    }
  }
}

DeviceSpan<float> MultiModalPipelineState::Run(int current_length, DeviceSpan<int32_t>& next_tokens, DeviceSpan<int32_t> next_indices) {
  // Pipeline state defines the pipeline of the execution of the models
  // Prompt stage:
  //   - pixel_values, [image_attention_mask], image_sizes -> |vision_model| -> image_features
  //   - audio_embeds, audio_sizes, audio_projection_mode -> |audio_model| -> audio_features
  //   - input_ids, image_features, audio_features -> |embeddings_model| -> inputs_embeds
  //   - inputs_embeds -> |decoder_model| -> logits
  // Generation stage:
  //   - input_ids, image_features, audio_features -> |embeddings_model| -> inputs_embeds
  //   - inputs_embeds -> |decoder_model| -> logits

  embedding_state_->UpdateInputsOutputs(next_tokens, is_prompt_);
  decoder_state_->UpdateInputsOutputs(next_tokens, current_length, next_indices);

  if (is_prompt_) {
    if (num_image_tokens_ > 0 && vision_state_) {
      vision_state_->Run(current_length, next_tokens, next_indices);
    }
    if (num_audio_tokens_ > 0 && speech_state_) {
      speech_state_->Run(current_length, next_tokens, next_indices);
    }
    if (vision_state_) {
      embedding_state_->image_features_->ReuseFeaturesBuffer(*vision_state_->image_features_);
    }
    if (speech_state_) embedding_state_->audio_features_->ReuseFeaturesBuffer(*speech_state_->audio_features_);
    embedding_state_->inputs_embeds_.ReuseEmbeddingsBuffer(decoder_state_->inputs_embeds_);
    embedding_state_->Run(current_length, next_tokens, next_indices);

    auto logits = decoder_state_->Run(current_length, next_tokens, next_indices);

    is_prompt_ = false;
    if (vision_state_) vision_state_.reset();  // The vision state is no longer needed in generation stage
    if (speech_state_) speech_state_.reset();  // The speech state is no longer needed in generation stage

    return logits;
  }

  embedding_state_->inputs_embeds_.ReuseEmbeddingsBuffer(decoder_state_->inputs_embeds_);
  embedding_state_->Run(current_length, next_tokens, next_indices);
  return decoder_state_->Run(current_length, next_tokens, next_indices);
}

OrtValue* MultiModalPipelineState::GetInput(const char* name) {
  if (vision_state_) {
    // Check if input name is in vision state's inputs
    for (size_t i = 0; i < vision_state_->input_names_.size(); i++) {
      if (std::strcmp(vision_state_->input_names_[i], name) == 0) {
        return vision_state_->inputs_[i];
      }
    }
  }

  if (speech_state_) {
    // Check if input name is in speech state's inputs
    for (size_t i = 0; i < speech_state_->input_names_.size(); i++) {
      if (std::strcmp(speech_state_->input_names_[i], name) == 0) {
        return speech_state_->inputs_[i];
      }
    }
  }

  // Check if input name is in embedding state's inputs
  for (size_t i = 0; i < embedding_state_->input_names_.size(); i++) {
    if (std::strcmp(embedding_state_->input_names_[i], name) == 0) {
      return embedding_state_->inputs_[i];
    }
  }

  // Check if input name is in decoder state's inputs
  for (size_t i = 0; i < decoder_state_->input_names_.size(); i++) {
    if (std::strcmp(decoder_state_->input_names_[i], name) == 0) {
      return decoder_state_->inputs_[i];
    }
  }

  return State::GetInput(name);
};

OrtValue* MultiModalPipelineState::GetOutput(const char* name) {
  if (vision_state_) {
    // Check if output name is in vision state's outputs
    for (size_t i = 0; i < vision_state_->output_names_.size(); i++) {
      if (std::strcmp(vision_state_->output_names_[i], name) == 0) {
        return vision_state_->outputs_[i];
      }
    }
  }

  if (speech_state_) {
    // Check if output name is in speech state's outputs
    for (size_t i = 0; i < speech_state_->output_names_.size(); i++) {
      if (std::strcmp(speech_state_->output_names_[i], name) == 0) {
        return speech_state_->outputs_[i];
      }
    }
  }

  // Check if output name is in embedding state's outputs
  for (size_t i = 0; i < embedding_state_->output_names_.size(); i++) {
    if (std::strcmp(embedding_state_->output_names_[i], name) == 0) {
      return embedding_state_->outputs_[i];
    }
  }

  // Check if output name is in decoder state's outputs
  for (size_t i = 0; i < decoder_state_->output_names_.size(); i++) {
    if (std::strcmp(decoder_state_->output_names_[i], name) == 0) {
      return decoder_state_->outputs_[i];
    }
  }

  return State::GetOutput(name);
};

}  // namespace Generators
