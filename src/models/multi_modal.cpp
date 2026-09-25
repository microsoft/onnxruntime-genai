// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "generator/generators.h"
#include "multi_modal.h"
#include "models/io/default_position_inputs.h"
#include "models/io/qwen_vl_position_inputs.h"
#include "qwen_vl_state.h"
#include <cstring>
#include <algorithm>
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

// Returns the number of images in the current batch. Most processors retain an
// image batch dimension; model-specific flattened layouts provide metadata.
int64_t GetImageFeatureBatchSize(const std::vector<ExtraInput>& extra_inputs) {
  for (size_t i = 0; i < extra_inputs.size(); ++i) {
    if (extra_inputs[i].name == Config::Defaults::PixelValuesName) {
      assert(extra_inputs[i].tensor->ort_tensor_);
      const auto num_dims = extra_inputs[i].tensor->ort_tensor_->GetTensorTypeAndShapeInfo()->GetShape().size();
      if (num_dims >= 3) {
        return extra_inputs[i].tensor->ort_tensor_->GetTensorTypeAndShapeInfo()->GetShape().front();
      }
      break;
    }
  }

  return GetQwenImageCount(extra_inputs);
}

}  // namespace

void CheckLfm2AudioSessionDevices(const Config& config, DeviceType decoder_device, DeviceType inputs_device,
                                  bool with_audio) {
  // Whether a buffer allocated on the device is device memory. OpenVINO allocates from the CPU and
  // QNN from shared memory, both of which a CPU session can use.
  const auto is_device_memory = [](DeviceType device) {
    switch (device) {
      case DeviceType::CUDA:
      case DeviceType::DML:
      case DeviceType::WEBGPU:
      case DeviceType::NvTensorRtRtx:
      case DeviceType::RyzenAI:
      case DeviceType::AMDGPU:
        return true;
      default:
        return false;
    }
  };
  const auto check = [](const std::optional<Config::SessionOptions>& options, const char* graph,
                        const char* buffers, DeviceType device) {
    // Without session_options of its own a graph follows the decoder; with them, no provider but CPU
    // leaves it on CPU.
    if (!options.has_value() || !std::all_of(options->providers.begin(), options->providers.end(),
                                             [](const std::string& provider) { return provider == "CPU"; })) {
      return;
    }
    throw std::runtime_error(std::string("lfm2_audio: model.") + graph + ".session_options run the " + graph +
                             " model on CPU, but " + buffers + " in " + to_string(device) +
                             " memory, which it cannot use. Remove model." + graph +
                             ".session_options so that it runs on the decoder's device.");
  };

  if (is_device_memory(inputs_device)) {
    check(config.model.embedding.session_options, "embedding", "the decoder takes its inputs", inputs_device);
  }
  if (with_audio && is_device_memory(decoder_device)) {
    check(config.model.speech.session_options, "speech", "the audio features are passed", decoder_device);
    check(config.model.embedding.session_options, "embedding", "the audio features are passed", decoder_device);
  }
}

MultiModalLanguageModel::MultiModalLanguageModel(std::unique_ptr<Config> config, OrtEnv& ort_env, bool vision, bool speech)
    : Model(std::move(config)) {
  if (config_->model.type == "lfm2_audio") {
    CheckLfm2AudioSessionDevices(*config_, p_device_->GetType(), p_device_inputs_->GetType(), /*with_audio=*/false);
  }
  // The non-decoder models don't support graph capture because of control flow nodes, so disable graph capture for them
  if (vision) {
    vision_session_options_ = OrtSessionOptions::Create();
    CreateSessionOptionsFromConfig(config_->model.vision.session_options.has_value() ? config_->model.vision.session_options.value() : config_->model.decoder.session_options, *vision_session_options_, true, /*disable_graph_capture=*/true);
    vision_session_ = CreateSession(ort_env, config_->model.vision.filename, vision_session_options_.get());
  }

  if (speech) {
    speech_session_options_ = OrtSessionOptions::Create();
    CreateSessionOptionsFromConfig(config_->model.speech.session_options.has_value() ? config_->model.speech.session_options.value() : config_->model.decoder.session_options, *speech_session_options_, true, /*disable_graph_capture=*/true);
    speech_session_ = CreateSession(ort_env, config_->model.speech.filename, speech_session_options_.get());
  }

  embedding_session_options_ = OrtSessionOptions::Create();
  CreateSessionOptionsFromConfig(config_->model.embedding.session_options.has_value() ? config_->model.embedding.session_options.value() : config_->model.decoder.session_options, *embedding_session_options_, true, /*disable_graph_capture=*/true);

  embedding_session_ = CreateSession(ort_env, config_->model.embedding.filename, embedding_session_options_.get());
  decoder_session_ = CreateSession(ort_env, config_->model.decoder.filename, session_options_.get());

  const auto& audio_output = config_->model.audio_output;
  if (!audio_output.depthformer.filename.empty() || !audio_output.embedding.filename.empty()) {
    if (audio_output.depthformer.filename.empty() || audio_output.embedding.filename.empty()) {
      throw std::runtime_error("model.audio_output needs both depthformer.filename and embedding.filename.");
    }
    // With speech output these tokens hand the turn to the depthformer; as stop tokens they would end
    // it there instead, which is what a text-only export needs and this one must not have.
    for (const int token : {audio_output.audio_start_token_id, audio_output.text_end_token_id}) {
      if (std::find(config_->model.eos_token_id.begin(), config_->model.eos_token_id.end(), token) !=
          config_->model.eos_token_id.end()) {
        throw std::runtime_error("model.eos_token_id holds " + std::to_string(token) +
                                 ", which model.audio_output uses to start speech. Remove it from eos_token_id.");
      }
    }
    depthformer_session_options_ = OrtSessionOptions::Create();
    CreateSessionOptionsFromConfig(audio_output.depthformer.session_options.has_value() ? audio_output.depthformer.session_options.value() : config_->model.decoder.session_options, *depthformer_session_options_, true, /*disable_graph_capture=*/true);
    depthformer_session_ = CreateSession(ort_env, audio_output.depthformer.filename, depthformer_session_options_.get());
    audio_embedding_session_options_ = OrtSessionOptions::Create();
    CreateSessionOptionsFromConfig(audio_output.embedding.session_options.has_value() ? audio_output.embedding.session_options.value() : config_->model.decoder.session_options, *audio_embedding_session_options_, true, /*disable_graph_capture=*/true);
    audio_embedding_session_ = CreateSession(ort_env, audio_output.embedding.filename, audio_embedding_session_options_.get());
  }

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
// PixtralVisionState: per-image slicing loop for Pixtral / Mistral3
// ---------------------------------------------------------------------------

void PixtralVisionState::SetExtraInputs(const std::vector<ExtraInput>& extra_inputs,
                                        const int64_t num_images,
                                        const int64_t num_image_tokens) {
  // Extract image_sizes[N, 2] before the base class filters extra_inputs
  // by vision session input names (image_sizes is metadata, not a vision input).
  image_heights_.clear();
  image_widths_.clear();
  for (const auto& input : extra_inputs) {
    if (input.name == Config::Defaults::ImageSizesName && input.tensor->ort_tensor_) {
      auto shape = input.tensor->ort_tensor_->GetTensorTypeAndShapeInfo()->GetShape();
      if (shape.size() != 2 || shape[1] != 2)
        throw std::runtime_error(
            "PixtralVisionState: image_sizes must be [N, 2], got [" +
            std::to_string(shape.size() > 0 ? shape[0] : 0) + ", " +
            std::to_string(shape.size() > 1 ? shape[1] : 0) + "]");
      const int64_t* data = input.tensor->ort_tensor_->GetTensorData<int64_t>();
      int64_t n = shape[0];
      for (int64_t i = 0; i < n; ++i) {
        image_heights_.push_back(data[i * 2]);
        image_widths_.push_back(data[i * 2 + 1]);
      }
      break;
    }
  }

  VisionState::SetExtraInputs(extra_inputs, num_images, num_image_tokens);
}

DeviceSpan<float> PixtralVisionState::Run(int current_length, DeviceSpan<int32_t>& next_tokens,
                                          DeviceSpan<int32_t> next_indices) {
  if (model_.config_->model.vision.run_options.has_value()) {
    State::SetRunOptions(model_.config_->model.vision.run_options.value());
  }

  // Single-image inputs can run vision.onnx directly.
  if (num_images_ <= 1) {
    State::Run(*model_.vision_session_);
    return {};
  }

  if (image_heights_.empty() || image_widths_.empty()) {
    throw std::runtime_error(
        "PixtralVisionState: multi-image inputs require image_sizes metadata");
  }

  if (static_cast<int64_t>(image_heights_.size()) < num_images_)
    throw std::runtime_error(
        "PixtralVisionState: image_heights_ has " + std::to_string(image_heights_.size()) +
        " entries but num_images_ is " + std::to_string(num_images_));

  if (static_cast<int64_t>(image_widths_.size()) < num_images_)
    throw std::runtime_error(
        "PixtralVisionState: image_widths_ has " + std::to_string(image_widths_.size()) +
        " entries but num_images_ is " + std::to_string(num_images_));

  // Multi-image: pixel_values is [N, C, H_max, W_max] with zero-padding.
  // image_heights_/image_widths_ hold the actual per-image dimensions.
  // Run vision.onnx once per image with [1, C, H_i, W_i].
  const std::string& pv_name = model_.config_->model.vision.inputs.pixel_values;

  size_t pv_idx = SIZE_MAX;
  for (size_t i = 0; i < input_names_.size(); ++i) {
    if (input_names_[i] == pv_name) {
      pv_idx = i;
      break;
    }
  }

  if (pv_idx == SIZE_MAX) {
    State::Run(*model_.vision_session_);
    return {};
  }

  OrtValue* pv_full = inputs_[pv_idx];
  OrtValue* feat_full = outputs_[0];

  auto pv_info = pv_full->GetTensorTypeAndShapeInfo();
  auto pv_shape = pv_info->GetShape();  // [N, C, H_max, W_max]
  auto pv_type = pv_info->GetElementType();

  if (pv_shape.size() != 4) {
    throw std::runtime_error(
        "PixtralVisionState: expected 4D pixel_values [N,C,H,W], got " +
        std::to_string(pv_shape.size()) + "D");
  }

  int64_t channels = pv_shape[1];
  int64_t h_max = pv_shape[2];
  int64_t w_max = pv_shape[3];

  auto feat_info = feat_full->GetTensorTypeAndShapeInfo();
  auto feat_shape = feat_info->GetShape();  // [total_features, hidden_size]
  auto feat_type = feat_info->GetElementType();
  int64_t hidden_size = feat_shape.back();

  auto element_size = [](ONNXTensorElementDataType type) -> size_t {
    switch (type) {
      case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT:
        return 4;
      case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16:
        return 2;
      case ONNX_TENSOR_ELEMENT_DATA_TYPE_BFLOAT16:
        return 2;
      default:
        throw std::runtime_error("PixtralVisionState: unsupported element type");
    }
  };
  size_t pv_elem_size = element_size(pv_type);
  size_t feat_elem_size = element_size(feat_type);

  uint8_t* pv_raw = static_cast<uint8_t*>(pv_full->GetTensorMutableRawData());

  int64_t feat_offset = 0;
  size_t image_stride = static_cast<size_t>(channels * h_max * w_max) * pv_elem_size;

  // TODO: Explore batching multiple images through the vision encoder to improve
  // throughput. Currently processes one image at a time due to variable image
  // resolutions producing different patch counts and 2D RoPE position grids.
  // Potential approach: pad to uniform size or restructure the ONNX graph for
  // batched inputs.
  for (int64_t img = 0; img < num_images_; ++img) {
    int64_t h_i = image_heights_[img];
    int64_t w_i = image_widths_[img];

    if (h_i <= 0 || h_i > h_max)
      throw std::runtime_error(
          "PixtralVisionState: image " + std::to_string(img) + " has h_i=" +
          std::to_string(h_i) + " which is out of valid range (0, " +
          std::to_string(h_max) + "]");
    if (w_i <= 0 || w_i > w_max)
      throw std::runtime_error(
          "PixtralVisionState: image " + std::to_string(img) + " has w_i=" +
          std::to_string(w_i) + " which is out of valid range (0, " +
          std::to_string(w_max) + "]");

    // Create a contiguous [1, C, H_i, W_i] tensor by copying valid rows
    // from the zero-padded [N, C, H_max, W_max] buffer.
    std::vector<int64_t> sub_pv_shape = {1, channels, h_i, w_i};
    auto sub_pv = OrtValue::CreateTensor(
        Ort::Allocator::GetWithDefaultOptions(), sub_pv_shape, pv_type);
    uint8_t* sub_pv_data = static_cast<uint8_t*>(sub_pv->GetTensorMutableRawData());

    uint8_t* src_image = pv_raw + img * image_stride;
    size_t dst_offset = 0;
    for (int64_t c = 0; c < channels; ++c) {
      uint8_t* src_channel = src_image + static_cast<size_t>(c * h_max * w_max) * pv_elem_size;
      for (int64_t row = 0; row < h_i; ++row) {
        size_t row_bytes = static_cast<size_t>(w_i) * pv_elem_size;
        std::memcpy(sub_pv_data + dst_offset,
                    src_channel + static_cast<size_t>(row * w_max) * pv_elem_size,
                    row_bytes);
        dst_offset += row_bytes;
      }
    }

    // Compute expected feature count for this image
    int64_t patch_size = model_.config_->model.vision.patch_size;
    int64_t merge_size = model_.config_->model.vision.spatial_merge_size;
    int64_t num_feats = (h_i / patch_size / merge_size) * (w_i / patch_size / merge_size);

    int64_t total_feats = feat_shape[0];
    if (feat_offset + num_feats > total_feats)
      throw std::runtime_error(
          "PixtralVisionState: feat_offset (" + std::to_string(feat_offset) +
          ") + num_feats (" + std::to_string(num_feats) +
          ") exceeds pre-allocated feature buffer (" + std::to_string(total_feats) + ")");

    // Run into a separate output tensor, then copy into the combined feature buffer.
    std::vector<int64_t> sub_feat_shape = {num_feats, hidden_size};
    auto sub_feat = OrtValue::CreateTensor(
        model_.p_device_->GetAllocator(), sub_feat_shape, feat_type);

    inputs_[pv_idx] = sub_pv.get();
    outputs_[0] = sub_feat.get();

    State::Run(*model_.vision_session_);

    size_t feature_offset_bytes = static_cast<size_t>(feat_offset * hidden_size) * feat_elem_size;
    size_t feature_size_bytes = static_cast<size_t>(num_feats * hidden_size) * feat_elem_size;
    ByteWrapTensor(*model_.p_device_, *feat_full)
        .subspan(feature_offset_bytes, feature_size_bytes)
        .CopyFrom(ByteWrapTensor(*model_.p_device_, *sub_feat));

    feat_offset += num_feats;
  }

  // Restore original pointers
  inputs_[pv_idx] = pv_full;
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
  if (ModelType::IsPixtralFamily(model.config_->model.type)) {
    return std::make_unique<PixtralVisionState>(model, params);
  }
  return std::make_unique<VisionState>(model, params);
}

SpeechState::SpeechState(const MultiModalLanguageModel& model, const GeneratorParams& params)
    : State{params, model},
      model_{model} {}

void SpeechState::SetExtraInputs(const std::vector<ExtraInput>& extra_inputs, const int64_t num_audio_tokens) {
  num_audio_tokens_ = num_audio_tokens;

  // Allocate 3D [batch, num_audio_tokens, hidden_size] matching the speech ONNX model's
  // output rank. Will be reshaped to 2D before passing to the embedding model.
  audio_features_ = std::make_unique<MultiModalFeatures>(*this, MultiModalFeatures::Mode::Output,
                                                         model_.config_->model.speech.outputs.audio_features,
                                                         params_->BatchBeamSize(), num_audio_tokens_);
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

void Lfm2AudioSpeechState::SetExtraInputs(const std::vector<ExtraInput>& extra_inputs, const int64_t num_audio_tokens) {
  // The feature buffer is one sequence wide, so beams would fail as a shape mismatch inside the
  // encoder. The reference decodes greedily.
  if (params_->search.num_beams > 1) {
    throw std::runtime_error("Lfm2AudioSpeechState: beam search is not supported for lfm2_audio; got num_beams " +
                             std::to_string(params_->search.num_beams) + ". Set num_beams to 1.");
  }

  SpeechState::SetExtraInputs(extra_inputs, num_audio_tokens);

  // audio_sizes holds the decoder tokens each clip contributes; its sum is num_audio_tokens.
  tokens_per_clip_.clear();
  for (const auto& input : extra_inputs) {
    if (input.name == model_.config_->model.speech.inputs.audio_sizes) {
      const auto info = input.tensor->ort_tensor_->GetTensorTypeAndShapeInfo();
      const int64_t* sizes = input.tensor->ort_tensor_->GetTensorData<int64_t>();
      tokens_per_clip_.assign(sizes, sizes + info->GetElementCount());
      break;
    }
  }
}

Lfm2AudioSpeechState::SpeechBindings Lfm2AudioSpeechState::ResolveBindings() const {
  SpeechBindings bindings;
  bindings.mel_index = FindInput(model_.config_->model.speech.inputs.audio_embeds);
  bindings.lengths_index = FindInput(model_.config_->model.speech.inputs.audio_lengths);
  bindings.features_index = FindOutput(model_.config_->model.speech.outputs.audio_features);

  const auto mel_info = inputs_[bindings.mel_index]->GetTensorTypeAndShapeInfo();
  const auto mel_shape = mel_info->GetShape();  // [num_clips, longest_clip, num_mels]
  if (mel_shape.size() != 3) {
    throw std::runtime_error("Lfm2AudioSpeechState: expected a 3D [num_clips, num_frames, num_mels] mel tensor, got rank " +
                             std::to_string(mel_shape.size()) + ".");
  }
  bindings.num_clips = mel_shape[0];
  bindings.longest_clip = mel_shape[1];
  bindings.num_mels = mel_shape[2];
  bindings.mel_type = mel_info->GetElementType();

  if (bindings.num_clips != static_cast<int64_t>(tokens_per_clip_.size())) {
    throw std::runtime_error("Lfm2AudioSpeechState: the mel tensor holds " + std::to_string(bindings.num_clips) +
                             " clips but audio_sizes has " + std::to_string(tokens_per_clip_.size()) + " entries.");
  }
  const auto lengths_info = inputs_[bindings.lengths_index]->GetTensorTypeAndShapeInfo();
  if (static_cast<int64_t>(lengths_info->GetElementCount()) != bindings.num_clips) {
    throw std::runtime_error("Lfm2AudioSpeechState: the mel tensor holds " + std::to_string(bindings.num_clips) +
                             " clips but " + model_.config_->model.speech.inputs.audio_lengths + " has " +
                             std::to_string(lengths_info->GetElementCount()) + " entries.");
  }

  const auto features_info = outputs_[bindings.features_index]->GetTensorTypeAndShapeInfo();
  const auto features_shape = features_info->GetShape();  // [1, num_audio_tokens, hidden_size]
  bindings.features_type = features_info->GetElementType();
  bindings.hidden_size = features_shape.back();
  return bindings;
}

std::unique_ptr<OrtValue> Lfm2AudioSpeechState::RunClip(const SpeechBindings& bindings, int64_t index,
                                                        int64_t num_frames) {
  const size_t mel_element_size = Ort::SizeOf(bindings.mel_type);
  const size_t clip_stride = static_cast<size_t>(bindings.longest_clip * bindings.num_mels) * mel_element_size;
  const auto* mel_data = static_cast<const uint8_t*>(inputs_[bindings.mel_index]->GetTensorRawData());

  auto clip_mel = OrtValue::CreateTensor(Ort::Allocator::GetWithDefaultOptions(),
                                         std::vector<int64_t>{1, num_frames, bindings.num_mels}, bindings.mel_type);
  std::memcpy(clip_mel->GetTensorMutableRawData(), mel_data + static_cast<size_t>(index) * clip_stride,
              static_cast<size_t>(num_frames * bindings.num_mels) * mel_element_size);

  auto clip_length = OrtValue::CreateTensor<int64_t>(Ort::Allocator::GetWithDefaultOptions(), std::vector<int64_t>{1});
  clip_length->GetTensorMutableData<int64_t>()[0] = num_frames;

  auto clip_features = OrtValue::CreateTensor(
      model_.p_device_->GetAllocator(),
      std::vector<int64_t>{1, tokens_per_clip_[static_cast<size_t>(index)], bindings.hidden_size},
      bindings.features_type);

  OrtValue* mel_batch = inputs_[bindings.mel_index];
  OrtValue* lengths = inputs_[bindings.lengths_index];
  OrtValue* features = outputs_[bindings.features_index];
  inputs_[bindings.mel_index] = clip_mel.get();
  inputs_[bindings.lengths_index] = clip_length.get();
  outputs_[bindings.features_index] = clip_features.get();
  State::Run(*model_.speech_session_);
  inputs_[bindings.mel_index] = mel_batch;
  inputs_[bindings.lengths_index] = lengths;
  outputs_[bindings.features_index] = features;

  return clip_features;
}

DeviceSpan<float> Lfm2AudioSpeechState::Run(int current_length, DeviceSpan<int32_t>& next_tokens, DeviceSpan<int32_t> next_indices) {
  CheckLfm2AudioSessionDevices(*model_.config_, model_.p_device_->GetType(), model_.p_device_inputs_->GetType(),
                               /*with_audio=*/true);
  if (model_.config_->model.speech.run_options.has_value()) {
    State::SetRunOptions(model_.config_->model.speech.run_options.value());
  }
  // A single clip fills the whole mel tensor and the whole feature buffer; run it as it stands.
  if (tokens_per_clip_.size() <= 1) {
    State::Run(*model_.speech_session_);
    return {};
  }

  const SpeechBindings bindings = ResolveBindings();
  const int64_t* frames_per_clip = inputs_[bindings.lengths_index]->GetTensorData<int64_t>();
  auto features_bytes = ByteWrapTensor(*model_.p_device_, *outputs_[bindings.features_index]);
  const size_t feature_row_bytes = static_cast<size_t>(bindings.hidden_size) * Ort::SizeOf(bindings.features_type);
  size_t destination = 0;

  // The published encoder export is traced for one clip (its subsampling mask cannot broadcast over
  // a batch), so run it once per clip on that clip's own frames — which also keeps the padding out
  // of the encoder entirely — and concatenate the results in clip order.
  for (int64_t clip = 0; clip < bindings.num_clips; ++clip) {
    const int64_t num_frames = frames_per_clip[clip];
    if (num_frames <= 0 || num_frames > bindings.longest_clip) {
      throw std::runtime_error("Lfm2AudioSpeechState: clip " + std::to_string(clip) + " reports " +
                               std::to_string(num_frames) + " mel frames, outside the 1.." +
                               std::to_string(bindings.longest_clip) + " the mel tensor holds.");
    }

    auto clip_features = RunClip(bindings, clip, num_frames);
    const size_t clip_bytes = static_cast<size_t>(tokens_per_clip_[static_cast<size_t>(clip)]) * feature_row_bytes;
    features_bytes.subspan(destination, clip_bytes).CopyFrom(ByteWrapTensor(*model_.p_device_, *clip_features));
    destination += clip_bytes;
  }
  return {};
}

size_t Lfm2AudioSpeechState::FindInput(const std::string& name) const {
  for (size_t i = 0; i < input_names_.size(); ++i) {
    if (name == input_names_[i]) return i;
  }
  throw std::runtime_error("Lfm2AudioSpeechState: speech input \"" + name + "\" is not bound.");
}

size_t Lfm2AudioSpeechState::FindOutput(const std::string& name) const {
  for (size_t i = 0; i < output_names_.size(); ++i) {
    if (name == output_names_[i]) return i;
  }
  throw std::runtime_error("Lfm2AudioSpeechState: speech output \"" + name + "\" is not bound.");
}

std::unique_ptr<SpeechState> CreateSpeechState(const MultiModalLanguageModel& model, const GeneratorParams& params) {
  if (model.config_->model.type == "lfm2_audio") {
    return std::make_unique<Lfm2AudioSpeechState>(model, params);
  }
  return std::make_unique<SpeechState>(model, params);
}

EmbeddingState::EmbeddingState(const MultiModalLanguageModel& model, const GeneratorParams& params)
    : State{params, model},
      model_{model} {
  input_ids_.Add();
  inputs_embeds_.Add();

  // Gemma4: embedding model produces per_layer_inputs alongside inputs_embeds
  if (!model_.config_->model.embedding.outputs.per_layer_inputs.empty()) {
    auto shape = model_.session_info_.GetOutputShape(model_.config_->model.embedding.outputs.per_layer_inputs);
    int64_t per_layer_dim = shape.size() >= 3 ? shape.back() : 0;
    per_layer_inputs_ = std::make_unique<Embeddings>(*this, Embeddings::Mode::Output,
                                                     model_.config_->model.embedding.outputs.per_layer_inputs, per_layer_dim);
    per_layer_inputs_->Add();
  }
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
  } else if (model_.session_info_.HasInput(model_.config_->model.embedding.inputs.audio_features)) {
    // No speech session, but embedding model requires audio_features — provide empty tensor with shape (0, hidden_size)
    audio_features_ = std::make_unique<MultiModalFeatures>(*this, MultiModalFeatures::Mode::Input,
                                                           model_.config_->model.embedding.inputs.audio_features,
                                                           -1, 0);
    audio_features_->Add();
    // Pre-allocate an empty tensor since there's no speech session to provide one via ReuseFeaturesBuffer
    audio_features_->AllocateEmptyFeatures();
  }
}

void EmbeddingState::UpdateInputsOutputs(DeviceSpan<int32_t>& next_tokens, bool is_prompt) {
  input_ids_.Update(next_tokens);
  if (model_.vision_session_) image_features_->Update(is_prompt);
  if (audio_features_) audio_features_->Update(is_prompt);
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
      position_inputs_{model_.p_device_inputs_->CreatePositionInputs(*this, sequence_lengths, model_.config_->model.decoder.inputs.attention_mask)},
      kv_cache_{model_.p_device_kvcache_->CreateKeyValueCache(*this)},
      recurrent_state_{CreateRecurrentState(*this, /*graph_capture_variants_supported=*/true)} {
  inputs_embeds_.Add();

  // Gemma4: decoder accepts per_layer_inputs from the embedding model
  if (!model_.config_->model.decoder.inputs.per_layer_inputs.empty()) {
    auto shape = model_.session_info_.GetInputShape(model_.config_->model.decoder.inputs.per_layer_inputs);
    int64_t per_layer_dim = shape.size() >= 3 ? shape.back() : 0;
    per_layer_inputs_ = std::make_unique<Embeddings>(*this, Embeddings::Mode::Input,
                                                     model_.config_->model.decoder.inputs.per_layer_inputs, per_layer_dim);
    per_layer_inputs_->Add();
  }

  // Some multimodal decoders (e.g., Gemma4) require input_ids alongside inputs_embeds.
  // Use a decoder-only SessionInfo to avoid false positives: the combined session_info_
  // includes embedding session inputs (which always has input_ids), causing this check
  // to incorrectly fire for models like mistral3 whose decoder has no input_ids input.
  {
    SessionInfo decoder_only_info;
    decoder_only_info.Add(*model_.decoder_session_);
    if (decoder_only_info.HasInput(model_.config_->model.decoder.inputs.input_ids)) {
      decoder_input_ids_ = std::make_unique<DefaultInputIDs>(*this);
      decoder_input_ids_->Add();
    }
  }

  position_inputs_->Add();
  logits_.Add();
  if (kv_cache_)
    kv_cache_->Add();
  if (recurrent_state_)
    recurrent_state_->Add();
}

DeviceSpan<float> DecoderState::Run(int current_length, DeviceSpan<int32_t>& next_tokens, DeviceSpan<int32_t> next_indices) {
  if (model_.config_->model.decoder.run_options.has_value()) {
    State::SetRunOptions(model_.config_->model.decoder.run_options.value());
  }

  const int seq_len = static_cast<int>(inputs_embeds_.GetShape()[1]);
  const bool graph_capture_this_run = params_->use_graph_capture && seq_len == 1;
  const int graph_capture_variant = recurrent_state_ ? recurrent_state_->GraphCaptureVariant() : 0;

  const int graph_id = seq_len * 2 + graph_capture_variant;
  if (graph_capture_this_run && recurrent_state_ && recurrent_state_->ShouldFixUpGraphCapture(graph_id)) {
    recurrent_state_->SaveForGraphCapture();
    State::Run(*model_.decoder_session_, true, seq_len, graph_capture_variant);
    recurrent_state_->RestoreAfterGraphCapture(graph_id);
  }
  State::Run(*model_.decoder_session_, graph_capture_this_run, seq_len, graph_capture_variant);
  return logits_.Get();
}

bool DecoderState::SupportsPrefillChunking(bool has_multimodal_content) const {
  // Chunking slices the pre-computed embeddings along the sequence dimension, which is only
  // contiguous for a single sequence. Continuous decoding of position ids/attention mask in
  // DefaultPositionInputs is likewise restricted to a batch-beam size of one.
  if (params_->BatchBeamSize() != 1)
    return false;

  // DefaultPositionInputs produces position ids sequentially, so chunking is always safe.
  if (dynamic_cast<const DefaultPositionInputs*>(position_inputs_.get()) != nullptr)
    return true;

  // Qwen-VL's 3D mRoPE position ids diverge from sequential positions only when vision/audio
  // content shifts the rope deltas. A text-only prompt reduces to sequential positions, so
  // chunking is safe; with multimodal content the ids must be produced in a single full pass.
  if (dynamic_cast<const Qwen2VLPositionInputs*>(position_inputs_.get()) != nullptr)
    return !has_multimodal_content;

  // Any other position-input type (e.g. WindowedPositionInputs) keeps the conservative
  // single-pass prefill behavior.
  return false;
}

void DecoderState::PrepareEmbeddingsForPrefill(size_t new_length) {
  // Allocate the embeddings buffers for the whole prompt. The embedding model writes into these
  // buffers in one run; the decoder then consumes them chunk by chunk.
  inputs_embeds_.UpdateSequenceLength(new_length);
  if (per_layer_inputs_) per_layer_inputs_->UpdateSequenceLength(new_length);
}

DeviceSpan<float> DecoderState::RunPrefillWithChunking(int current_length, DeviceSpan<int32_t>& next_tokens,
                                                       DeviceSpan<int32_t> next_indices, size_t chunk_size) {
  if (model_.config_->model.decoder.run_options.has_value()) {
    State::SetRunOptions(model_.config_->model.decoder.run_options.value());
  }

  const size_t num_tokens = next_tokens.size();
  size_t processed_tokens = 0;
  int length = current_length - static_cast<int>(num_tokens);

  while (processed_tokens < num_tokens) {
    const size_t current_chunk_size = std::min(chunk_size, num_tokens - processed_tokens);
    auto chunk_tokens = next_tokens.subspan(processed_tokens, current_chunk_size);
    length += static_cast<int>(current_chunk_size);

    if (decoder_input_ids_) decoder_input_ids_->Update(chunk_tokens);
    position_inputs_->Update(chunk_tokens, length, static_cast<int>(current_chunk_size));
    kv_cache_->Update(next_indices, length);
    if (recurrent_state_)
      recurrent_state_->Update();
    logits_.Update(chunk_tokens, current_chunk_size);

    // Feed only this chunk's slice of the pre-computed embeddings to the decoder.
    inputs_embeds_.UseChunkView(processed_tokens, current_chunk_size);
    if (per_layer_inputs_) per_layer_inputs_->UseChunkView(processed_tokens, current_chunk_size);

    // Graph capture is disabled during prefill chunking.
    State::Run(*model_.decoder_session_, /*graph_capture_this_run=*/false);

    processed_tokens += current_chunk_size;
  }

  inputs_embeds_.RestoreFullView();
  if (per_layer_inputs_) per_layer_inputs_->RestoreFullView();

  // Logits of the last chunk contain the logits for the last prompt token.
  return logits_.Get();
}

void DecoderState::UpdateInputsOutputs(DeviceSpan<int32_t>& next_tokens, int total_length, DeviceSpan<int32_t> beam_indices) {
  int batch_size = static_cast<int>(inputs_embeds_.GetShape()[0]);
  size_t new_length = next_tokens.size() / batch_size;
  if (decoder_input_ids_) decoder_input_ids_->Update(next_tokens);
  position_inputs_->Update(next_tokens, total_length, static_cast<int>(new_length));
  if (kv_cache_)
    kv_cache_->Update(beam_indices, total_length);
  if (recurrent_state_)
    recurrent_state_->Update();
  logits_.Update(next_tokens, new_length);
  inputs_embeds_.UpdateSequenceLength(new_length);
  if (per_layer_inputs_) per_layer_inputs_->UpdateSequenceLength(new_length);
}

// Overload for pipeline to call
void DecoderState::UpdateInputsOutputs(DeviceSpan<int32_t>& next_tokens, int total_length, DeviceSpan<int32_t> beam_indices, size_t new_length) {
  if (decoder_input_ids_) decoder_input_ids_->Update(next_tokens);
  if (kv_cache_)
    kv_cache_->Update(beam_indices, total_length);
  if (recurrent_state_)
    recurrent_state_->Update();
  logits_.Update(next_tokens, new_length);
  inputs_embeds_.UpdateSequenceLength(new_length);
  if (per_layer_inputs_) per_layer_inputs_->UpdateSequenceLength(new_length);
}

MultiModalPipelineState::MultiModalPipelineState(const MultiModalLanguageModel& model, DeviceSpan<int32_t> sequence_lengths, const GeneratorParams& params)
    : State{params, model},
      model_{model},
      adapters_{std::make_shared<Adapters>(&model_)} {
  if (model_.vision_session_) {
    vision_state_ = CreateVisionState(model_, params);
  }
  if (model_.speech_session_) {
    speech_state_ = CreateSpeechState(model_, params);
  }
  embedding_state_ = std::make_unique<EmbeddingState>(model, params);
  decoder_state_ = std::make_unique<DecoderState>(model_, sequence_lengths, params);
  if (model_.depthformer_session_) {
    audio_output_ = std::make_unique<Lfm2AudioOutput>(model_, params);
  }

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

  if (audio_output_) {
    audio_output_->BeginStep(next_tokens, is_prompt_);
  }
  embedding_state_->UpdateInputsOutputs(next_tokens, is_prompt_);

  // Prefill chunking (search.chunk_size): during the prompt stage the decoder can process the
  // prompt embeddings in several smaller runs to bound peak memory usage.
  const auto& chunk_size_opt = params_->search.chunk_size;
  const size_t num_tokens = next_tokens.size();
  const bool has_multimodal_content = num_image_tokens_ != 0 || num_audio_tokens_ != 0;
  const bool chunk_prefill = is_prompt_ && chunk_size_opt.has_value() && chunk_size_opt.value() > 0 &&
                             num_tokens > chunk_size_opt.value() && decoder_state_->SupportsPrefillChunking(has_multimodal_content);

  if (chunk_prefill) {
    decoder_state_->PrepareEmbeddingsForPrefill(num_tokens);
  } else {
    decoder_state_->UpdateInputsOutputs(next_tokens, current_length, next_indices);
  }

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
    if (speech_state_ && num_audio_tokens_ > 0) {
      // Reshape speech output from 3D [B, T, hidden] to 2D [B*T, hidden]
      // to match embedding model's expected 2D audio_features input rank.
      auto& speech_shape = speech_state_->audio_features_->GetShape();
      if (speech_shape.size() == 3) {
        speech_state_->audio_features_->ReshapeFeatures(
            {speech_shape[0] * speech_shape[1], speech_shape[2]});
      }
      embedding_state_->audio_features_->ReuseFeaturesBuffer(*speech_state_->audio_features_);
    } else if (embedding_state_->audio_features_) {
      // No audio: provide empty 2D tensor [0, hidden_size] for the embedding model
      embedding_state_->audio_features_->AllocateEmptyFeatures();
    }
    embedding_state_->inputs_embeds_.ReuseEmbeddingsBuffer(decoder_state_->inputs_embeds_);
    if (embedding_state_->per_layer_inputs_ && decoder_state_->per_layer_inputs_) {
      embedding_state_->per_layer_inputs_->ReuseEmbeddingsBuffer(*decoder_state_->per_layer_inputs_);
    }
    embedding_state_->Run(current_length, next_tokens, next_indices);

    auto logits = chunk_prefill
                      ? decoder_state_->RunPrefillWithChunking(current_length, next_tokens, next_indices, chunk_size_opt.value())
                      : decoder_state_->Run(current_length, next_tokens, next_indices);

    is_prompt_ = false;
    if (vision_state_) vision_state_.reset();  // The vision state is no longer needed in generation stage
    if (speech_state_) speech_state_.reset();  // The speech state is no longer needed in generation stage

    return audio_output_ ? SampleAudioOrText(logits) : logits;
  }

  embedding_state_->inputs_embeds_.ReuseEmbeddingsBuffer(decoder_state_->inputs_embeds_);
  if (embedding_state_->per_layer_inputs_ && decoder_state_->per_layer_inputs_) {
    embedding_state_->per_layer_inputs_->ReuseEmbeddingsBuffer(*decoder_state_->per_layer_inputs_);
  }
  if (audio_output_ && audio_output_->HasPendingFrame()) {
    // The token is only the placeholder of the last audio frame: the decoder takes the frame itself,
    // and the embedding model would look for audio features to put in the placeholder's place.
    audio_output_->WritePendingFrame(*decoder_state_->inputs_embeds_.Get());
  } else {
    embedding_state_->Run(current_length, next_tokens, next_indices);
  }
  auto logits = decoder_state_->Run(current_length, next_tokens, next_indices);
  return audio_output_ ? SampleAudioOrText(logits) : logits;
}

DeviceSpan<float> MultiModalPipelineState::SampleAudioOrText(DeviceSpan<float> logits) {
  if (!audio_output_->InAudio()) {
    return logits;
  }
  const std::string& configured = model_.config_->model.decoder.outputs.hidden_states;
  const std::string name = configured.empty() ? std::string(Config::Defaults::HiddenStatesName) : configured;
  OrtValue* hidden_states = decoder_state_->GetOutput(name.c_str());
  if (!hidden_states) {
    throw std::runtime_error("Speech output needs the decoder's \"" + name +
                             "\" output. Build the decoder with --extra_options include_hidden_states=true.");
  }
  return audio_output_->SampleFrame(*hidden_states);
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
  if (audio_output_ && std::strcmp(name, "audio_codes") == 0) {
    return audio_output_->GetAudioCodes();
  }

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
