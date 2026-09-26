// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "generator/generators.h"
#include "models/speech/lfm2_audio_speech_state.h"
#include "models/multi_modal.h"

#include <algorithm>
#include <cstring>

namespace Generators {

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

void Lfm2AudioSpeechState::SetExtraInputs(const std::vector<ExtraInput>& extra_inputs, const int64_t num_audio_tokens) {
  // The feature buffer is one sequence wide, so beams would fail as a shape mismatch inside the
  // encoder. The reference decodes greedily.
  if (params_->search.num_beams > 1) {
    throw std::runtime_error("Lfm2AudioSpeechState: beam search is not supported for lfm2_audio; got num_beams " +
                             std::to_string(params_->search.num_beams) + ". Set num_beams to 1.");
  }

  num_audio_tokens_ = num_audio_tokens;
  audio_features_ = std::make_unique<MultiModalFeatures>(*this, MultiModalFeatures::Mode::Output,
                                                         model_.config_->model.speech.outputs.audio_features,
                                                         params_->BatchBeamSize(), num_audio_tokens_);
  audio_features_->Add();
  extra_inputs_.Add(extra_inputs, model_.speech_session_->GetInputNames());

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

void Lfm2AudioSpeechState::ReuseFeaturesBuffer(MultiModalFeatures& embedding_features) {
  auto& speech_shape = audio_features_->GetShape();
  if (speech_shape.size() == 3) {
    audio_features_->ReshapeFeatures({speech_shape[0] * speech_shape[1], speech_shape[2]});
  }
  SpeechState::ReuseFeaturesBuffer(embedding_features);
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

}  // namespace Generators
