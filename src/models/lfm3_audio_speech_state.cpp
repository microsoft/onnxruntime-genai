// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "generator/generators.h"
#include "lfm3_audio_speech_state.h"
#include "models/preprocessing/lfm2_audio_processor.h"

#include <algorithm>
#include <cstring>

namespace Generators {

namespace {

// Holds the vector and an index, not an element reference: the first State::Run can grow outputs_ for extra outputs.
struct OrtValuePointerRestore {
  std::vector<OrtValue*>& slots;
  size_t index;
  OrtValue* value;

  ~OrtValuePointerRestore() { slots[index] = value; }
};

std::vector<int64_t> ReadCounts(const ExtraInput& input) {
  const auto info = input.tensor->ort_tensor_->GetTensorTypeAndShapeInfo();
  if (info->GetElementType() != ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64) {
    throw std::runtime_error("lfm3_audio: " + input.name + " must be int64.");
  }
  const int64_t* counts = input.tensor->ort_tensor_->GetTensorData<int64_t>();
  return {counts, counts + info->GetElementCount()};
}

}  // namespace

Lfm3AudioSpeechState::Lfm3AudioSpeechState(const MultiModalLanguageModel& model, const GeneratorParams& params)
    : SpeechState{model, params} {
  // The processor fills one sequence and nothing pads or repeats audio features across rows or beams.
  if (params.search.batch_size > 1 || params.search.num_beams > 1) {
    throw std::runtime_error("lfm3_audio supports batch_size 1 and num_beams 1 only; got batch_size " +
                             std::to_string(params.search.batch_size) + " and num_beams " +
                             std::to_string(params.search.num_beams) + ".");
  }
}

void Lfm3AudioSpeechState::SetExtraInputs(const std::vector<ExtraInput>& extra_inputs, const int64_t num_audio_tokens) {
  const auto& names = model_.config_->model.speech.inputs;
  // The checks below see the last entry of a name, but Run and num_audio_tokens use the first.
  for (const std::string& name : {names.audio_embeds, names.audio_lengths, names.audio_sizes}) {
    const auto is_name = [&name](const ExtraInput& input) { return input.name == name; };
    if (std::count_if(extra_inputs.begin(), extra_inputs.end(), is_name) > 1) {
      throw std::runtime_error("lfm3_audio: got " + name +
                               " more than once. A failed set_inputs keeps its inputs, so retry with a new generator.");
    }
  }
  frames_per_clip_.clear();
  tokens_per_clip_.clear();
  std::vector<int64_t> mel_shape;
  for (const auto& input : extra_inputs) {
    if (input.name == names.audio_lengths) {
      frames_per_clip_ = ReadCounts(input);
    } else if (input.name == names.audio_sizes) {
      tokens_per_clip_ = ReadCounts(input);
    } else if (input.name == names.audio_embeds) {
      mel_shape = input.tensor->ort_tensor_->GetTensorTypeAndShapeInfo()->GetShape();
    }
  }

  if (!mel_shape.empty() || num_audio_tokens > 0) {
    if (mel_shape.size() != 3 || mel_shape[0] < 1) {
      throw std::runtime_error("lfm3_audio: " + names.audio_embeds +
                               " must be [num_clips, num_mels, frames] with at least one clip, got rank " +
                               std::to_string(mel_shape.size()) +
                               (mel_shape.size() != 3 ? "" : " and " + std::to_string(mel_shape[0]) + " clips") + ".");
    }
    const auto num_clips = static_cast<size_t>(mel_shape[0]);
    if (frames_per_clip_.size() != num_clips || tokens_per_clip_.size() != num_clips) {
      throw std::runtime_error("lfm3_audio: " + names.audio_embeds + " holds " + std::to_string(num_clips) + " clips, but " +
                               names.audio_lengths + " has " + std::to_string(frames_per_clip_.size()) + " entries and " +
                               names.audio_sizes + " has " + std::to_string(tokens_per_clip_.size()) + ".");
    }
    const int64_t subsampling_factor = model_.config_->model.subsampling_factor;
    for (size_t clip = 0; clip < num_clips; ++clip) {
      const int64_t frames = frames_per_clip_[clip];
      if (frames < 1 || frames > mel_shape[2] || tokens_per_clip_[clip] != Lfm2AudioNumTokens(frames, subsampling_factor, "lfm3_audio")) {
        throw std::runtime_error("lfm3_audio: clip " + std::to_string(clip) + " reports " + std::to_string(frames) +
                                 " frames and " + std::to_string(tokens_per_clip_[clip]) + " tokens, against " +
                                 std::to_string(mel_shape[2]) + " frames in " + names.audio_embeds +
                                 " and one token per " + std::to_string(subsampling_factor) + " frames.");
      }
    }
    if (*std::max_element(frames_per_clip_.begin(), frames_per_clip_.end()) != mel_shape[2]) {
      throw std::runtime_error("lfm3_audio: the longest clip must fill " + names.audio_embeds + "'s " +
                               std::to_string(mel_shape[2]) + " frames.");
    }
  }

  SpeechState::SetExtraInputs(extra_inputs, num_audio_tokens);
}

DeviceSpan<float> Lfm3AudioSpeechState::Run(int current_length, DeviceSpan<int32_t>& next_tokens,
                                            DeviceSpan<int32_t> next_indices) {
  if (frames_per_clip_.size() <= 1) {
    return SpeechState::Run(current_length, next_tokens, next_indices);
  }
  if (model_.config_->model.speech.run_options.has_value()) {
    State::SetRunOptions(model_.config_->model.speech.run_options.value());
  }

  const auto& mel_name = model_.config_->model.speech.inputs.audio_embeds;
  const auto mel_input = std::find(input_names_.begin(), input_names_.end(), mel_name);
  if (mel_input == input_names_.end()) {
    throw std::runtime_error("lfm3_audio: the speech model has no bound input named \"" + mel_name + "\".");
  }
  const auto mel_index = static_cast<size_t>(mel_input - input_names_.begin());
  const auto& features_name = model_.config_->model.speech.outputs.audio_features;
  const auto features_output = std::find(output_names_.begin(), output_names_.end(), features_name);
  if (features_output == output_names_.end()) {
    throw std::runtime_error("lfm3_audio: the speech model has no bound output named \"" + features_name + "\".");
  }
  const auto features_index = static_cast<size_t>(features_output - output_names_.begin());
  OrtValue* padded_mel = inputs_[mel_index];
  OrtValue* features = outputs_[features_index];
  // Each run binds one clip's tensors, which die with their loop iteration.
  const OrtValuePointerRestore restore_mel{inputs_, mel_index, padded_mel};
  const OrtValuePointerRestore restore_features{outputs_, features_index, features};

  const auto mel_info = padded_mel->GetTensorTypeAndShapeInfo();
  const auto mel_type = mel_info->GetElementType();
  const auto mel_shape = mel_info->GetShape();
  const int64_t num_mels = mel_shape[1];
  const int64_t longest = mel_shape[2];
  const size_t element_bytes = Ort::SizeOf(mel_type);
  const auto feature_info = features->GetTensorTypeAndShapeInfo();
  const auto feature_type = feature_info->GetElementType();
  const int64_t hidden_size = feature_info->GetShape().back();
  const size_t token_bytes = static_cast<size_t>(hidden_size) * Ort::SizeOf(feature_type);

  // The padded mel is the caller's CPU tensor; the features live on the speech session's device.
  const auto* padded_data = static_cast<const uint8_t*>(padded_mel->GetTensorRawData());
  auto feature_bytes = ByteWrapTensor(*p_session_device_, *features);
  size_t feature_offset = 0;
  for (size_t clip = 0; clip < frames_per_clip_.size(); ++clip) {
    const int64_t frames = frames_per_clip_[clip];
    const int64_t tokens = tokens_per_clip_[clip];
    auto clip_mel = OrtValue::CreateTensor(Ort::Allocator::GetWithDefaultOptions(),
                                           std::vector<int64_t>{1, num_mels, frames}, mel_type);
    auto* clip_data = static_cast<uint8_t*>(clip_mel->GetTensorMutableRawData());
    const uint8_t* source = padded_data + clip * static_cast<size_t>(num_mels * longest) * element_bytes;
    for (size_t m = 0; m < static_cast<size_t>(num_mels); ++m) {
      std::memcpy(clip_data + m * static_cast<size_t>(frames) * element_bytes,
                  source + m * static_cast<size_t>(longest) * element_bytes,
                  static_cast<size_t>(frames) * element_bytes);
    }
    auto clip_features = OrtValue::CreateTensor(p_session_device_->GetAllocator(),
                                                std::vector<int64_t>{tokens, hidden_size}, feature_type);

    inputs_[mel_index] = clip_mel.get();
    outputs_[features_index] = clip_features.get();
    State::Run(*model_.speech_session_);

    const size_t clip_bytes = static_cast<size_t>(tokens) * token_bytes;
    feature_bytes.subspan(feature_offset, clip_bytes).CopyFrom(ByteWrapTensor(*p_session_device_, *clip_features));
    feature_offset += clip_bytes;
  }
  return {};
}

}  // namespace Generators
