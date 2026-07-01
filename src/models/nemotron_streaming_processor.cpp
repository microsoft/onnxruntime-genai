// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <algorithm>
#include <cstring>

#include "../generators.h"
#include "nemotron_streaming_processor.h"

namespace Generators {

template <typename T, typename Convert>
void PopulateMelTensorImpl(T* output, std::span<const float> cache,
                           int cache_pos, std::span<const float> mel,
                           int num_frames, int num_mels, Convert convert) {
  const int cache_frames = static_cast<int>(cache.size()) / num_mels;
  for (int frame = 0; frame < cache_frames; ++frame) {
    const int source_frame = (cache_pos + frame) % cache_frames;
    for (int mel_bin = 0; mel_bin < num_mels; ++mel_bin)
      output[frame * num_mels + mel_bin] = convert(cache[source_frame * num_mels + mel_bin]);
  }

  T* chunk_output = output + cache.size();
  for (int frame = 0; frame < num_frames; ++frame) {
    for (int mel_bin = 0; mel_bin < num_mels; ++mel_bin)
      chunk_output[frame * num_mels + mel_bin] = convert(mel[mel_bin * num_frames + frame]);
  }
}

void PopulateMelTensor(OrtValue& output, std::span<const float> cache,
                       int cache_pos, std::span<const float> mel,
                       int num_frames, int num_mels) {
  if (num_frames < 0 || num_mels <= 0 ||
      cache.size() % static_cast<size_t>(num_mels) != 0 ||
      mel.size() != static_cast<size_t>(num_frames * num_mels) ||
      output.GetTensorTypeAndShapeInfo()->GetElementCount() != cache.size() + mel.size())
    throw std::runtime_error("PopulateMelTensor: incompatible buffer dimensions");

  auto output_type = output.GetTensorTypeAndShapeInfo()->GetElementType();
  if (output_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
    auto* output_data = output.GetTensorMutableData<float>();
    const int cache_frames = static_cast<int>(cache.size()) / num_mels;
    const int first_run = cache_frames - cache_pos;
    std::memcpy(output_data,
                cache.data() + cache_pos * num_mels,
                static_cast<size_t>(first_run * num_mels) * sizeof(float));
    std::memcpy(output_data + first_run * num_mels,
                cache.data(),
                static_cast<size_t>((cache_frames - first_run) * num_mels) * sizeof(float));

    auto* chunk_output = output_data + cache.size();
    for (int frame = 0; frame < num_frames; ++frame) {
      for (int mel_bin = 0; mel_bin < num_mels; ++mel_bin)
        chunk_output[frame * num_mels + mel_bin] = mel[mel_bin * num_frames + frame];
    }
  } else if (output_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16) {
    PopulateMelTensorImpl(output.GetTensorMutableData<Ort::Float16_t>(), cache, cache_pos,
                          mel, num_frames, num_mels, [](float value) {
                            return Ort::Float16_t{FastFloat32ToFloat16(value)};
                          });
  } else {
    throw std::runtime_error("PopulateMelTensor: output must be float32 or float16");
  }
}

void UpdateMelCache(std::span<float> cache, int& cache_pos,
                    std::span<const float> mel, int num_frames, int num_mels) {
  const int cache_frames = static_cast<int>(cache.size()) / num_mels;
  const int frames_to_cache = std::min(num_frames, cache_frames);
  const int first_frame_to_cache = num_frames - frames_to_cache;
  for (int frame = 0; frame < frames_to_cache; ++frame) {
    const int destination_frame = (cache_pos + frame) % cache_frames;
    for (int mel_bin = 0; mel_bin < num_mels; ++mel_bin) {
      cache[destination_frame * num_mels + mel_bin] =
          mel[mel_bin * num_frames + first_frame_to_cache + frame];
    }
  }
  cache_pos = (cache_pos + frames_to_cache) % cache_frames;
}

NemotronStreamingProcessor::NemotronStreamingProcessor(Model& model)
    : model_{model} {
  auto* nemotron_model = dynamic_cast<NemotronSpeechModel*>(&model);
  if (!nemotron_model) {
    throw std::runtime_error("NemotronStreamingProcessor requires a nemotron_speech model type. Got: " + model.config_->model.type);
  }

  nemotron_config_ = nemotron_model->nemotron_config_;

  if (nemotron_config_.pre_encode_cache_size <= 0) {
    throw std::runtime_error("NemotronStreamingProcessor requires pre_encode_cache_size > 0. Got: " +
                             std::to_string(nemotron_config_.pre_encode_cache_size));
  }

  // Initialize mel extractor from config
  nemo_mel::NemoMelConfig mel_cfg{
      nemotron_config_.num_mels, nemotron_config_.fft_size,
      nemotron_config_.hop_length, nemotron_config_.win_length,
      nemotron_config_.sample_rate,
      nemotron_config_.preemph, nemotron_config_.log_eps};
  mel_extractor_ = nemo_mel::NemoStreamingMelExtractor{mel_cfg};

  // Initialize mel pre-encode cache (time-major ring buffer, zeros for first chunk)
  mel_pre_encode_cache_.assign(
      static_cast<size_t>(nemotron_config_.pre_encode_cache_size) * nemotron_config_.num_mels, 0.0f);
  cache_pos_ = 0;

  // Initialize VAD from config
  InitVadFromConfig(model);
}

NemotronStreamingProcessor::~NemotronStreamingProcessor() = default;

std::unique_ptr<NamedTensors> NemotronStreamingProcessor::Process(const float* audio_data, size_t num_samples) {
  // Append incoming audio to accumulation buffer
  audio_buffer_.insert(audio_buffer_.end(), audio_data, audio_data + num_samples);

  const size_t chunk_size = static_cast<size_t>(nemotron_config_.chunk_samples);

  // Process the first complete chunk available
  if (audio_buffer_.size() >= chunk_size) {
    const float* chunk_data = audio_buffer_.data();

    // VAD check: drop chunk if prolonged silence detected
    if (ShouldDropChunk(chunk_data, chunk_size)) {
      audio_buffer_.erase(audio_buffer_.begin(),
                          audio_buffer_.begin() + static_cast<ptrdiff_t>(chunk_size));
      return nullptr;
    }

    auto mel = BuildMelTensor(chunk_data, chunk_size);
    audio_buffer_.erase(audio_buffer_.begin(),
                        audio_buffer_.begin() + static_cast<ptrdiff_t>(chunk_size));
    auto result = std::make_unique<NamedTensors>();
    result->emplace(Config::Defaults::AudioFeaturesName, std::make_shared<Tensor>(std::move(mel)));
    return result;
  }

  return nullptr;  // Not enough audio yet
}

std::unique_ptr<NamedTensors> NemotronStreamingProcessor::Flush() {
  if (audio_buffer_.empty()) {
    return nullptr;
  }

  const size_t chunk_size = static_cast<size_t>(nemotron_config_.chunk_samples);
  audio_buffer_.resize(chunk_size, 0.0f);  // Pad with silence

  auto mel = BuildMelTensor(audio_buffer_.data(), chunk_size);
  audio_buffer_.clear();
  auto result = std::make_unique<NamedTensors>();
  result->emplace(Config::Defaults::AudioFeaturesName, std::make_shared<Tensor>(std::move(mel)));
  return result;
}

std::unique_ptr<OrtValue> NemotronStreamingProcessor::BuildMelTensor(const float* audio_chunk, size_t chunk_samples) {
  auto& allocator = GetDeviceInterface(DeviceType::CPU)->GetAllocator();

  // Compute mel spectrogram for this chunk: returns [num_mels, num_frames] (frequency-major)
  auto [mel_data, num_frames] = mel_extractor_.Process(audio_chunk, chunk_samples);

  const int cache_size = nemotron_config_.pre_encode_cache_size;
  const int num_mels = nemotron_config_.num_mels;
  const int total_mel_frames = cache_size + num_frames;

  // Create output tensor: [1, total_mel_frames, num_mels] (time-major)
  auto signal_type = model_.session_info_.GetInputDataType(nemotron_config_.enc_in_audio);
  if (signal_type != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT &&
      signal_type != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16) {
    throw std::runtime_error("NemotronStreamingProcessor only supports float32 or float16 encoder input. Got type: " + std::to_string(signal_type));
  }
  auto signal_shape = std::array<int64_t, 3>{1, total_mel_frames, num_mels};
  auto processed_signal = OrtValue::CreateTensor(allocator, signal_shape, signal_type);
  PopulateMelTensor(*processed_signal, mel_pre_encode_cache_, cache_pos_,
                    mel_data, num_frames, num_mels);

  UpdateMelCache(mel_pre_encode_cache_, cache_pos_, mel_data, num_frames, num_mels);

  return processed_signal;
}

std::unique_ptr<StreamingProcessor> CreateStreamingProcessor(Model& model) {
  return std::make_unique<NemotronStreamingProcessor>(model);
}

}  // namespace Generators
