// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <algorithm>
#include <cstring>

#include "../generators.h"
#include "nemotron_streaming_processor.h"

namespace Generators {

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
  auto& allocator = model_.allocator_cpu_;

  // Compute mel spectrogram for this chunk: returns [num_mels, num_frames] (frequency-major)
  auto [mel_data, num_frames] = mel_extractor_.Process(audio_chunk, chunk_samples);

  const int cache_size = nemotron_config_.pre_encode_cache_size;
  const int num_mels = nemotron_config_.num_mels;
  const int total_mel_frames = cache_size + num_frames;

  // Create output tensor: [1, total_mel_frames, num_mels] (time-major)
  auto signal_type = model_.session_info_.GetInputDataType(nemotron_config_.enc_in_audio);

  // TODO: Optimize for GPU/CUDA later, CPU always expects float32.
  if (signal_type != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
    throw std::runtime_error("NemotronStreamingProcessor only supports float32 encoder input. Got type: " + std::to_string(signal_type));
  }
  auto signal_shape = std::array<int64_t, 3>{1, total_mel_frames, num_mels};
  auto processed_signal = OrtValue::CreateTensor(allocator, signal_shape, signal_type);
  float* signal_data = processed_signal->GetTensorMutableData<float>();

  // Materialize cache frames from ring buffer (oldest-first starting at cache_pos_)
  // Use at most 2 memcpys instead of per-frame copies
  int first_run = std::min(cache_size - cache_pos_, cache_size);
  std::memcpy(signal_data,
              mel_pre_encode_cache_.data() + cache_pos_ * num_mels,
              first_run * num_mels * sizeof(float));
  if (first_run < cache_size) {
    std::memcpy(signal_data + first_run * num_mels,
                mel_pre_encode_cache_.data(),
                (cache_size - first_run) * num_mels * sizeof(float));
  }

  // Transpose mel from [num_mels, num_frames] directly into output tensor after cache
  float* out_ptr = signal_data + cache_size * num_mels;
  for (int t = 0; t < num_frames; ++t) {
    for (int m = 0; m < num_mels; ++m) {
      out_ptr[t * num_mels + m] = mel_data[m * num_frames + t];
    }
  }

  // Update ring buffer with the last cache_size frames (or all if fewer)
  int frames_to_cache = std::min(num_frames, cache_size);
  const float* cache_src = out_ptr + (num_frames - frames_to_cache) * num_mels;
  int frames_to_end = std::min(frames_to_cache, cache_size - cache_pos_);
  std::memcpy(mel_pre_encode_cache_.data() + cache_pos_ * num_mels,
              cache_src,
              frames_to_end * num_mels * sizeof(float));
  if (frames_to_end < frames_to_cache) {
    std::memcpy(mel_pre_encode_cache_.data(),
                cache_src + frames_to_end * num_mels,
                (frames_to_cache - frames_to_end) * num_mels * sizeof(float));
  }
  cache_pos_ = (cache_pos_ + frames_to_cache) % cache_size;

  return processed_signal;
}

std::unique_ptr<StreamingProcessor> CreateStreamingProcessor(Model& model) {
  return std::make_unique<NemotronStreamingProcessor>(model);
}

}  // namespace Generators
