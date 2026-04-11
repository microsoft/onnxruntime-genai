// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <algorithm>
#include <cstring>
#include <fstream>

#include "../generators.h"
#include "nemotron_streaming_processor.h"

namespace Generators {

NemotronStreamingProcessor::NemotronStreamingProcessor(Model& model)
    : model_{model} {
  auto* nemotron_model = dynamic_cast<NemotronSpeechModel*>(&model);
  if (!nemotron_model) {
    throw std::runtime_error("NemotronStreamingProcessor requires a nemotron_speech model type. Got: " + model.config_->model.type);
  }

  cache_config_ = nemotron_model->cache_config_;

  if (cache_config_.pre_encode_cache_size <= 0) {
    throw std::runtime_error("NemotronStreamingProcessor requires pre_encode_cache_size > 0. Got: " +
                             std::to_string(cache_config_.pre_encode_cache_size));
  }

  // Compute first chunk sample count to match NeMo's shorter first chunk
  {
    const int sub = cache_config_.subsampling_factor;
    const int chunk_enc_frames = cache_config_.chunk_samples / cache_config_.hop_length / sub;
    const int first_chunk_mel = (chunk_enc_frames - 1) * sub + 1;
    first_chunk_samples_ = first_chunk_mel * cache_config_.hop_length;
  }

  // Initialize mel extractor from config
  nemo_mel::NemoMelConfig mel_cfg{
      cache_config_.num_mels, cache_config_.fft_size,
      cache_config_.hop_length, cache_config_.win_length,
      cache_config_.sample_rate,
      cache_config_.preemph, cache_config_.log_eps};
  mel_extractor_ = nemo_mel::NemoStreamingMelExtractor{mel_cfg};

  // Initialize mel pre-encode cache (time-major ring buffer, zeros for first chunk)
  mel_pre_encode_cache_.assign(
      static_cast<size_t>(cache_config_.pre_encode_cache_size) * cache_config_.num_mels, 0.0f);
  cache_pos_ = 0;

  // Initialize VAD from config
  InitVadFromConfig(model);
}

NemotronStreamingProcessor::~NemotronStreamingProcessor() = default;

std::unique_ptr<NamedTensors> NemotronStreamingProcessor::Process(const float* audio_data, size_t num_samples) {
  // Append incoming audio to accumulation buffer
  audio_buffer_.insert(audio_buffer_.end(), audio_data, audio_data + num_samples);

  const size_t chunk_size = static_cast<size_t>(cache_config_.chunk_samples);

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

  const size_t chunk_size = static_cast<size_t>(cache_config_.chunk_samples);
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
  auto [mel_data, num_frames_raw] = mel_extractor_.Process(audio_chunk, chunk_samples);

  const int cache_size = cache_config_.pre_encode_cache_size;
  const int num_mels = cache_config_.num_mels;
  const int sub = cache_config_.subsampling_factor;
  // Derive encoder chunk frames from audio config: chunk_samples / hop_length / subsampling_factor
  const int chunk_enc_frames = cache_config_.chunk_samples / cache_config_.hop_length / sub;

  // NeMo streaming_cfg.chunk_size = [(chunk_frames-1)*sub+1, chunk_frames*sub]
  // First chunk: only (chunk_frames-1)*sub+1 real mel frames, rest zero-padded
  // Subsequent chunks: full chunk_frames*sub mel frames + pre-encode cache
  const int first_chunk_mel = (chunk_enc_frames - 1) * sub + 1;
  const int rest_chunk_mel = chunk_enc_frames * sub;

  int num_frames;
  int prepend_cache;
  int zero_pad_front;
  if (is_first_chunk_) {
    num_frames = std::min(num_frames_raw, first_chunk_mel);
    prepend_cache = 0;
    zero_pad_front = 0;
  } else {
    num_frames = std::min(num_frames_raw, rest_chunk_mel);
    prepend_cache = cache_size;
    zero_pad_front = 0;
  }
  const int total_mel_frames = zero_pad_front + prepend_cache + num_frames;

  // Create output tensor: [1, total_mel_frames, num_mels] (time-major)
  auto signal_type = model_.session_info_.GetInputDataType(cache_config_.enc_in_audio);

  // TODO: Optimize for GPU/CUDA later, CPU always expects float32.
  if (signal_type != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
    throw std::runtime_error("NemotronStreamingProcessor only supports float32 encoder input. Got type: " + std::to_string(signal_type));
  }
  auto signal_shape = std::array<int64_t, 3>{1, total_mel_frames, num_mels};
  auto processed_signal = OrtValue::CreateTensor(allocator, signal_shape, signal_type);
  float* signal_data = processed_signal->GetTensorMutableData<float>();

  // Pad front with log(eps) for the first chunk to match NeMo's native [49] behavior.
  // NeMo feeds only 49 mel frames with no cache on the first chunk, but the ONNX encoder
  // requires a fixed 65-frame input. Padding with log(eps) (silence-level mel) produces
  // identical conv subsampling output to NeMo's unpadded path (empirically verified).
  // Using 0.0 would represent max energy and corrupts the conv subsampling boundary frames.
  if (zero_pad_front > 0) {
    const float pad_val = std::log(cache_config_.log_eps);
    std::fill(signal_data, signal_data + zero_pad_front * num_mels, pad_val);
  }

  if (!is_first_chunk_) {
    // Materialize cache frames from ring buffer (oldest-first starting at cache_pos_)
    int first_run = std::min(cache_size - cache_pos_, cache_size);
    std::memcpy(signal_data,
                mel_pre_encode_cache_.data() + cache_pos_ * num_mels,
                first_run * num_mels * sizeof(float));
    if (first_run < cache_size) {
      std::memcpy(signal_data + first_run * num_mels,
                  mel_pre_encode_cache_.data(),
                  (cache_size - first_run) * num_mels * sizeof(float));
    }
  }

  // Transpose mel from [num_mels, num_frames_raw] directly into output tensor after front padding + cache
  // Note: mel_data stride is num_frames_raw even if we only use num_frames
  float* out_ptr = signal_data + (zero_pad_front + prepend_cache) * num_mels;
  for (int t = 0; t < num_frames; ++t) {
    for (int m = 0; m < num_mels; ++m) {
      out_ptr[t * num_mels + m] = mel_data[m * num_frames_raw + t];
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

  is_first_chunk_ = false;

  return processed_signal;
}

std::unique_ptr<StreamingProcessor> CreateStreamingProcessor(Model& model) {
  return std::make_unique<NemotronStreamingProcessor>(model);
}

}  // namespace Generators
