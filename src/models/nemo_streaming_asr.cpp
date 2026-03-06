// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <algorithm>
#include <cmath>
#include <cstring>
#include <fstream>
#include <numeric>

#include "../generators.h"
#include "nemo_streaming_asr.h"

namespace Generators {

void NemoStreamingASR::LoadVocab() {
  if (vocab_loaded_) return;

  auto tokenizer = model_.CreateTokenizer();
  vocab_.resize(cache_config_.vocab_size);
  for (int i = 0; i < cache_config_.vocab_size; ++i) {
    try {
      std::vector<int32_t> ids = {static_cast<int32_t>(i)};
      vocab_[i] = tokenizer->Decode(ids);
    } catch (...) {
      vocab_[i] = "";
    }
  }

  // Pre-process sentencepiece space markers
  for (auto& tok : vocab_) {
    size_t pos = 0;
    while ((pos = tok.find("\xe2\x96\x81", pos)) != std::string::npos) {
      tok.replace(pos, 3, " ");
      pos += 1;
    }
  }
  vocab_loaded_ = true;
}

NemoStreamingASR::NemoStreamingASR(Model& model)
    : model_{model} {
  // Get the NemotronSpeechModel to access its sessions
  auto* nemotron_model = dynamic_cast<NemotronSpeechModel*>(&model);
  if (!nemotron_model) {
    throw std::runtime_error("NemoStreamingASR requires a nemotron_speech model type. Got: " + model.config_->model.type);
  }

  encoder_session_ = std::move(nemotron_model->session_encoder_);
  decoder_session_ = std::move(nemotron_model->session_decoder_);
  joiner_session_ = std::move(nemotron_model->session_joiner_);
  run_options_ = OrtRunOptions::Create();
  cache_config_ = nemotron_model->cache_config_;

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

  // Initialize streaming state
  auto& allocator = model_.allocator_cpu_;
  auto& device = *model_.p_device_;
  encoder_cache_.Initialize(cache_config_, allocator, device);
  decoder_state_.Initialize(cache_config_, allocator, device);
}

NemoStreamingASR::~NemoStreamingASR() = default;

void NemoStreamingASR::Reset() {
  auto& allocator = model_.allocator_cpu_;
  auto& device = *model_.p_device_;
  encoder_cache_.Reset(cache_config_, allocator, device);
  decoder_state_.Reset(cache_config_, allocator, device);
  full_transcript_.clear();
  mel_extractor_.Reset();
  mel_pre_encode_cache_.assign(
      static_cast<size_t>(cache_config_.pre_encode_cache_size) * cache_config_.num_mels, 0.0f);
  cache_pos_ = 0;
  audio_buffer_.clear();
}

std::string NemoStreamingASR::TranscribeChunk(const float* audio_data, size_t num_samples) {
  LoadVocab();

  // Append incoming audio to accumulation buffer
  audio_buffer_.insert(audio_buffer_.end(), audio_data, audio_data + num_samples);

  std::string result;
  const size_t chunk_size = static_cast<size_t>(cache_config_.chunk_samples);

  // Process chunks as soon as you get the full chunk size.
  size_t offset = 0;
  while (audio_buffer_.size() - offset >= chunk_size) {
    // Compute mel for this chunk
    auto [mel_data, num_frames] = mel_extractor_.Process(audio_buffer_.data() + offset, chunk_size);

    result += TranscribeMelChunk(mel_data, num_frames);

    // Advance by full chunk, Nemo models do not require overlapping audio
    offset += chunk_size;
  }

  if (offset > 0) {
    audio_buffer_.erase(audio_buffer_.begin(),
                        audio_buffer_.begin() + static_cast<ptrdiff_t>(offset));
  }

  return result;
}

std::string NemoStreamingASR::Flush() {
  LoadVocab();

  std::string result;
  const size_t chunk_size = static_cast<size_t>(cache_config_.chunk_samples);

  // Process any remaining audio (pad to full chunk with silence)
  if (!audio_buffer_.empty()) {
    audio_buffer_.resize(chunk_size, 0.0f);

    auto [mel_data, num_frames] = mel_extractor_.Process(audio_buffer_.data(), chunk_size);
    result += TranscribeMelChunk(mel_data, num_frames);

    audio_buffer_.clear();
  }

  return result;
}

std::string NemoStreamingASR::TranscribeMelChunk(const std::vector<float>& mel_data, int num_frames) {
  auto& allocator = model_.allocator_cpu_;
  int cache_size = cache_config_.pre_encode_cache_size;
  const int num_mels = cache_config_.num_mels;

  int total_mel_frames = cache_size + num_frames;

  std::vector<float> mel_time_major(static_cast<size_t>(num_frames) * num_mels);
  for (int t = 0; t < num_frames; ++t) {
    for (int m = 0; m < num_mels; ++m) {
      mel_time_major[t * num_mels + m] = mel_data[m * num_frames + t];
    }
  }

  // Create processed_signal: [1, total_mel_frames, num_mels] (time-major layout)
  auto signal_type = model_.session_info_.GetInputDataType(cache_config_.enc_in_audio);
  auto signal_shape = std::array<int64_t, 3>{1, total_mel_frames, num_mels};
  auto processed_signal = OrtValue::CreateTensor(allocator, signal_shape, signal_type);
  float* signal_data = processed_signal->GetTensorMutableData<float>();

  // Materialize cache frames into processed_signal [total_mel_frames, num_mels].
  // Ring buffer is time-major [cache_size, num_mels]; read oldest-first starting at cache_pos_.
  for (int t = 0; t < cache_size; ++t) {
    int ring_idx = (cache_pos_ + t) % cache_size;
    const float* src = mel_pre_encode_cache_.data() + ring_idx * num_mels;
    std::memcpy(signal_data + t * num_mels, src, num_mels * sizeof(float));
  }

  std::memcpy(signal_data + cache_size * num_mels,
              mel_time_major.data(),
              static_cast<size_t>(num_frames) * num_mels * sizeof(float));

  // Update ring buffer with time-major data (contiguous memcpy per frame)
  for (int t = 0; t < num_frames; ++t) {
    float* destination = mel_pre_encode_cache_.data() + cache_pos_ * num_mels;
    std::memcpy(destination, mel_time_major.data() + t * num_mels, num_mels * sizeof(float));
    cache_pos_ = (cache_pos_ + 1) % cache_size;
  }

  // Create processed_signal_length: [1]
  auto len_shape = std::array<int64_t, 1>{1};
  auto len_type = model_.session_info_.GetInputDataType(cache_config_.enc_in_length);
  auto signal_length = OrtValue::CreateTensor(allocator, len_shape, len_type);
  *signal_length->GetTensorMutableData<int64_t>() = static_cast<int64_t>(total_mel_frames);

  // Encoder inputs
  const char* enc_input_names[] = {
      cache_config_.enc_in_audio.c_str(), cache_config_.enc_in_length.c_str(),
      cache_config_.enc_in_cache_channel.c_str(), cache_config_.enc_in_cache_time.c_str(),
      cache_config_.enc_in_cache_channel_len.c_str()};
  OrtValue* enc_inputs[] = {
      processed_signal.get(), signal_length.get(),
      encoder_cache_.cache_last_channel.get(),
      encoder_cache_.cache_last_time.get(),
      encoder_cache_.cache_last_channel_len.get()};

  const char* enc_output_names[] = {
      cache_config_.enc_out_encoded.c_str(), cache_config_.enc_out_length.c_str(),
      cache_config_.enc_out_cache_channel.c_str(), cache_config_.enc_out_cache_time.c_str(),
      cache_config_.enc_out_cache_channel_len.c_str()};

  // Run encoder
  auto enc_outputs = encoder_session_->Run(
      run_options_.get(),
      enc_input_names, enc_inputs, 5,
      enc_output_names, 5);

  // Parse encoder outputs
  auto* encoded = enc_outputs[0].get();
  int64_t encoded_len = *enc_outputs[1]->GetTensorData<int64_t>();

  // Update cache
  encoder_cache_.cache_last_channel = std::move(enc_outputs[2]);
  encoder_cache_.cache_last_time = std::move(enc_outputs[3]);
  encoder_cache_.cache_last_channel_len = std::move(enc_outputs[4]);

  // Run RNNT decoder on ALL encoder output frames (no drop_last needed)
  std::string chunk_text = RunRNNTDecoder(encoded, encoded_len);
  full_transcript_ += chunk_text;

  return chunk_text;
}

std::string NemoStreamingASR::RunRNNTDecoder(OrtValue* encoder_output, int64_t encoded_len) {
  auto& allocator = model_.allocator_cpu_;
  std::string result;

  auto enc_info = encoder_output->GetTensorTypeAndShapeInfo();
  auto enc_shape = enc_info->GetShape();
  // Encoder output layout: [batch, time, hidden_dim]
  int64_t time_steps = std::min(enc_shape[1], encoded_len);
  int64_t hidden_dim = enc_shape[2];

  auto run_options = OrtRunOptions::Create();

  // Pre-allocate reusable tensors
  auto enc_out_type = model_.session_info_.GetOutputDataType(cache_config_.enc_out_encoded);
  auto frame_shape = std::array<int64_t, 3>{1, 1, hidden_dim};
  auto encoder_frame = OrtValue::CreateTensor(allocator, frame_shape, enc_out_type);

  // Wrap full encoder output and frame tensor as DeviceSpans for efficient copy
  auto enc_span = ByteWrapTensor(*model_.p_device_, *encoder_output);
  auto frame_span = ByteWrapTensor(*model_.p_device_, *encoder_frame);
  const size_t frame_bytes = static_cast<size_t>(hidden_dim) * sizeof(float);

  auto targets_shape = std::array<int64_t, 2>{1, 1};
  auto targets_type = model_.session_info_.GetInputDataType(cache_config_.dec_in_targets);
  auto targets = OrtValue::CreateTensor(allocator, targets_shape, targets_type);
  int64_t* targets_data = targets->GetTensorMutableData<int64_t>();

  auto tgt_len_shape = std::array<int64_t, 1>{1};
  auto tgt_len_type = model_.session_info_.GetInputDataType(cache_config_.dec_in_target_length);
  auto target_length = OrtValue::CreateTensor(allocator, tgt_len_shape, tgt_len_type);
  *target_length->GetTensorMutableData<int64_t>() = 1;

  const int max_sym = cache_config_.max_symbols_per_step;

  for (int64_t t = 0; t < time_steps; ++t) {
    auto src_frame = enc_span.subspan(static_cast<size_t>(t) * frame_bytes, frame_bytes);
    frame_span.CopyFrom(src_frame);

    for (int sym = 0; sym < max_sym; ++sym) {
      *targets_data = decoder_state_.last_token;

      const char* dec_input_names[] = {
          cache_config_.dec_in_targets.c_str(), cache_config_.dec_in_target_length.c_str(),
          cache_config_.dec_in_lstm_hidden.c_str(), cache_config_.dec_in_lstm_cell.c_str()};
      OrtValue* dec_inputs[] = {
          targets.get(), target_length.get(),
          decoder_state_.lstm_hidden_state.get(), decoder_state_.lstm_cell_state.get()};

      const char* dec_output_names[] = {
          cache_config_.dec_out_outputs.c_str(), cache_config_.dec_out_prednet_lengths.c_str(),
          cache_config_.dec_out_lstm_hidden.c_str(), cache_config_.dec_out_lstm_cell.c_str()};

      auto dec_outputs = decoder_session_->Run(
          run_options_.get(),
          dec_input_names, dec_inputs, 4,
          dec_output_names, 4);

      auto dec_out_shape = dec_outputs[0]->GetTensorTypeAndShapeInfo()->GetShape();
      auto decoder_frame_shape = std::array<int64_t, 3>{1, 1, dec_out_shape[1]};
      auto dec_out_type = model_.session_info_.GetOutputDataType(cache_config_.dec_out_outputs);
      auto decoder_frame = OrtValue::CreateTensor(allocator, decoder_frame_shape, dec_out_type);
      auto source_span = ByteWrapTensor(*model_.p_device_, *dec_outputs[0]);
      auto destination_span = ByteWrapTensor(*model_.p_device_, *decoder_frame);
      destination_span.CopyFrom(source_span);

      // Run joiner
      const char* join_input_names[] = {
          cache_config_.join_in_encoder.c_str(), cache_config_.join_in_decoder.c_str()};
      OrtValue* join_inputs[] = {
          encoder_frame.get(), decoder_frame.get()};

      const char* join_output_names[] = {cache_config_.join_out_logits.c_str()};

      auto join_outputs = joiner_session_->Run(
          run_options_.get(),
          join_input_names, join_inputs, 2,
          join_output_names, 1);

      // Find argmax
      const float* logits_data = join_outputs[0]->GetTensorData<float>();  // Logits are always float for argmax
      auto logits_shape = join_outputs[0]->GetTensorTypeAndShapeInfo()->GetShape();
      int total_logits = 1;
      for (auto d : logits_shape) total_logits *= static_cast<int>(d);

      int best_token = 0;
      float best_score = logits_data[0];
      for (int i = 1; i < total_logits; ++i) {
        if (logits_data[i] > best_score) {
          best_score = logits_data[i];
          best_token = i;
        }
      }

      // Blank means current frame is done, move to the next frame.
      if (best_token == cache_config_.blank_id || best_token >= cache_config_.vocab_size) {
        break;
      }

      // Emit token & update state
      decoder_state_.last_token = best_token;
      decoder_state_.lstm_hidden_state = std::move(dec_outputs[2]);
      decoder_state_.lstm_cell_state = std::move(dec_outputs[3]);

      result += vocab_[best_token];
    }
  }

  return result;
}

std::unique_ptr<StreamingASR> CreateStreamingASR(Model& model) {
  const auto& model_type = model.config_->model.type;
  if (model_type == "nemotron_speech") {
    return std::make_unique<NemoStreamingASR>(model);
  }
  throw std::runtime_error("Unsupported model type for StreamingASR: " + model_type);
}

}  // namespace Generators
