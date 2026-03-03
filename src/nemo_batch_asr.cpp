// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <algorithm>
#include <cmath>
#include <cstring>
#include <numeric>
#include <vector>

#include "generators.h"
#include "nemo_batch_asr.h"

namespace Generators {

void NemoBatchASR::LoadVocab() {
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

NemoBatchASR::NemoBatchASR(Model& model)
    : model_{model} {
  auto* nemotron_model = dynamic_cast<NemotronSpeechModel*>(&model);
  if (!nemotron_model) {
    throw std::runtime_error("NemoBatchASR requires a nemotron_speech model type. Got: " + model.config_->model.type);
  }

  encoder_session_ = nemotron_model->session_encoder_.get();
  decoder_session_ = nemotron_model->session_decoder_.get();
  joiner_session_ = nemotron_model->session_joiner_.get();
  cache_config_ = nemotron_model->cache_config_;
}

NemoBatchASR::~NemoBatchASR() = default;

std::string NemoBatchASR::Transcribe(const float* audio_data, size_t num_samples) {
  LoadVocab();

  auto& allocator = model_.allocator_cpu_;

  // 1. Compute mel spectrogram on the FULL audio at once (batch mode).
  //    This gives the cleanest spectrogram — no chunk-boundary artifacts.
  nemo_mel::NemoMelConfig mel_cfg{
      cache_config_.num_mels, cache_config_.fft_size,
      cache_config_.hop_length, cache_config_.win_length,
      cache_config_.sample_rate,
      cache_config_.preemph, cache_config_.log_eps};

  int total_mel_frames = 0;
  std::vector<float> full_mel = nemo_mel::NemoComputeLogMelBatch(
      audio_data, num_samples, mel_cfg, total_mel_frames);
  // full_mel layout: row-major [num_mels, total_mel_frames]

  const int num_mels = cache_config_.num_mels;
  const int cache_size = cache_config_.pre_encode_cache_size;

  // 2. The ONNX encoder was exported from NeMo with streaming_post_process(),
  //    which slices output to valid_out_len frames per call. So we MUST chunk
  //    the mel using the same chunk size the model was exported with.
  //    The batch advantage is that mel was computed on the full audio (step 1),
  //    avoiding chunk-boundary artifacts in the spectrogram.
  const int frames_per_chunk = cache_config_.chunk_samples / cache_config_.hop_length;

  NemotronEncoderCache encoder_cache;
  NemotronDecoderState decoder_state;
  encoder_cache.Initialize(cache_config_, allocator);
  decoder_state.Initialize(cache_config_, allocator);

  // Pre-encode cache: last cache_size mel frames from previous chunk
  std::vector<float> mel_pre_encode_cache(
      static_cast<size_t>(num_mels) * cache_size, 0.0f);

  std::string transcript;
  auto run_options = OrtRunOptions::Create();

  int mel_offset = 0;
  while (mel_offset < total_mel_frames) {
    int chunk_frames = std::min(frames_per_chunk, total_mel_frames - mel_offset);
    int encoder_frames = cache_size + chunk_frames;

    // Build encoder input: [1, num_mels, cache_size + chunk_frames]
    auto signal_shape = std::array<int64_t, 3>{1, num_mels, static_cast<int64_t>(encoder_frames)};
    auto processed_signal = OrtValue::CreateTensor(allocator, signal_shape, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT);
    float* signal_data = processed_signal->GetTensorMutableData<float>();

    for (int m = 0; m < num_mels; ++m) {
      // Pre-encode cache for this mel bin
      std::memcpy(signal_data + m * encoder_frames,
                  mel_pre_encode_cache.data() + m * cache_size,
                  cache_size * sizeof(float));
      // Current chunk's mel for this mel bin
      std::memcpy(signal_data + m * encoder_frames + cache_size,
                  full_mel.data() + m * total_mel_frames + mel_offset,
                  chunk_frames * sizeof(float));
    }

    // Update pre-encode cache for next iteration
    if (chunk_frames >= cache_size) {
      for (int m = 0; m < num_mels; ++m) {
        std::memcpy(mel_pre_encode_cache.data() + m * cache_size,
                    full_mel.data() + m * total_mel_frames + mel_offset + (chunk_frames - cache_size),
                    cache_size * sizeof(float));
      }
    } else {
      int keep = cache_size - chunk_frames;
      for (int m = 0; m < num_mels; ++m) {
        std::memmove(mel_pre_encode_cache.data() + m * cache_size,
                     mel_pre_encode_cache.data() + m * cache_size + chunk_frames,
                     keep * sizeof(float));
        std::memcpy(mel_pre_encode_cache.data() + m * cache_size + keep,
                    full_mel.data() + m * total_mel_frames + mel_offset,
                    chunk_frames * sizeof(float));
      }
    }

    auto len_shape = std::array<int64_t, 1>{1};
    auto signal_length = OrtValue::CreateTensor(allocator, len_shape, ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64);
    *signal_length->GetTensorMutableData<int64_t>() = static_cast<int64_t>(encoder_frames);

    const char* enc_input_names[] = {
        cache_config_.enc_in_audio.c_str(), cache_config_.enc_in_length.c_str(),
        cache_config_.enc_in_cache_channel.c_str(), cache_config_.enc_in_cache_time.c_str(),
        cache_config_.enc_in_cache_channel_len.c_str()};
    OrtValue* enc_inputs[] = {
        processed_signal.get(), signal_length.get(),
        encoder_cache.cache_last_channel.get(),
        encoder_cache.cache_last_time.get(),
        encoder_cache.cache_last_channel_len.get()};

    const char* enc_output_names[] = {
        cache_config_.enc_out_encoded.c_str(), cache_config_.enc_out_length.c_str(),
        cache_config_.enc_out_cache_channel.c_str(), cache_config_.enc_out_cache_time.c_str(),
        cache_config_.enc_out_cache_channel_len.c_str()};

    auto enc_outputs = encoder_session_->Run(
        run_options.get(),
        enc_input_names, enc_inputs, 5,
        enc_output_names, 5);

    auto* encoded = enc_outputs[0].get();
    int64_t encoded_len = *enc_outputs[1]->GetTensorData<int64_t>();

    encoder_cache.cache_last_channel = std::move(enc_outputs[2]);
    encoder_cache.cache_last_time = std::move(enc_outputs[3]);
    encoder_cache.cache_last_channel_len = std::move(enc_outputs[4]);

    transcript += RunRNNTDecoder(encoded, encoded_len, decoder_state);
    mel_offset += chunk_frames;
  }

  // Flush: send silence chunks to drain remaining encoder context
  for (int i = 0; i < 4; ++i) {
    int encoder_frames = cache_size + frames_per_chunk;
    auto signal_shape = std::array<int64_t, 3>{1, num_mels, static_cast<int64_t>(encoder_frames)};
    auto processed_signal = OrtValue::CreateTensor(allocator, signal_shape, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT);
    std::memset(processed_signal->GetTensorMutableData<float>(), 0,
                static_cast<size_t>(num_mels) * encoder_frames * sizeof(float));

    auto len_shape = std::array<int64_t, 1>{1};
    auto signal_length = OrtValue::CreateTensor(allocator, len_shape, ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64);
    *signal_length->GetTensorMutableData<int64_t>() = static_cast<int64_t>(encoder_frames);

    const char* enc_input_names[] = {
        cache_config_.enc_in_audio.c_str(), cache_config_.enc_in_length.c_str(),
        cache_config_.enc_in_cache_channel.c_str(), cache_config_.enc_in_cache_time.c_str(),
        cache_config_.enc_in_cache_channel_len.c_str()};
    OrtValue* enc_inputs[] = {
        processed_signal.get(), signal_length.get(),
        encoder_cache.cache_last_channel.get(),
        encoder_cache.cache_last_time.get(),
        encoder_cache.cache_last_channel_len.get()};

    const char* enc_output_names[] = {
        cache_config_.enc_out_encoded.c_str(), cache_config_.enc_out_length.c_str(),
        cache_config_.enc_out_cache_channel.c_str(), cache_config_.enc_out_cache_time.c_str(),
        cache_config_.enc_out_cache_channel_len.c_str()};

    auto enc_outputs = encoder_session_->Run(
        run_options.get(),
        enc_input_names, enc_inputs, 5,
        enc_output_names, 5);

    auto* encoded = enc_outputs[0].get();
    int64_t encoded_len = *enc_outputs[1]->GetTensorData<int64_t>();

    encoder_cache.cache_last_channel = std::move(enc_outputs[2]);
    encoder_cache.cache_last_time = std::move(enc_outputs[3]);
    encoder_cache.cache_last_channel_len = std::move(enc_outputs[4]);

    transcript += RunRNNTDecoder(encoded, encoded_len, decoder_state);
  }

  return transcript;
}

std::string NemoBatchASR::RunRNNTDecoder(OrtValue* encoder_output, int64_t encoded_len,
                                          NemotronDecoderState& decoder_state) {
  auto& allocator = model_.allocator_cpu_;
  std::string result;

  auto enc_info = encoder_output->GetTensorTypeAndShapeInfo();
  auto enc_shape = enc_info->GetShape();
  int64_t hidden_dim = enc_shape[1];

  int64_t time_steps = std::min(enc_shape[2], encoded_len);
  const float* enc_data = encoder_output->GetTensorData<float>();

  auto run_options = OrtRunOptions::Create();

  // Pre-allocate reusable tensors
  auto frame_shape = std::array<int64_t, 3>{1, hidden_dim, 1};
  auto encoder_frame = OrtValue::CreateTensor(allocator, frame_shape, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT);
  float* frame_data = encoder_frame->GetTensorMutableData<float>();

  auto targets_shape = std::array<int64_t, 2>{1, 1};
  auto targets = OrtValue::CreateTensor(allocator, targets_shape, ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32);
  int32_t* targets_data = targets->GetTensorMutableData<int32_t>();

  auto tgt_len_shape = std::array<int64_t, 1>{1};
  auto target_length = OrtValue::CreateTensor(allocator, tgt_len_shape, ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32);
  *target_length->GetTensorMutableData<int32_t>() = 1;

  const int max_sym = cache_config_.max_symbols_per_step;

  for (int64_t t = 0; t < time_steps; ++t) {
    // Fill encoder frame data
    for (int64_t d = 0; d < hidden_dim; ++d) {
      frame_data[d] = enc_data[d * enc_shape[2] + t];
    }

    for (int sym = 0; sym < max_sym; ++sym) {
      *targets_data = decoder_state.last_token;

      const char* dec_input_names[] = {
          cache_config_.dec_in_targets.c_str(), cache_config_.dec_in_target_length.c_str(),
          cache_config_.dec_in_states_1.c_str(), cache_config_.dec_in_states_2.c_str()};
      OrtValue* dec_inputs[] = {
          targets.get(), target_length.get(),
          decoder_state.state_1.get(), decoder_state.state_2.get()};

      const char* dec_output_names[] = {
          cache_config_.dec_out_outputs.c_str(), cache_config_.dec_out_prednet_lengths.c_str(),
          cache_config_.dec_out_states_1.c_str(), cache_config_.dec_out_states_2.c_str()};

      auto dec_outputs = decoder_session_->Run(
          run_options.get(),
          dec_input_names, dec_inputs, 4,
          dec_output_names, 4);

      // Run joiner
      const char* join_input_names[] = {
          cache_config_.join_in_encoder.c_str(), cache_config_.join_in_decoder.c_str()};
      OrtValue* join_inputs[] = {
          encoder_frame.get(), dec_outputs[0].get()};

      const char* join_output_names[] = {cache_config_.join_out_logits.c_str()};

      auto join_outputs = joiner_session_->Run(
          run_options.get(),
          join_input_names, join_inputs, 2,
          join_output_names, 1);

      // Find argmax
      const float* logits_data = join_outputs[0]->GetTensorData<float>();
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

      // Blank means current frame is done
      if (best_token == cache_config_.blank_id || best_token >= cache_config_.vocab_size) {
        break;
      }

      // Emit token & update state
      decoder_state.last_token = best_token;
      decoder_state.state_1 = std::move(dec_outputs[2]);
      decoder_state.state_2 = std::move(dec_outputs[3]);

      result += vocab_[best_token];
    }
  }

  return result;
}

std::unique_ptr<BatchASR> CreateBatchASR(Model& model) {
  return std::make_unique<NemoBatchASR>(model);
}

}  // namespace Generators
