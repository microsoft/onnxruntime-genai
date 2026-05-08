// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
//
// Parakeet TDT speech recognition model
//

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <limits>
#include <stdexcept>

#include "../generators.h"
#include "parakeet.h"

namespace Generators {

void ParakeetTdtConfig::PopulateFromConfig(const Config& config) {
  const auto& enc = config.model.encoder;
  const auto& dec = config.model.decoder;
  const auto& m = config.model;
  const auto& jo = config.model.joiner;

  hidden_dim = enc.hidden_size;
  num_encoder_layers = enc.num_hidden_layers;

  decoder_lstm_dim = dec.hidden_size;
  decoder_lstm_layers = dec.num_hidden_layers;

  subsampling_factor = m.subsampling_factor;
  sample_rate = m.sample_rate;
  chunk_samples = m.chunk_samples;
  blank_id = m.blank_id;
  max_symbols_per_step = m.max_symbols_per_step;
  left_context_samples = m.left_context_samples;
  right_context_samples = m.right_context_samples;

  tdt_durations = m.tdt_durations;

  enc_in_audio = enc.inputs.audio_features;
  enc_out_encoded = enc.outputs.encoder_outputs;
  enc_in_length = enc.inputs.input_lengths;
  enc_out_length = enc.outputs.output_lengths;

  join_in_encoder = jo.inputs.encoder_outputs;
  join_in_decoder = jo.inputs.decoder_outputs;
  join_out_logits = jo.outputs.logits;

  dec_in_targets = dec.inputs.targets;
  dec_in_targets_length = dec.inputs.targets_length;
  dec_in_lstm_hidden_state = dec.inputs.lstm_hidden_state;
  dec_in_lstm_cell_state = dec.inputs.lstm_cell_state;
  dec_out_outputs = dec.outputs.outputs;
  dec_out_outputs_length = dec.outputs.outputs_length;
  dec_out_lstm_hidden_state = dec.outputs.lstm_hidden_state;
  dec_out_lstm_cell_state = dec.outputs.lstm_cell_state;
}

ParakeetTdtModel::ParakeetTdtModel(std::unique_ptr<Config> config, OrtEnv& ort_env)
    : Model{std::move(config)} {
  parakeet_config_.PopulateFromConfig(*config_);

  encoder_session_options_ = OrtSessionOptions::Create();
  decoder_session_options_ = OrtSessionOptions::Create();
  joiner_session_options_ = OrtSessionOptions::Create();

  if (config_->model.encoder.session_options.has_value()) {
    CreateSessionOptionsFromConfig(config_->model.encoder.session_options.value(),
                                   *encoder_session_options_, true);
  } else {
    CreateSessionOptionsFromConfig(config_->model.decoder.session_options,
                                   *encoder_session_options_, true);
  }
  CreateSessionOptionsFromConfig(config_->model.decoder.session_options,
                                 *decoder_session_options_, true);
  if (config_->model.joiner.session_options.has_value()) {
    CreateSessionOptionsFromConfig(config_->model.joiner.session_options.value(),
                                   *joiner_session_options_, true);
  } else {
    CreateSessionOptionsFromConfig(config_->model.decoder.session_options,
                                   *joiner_session_options_, true);
  }

  std::string encoder_filename = config_->model.encoder.filename;

  std::string decoder_filename = config_->model.decoder.filename;

  std::string joiner_filename = config_->model.joiner.filename;

  session_encoder_ = CreateSession(ort_env, encoder_filename, encoder_session_options_.get());
  session_decoder_ = CreateSession(ort_env, decoder_filename, decoder_session_options_.get());
  session_joiner_ = CreateSession(ort_env, joiner_filename, joiner_session_options_.get());

  session_info_.Add(*session_encoder_);
  session_info_.Add(*session_decoder_);
  session_info_.Add(*session_joiner_);
}

std::unique_ptr<State> ParakeetTdtModel::CreateState(DeviceSpan<int32_t> /*sequence_lengths*/,
                                                     const GeneratorParams& params) const {
  return std::make_unique<ParakeetTdtState>(*this, params);
}

ParakeetTdtState::ParakeetTdtState(const ParakeetTdtModel& model, const GeneratorParams& params)
    : State{params, model},
      model_{model},
      cfg_{model.parakeet_config_} {
  logits_size_ = model_.config_->model.vocab_size;

  // Allocate the persistent logits buffer (CPU-resident, one-hot).
  logits_buffer_.assign(static_cast<size_t>(logits_size_), 0.0f);
}

void ParakeetTdtState::InitializeDecoderState() {
  auto& inference_device = *model_.p_device_inputs_;

  auto state_shape = std::array<int64_t, 3>{cfg_.decoder_lstm_layers, 1, cfg_.decoder_lstm_dim};
  dec_.state_h = OrtValue::CreateTensor(inference_device.GetAllocator(), state_shape, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT);
  ByteWrapTensor(inference_device, *dec_.state_h).Zero();
  dec_.state_c = OrtValue::CreateTensor(inference_device.GetAllocator(), state_shape, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT);
  ByteWrapTensor(inference_device, *dec_.state_c).Zero();

  // Prime the decoder with the blank token to obtain the initial decoder_output.
  StepDecoder(static_cast<int32_t>(cfg_.blank_id));
  dec_.last_token = cfg_.blank_id;
}

void ParakeetTdtState::StepDecoder(int32_t token_id) {
  // Tiny scalar inputs stay on CPU; ORT bridges them automatically when the
  // decoder session is on a different EP.
  auto& cpu_allocator = model_.allocator_cpu_;
  auto run_options = OrtRunOptions::Create();

  auto targets_shape = std::array<int64_t, 2>{1, 1};
  auto targets = OrtValue::CreateTensor(cpu_allocator, targets_shape, ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64);
  *targets->GetTensorMutableData<int64_t>() = token_id;

  auto tgt_len_shape = std::array<int64_t, 1>{1};
  auto targets_length = OrtValue::CreateTensor(cpu_allocator, tgt_len_shape, ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64);
  *targets_length->GetTensorMutableData<int64_t>() = 1;

  const char* dec_input_names[] = {
      cfg_.dec_in_targets.c_str(), cfg_.dec_in_targets_length.c_str(),
      cfg_.dec_in_lstm_hidden_state.c_str(), cfg_.dec_in_lstm_cell_state.c_str()};
  OrtValue* dec_inputs[] = {targets.get(), targets_length.get(),
                            dec_.state_h.get(), dec_.state_c.get()};

  const char* dec_output_names[] = {
      cfg_.dec_out_outputs.c_str(), cfg_.dec_out_outputs_length.c_str(),
      cfg_.dec_out_lstm_hidden_state.c_str(), cfg_.dec_out_lstm_cell_state.c_str()};

  auto dec_outputs = model_.session_decoder_->Run(
      run_options.get(),
      dec_input_names, dec_inputs, 4,
      dec_output_names, 4);

  // Decoder is run with targets_length=1, so its output already has shape
  // [1, dec_dim, 1] — exactly what the joiner expects. Just take ownership.
  dec_.decoder_output = std::move(dec_outputs[0]);
  dec_.state_h = std::move(dec_outputs[2]);
  dec_.state_c = std::move(dec_outputs[3]);
  dec_.last_token = token_id;
}

void ParakeetTdtState::EncodeNextChunk() {
  const auto& m = model_.config_->model;
  const int hop = m.hop_length;
  const int sub = cfg_.subsampling_factor;
  const int num_mels = m.num_mels;
  const size_t chunk_sz = static_cast<size_t>(cfg_.chunk_samples);
  const size_t left_samples = static_cast<size_t>(cfg_.left_context_samples);
  const size_t right_samples = static_cast<size_t>(cfg_.right_context_samples);

  // Window in audio-sample space (preserved so the mel-frame slice matches
  // what we'd get if the encoder were fed the same audio range directly).
  size_t chunk_start = next_chunk_start_;
  size_t chunk_end = std::min(chunk_start + chunk_sz, total_audio_);
  bool is_last = (chunk_end >= total_audio_);
  size_t win_left = (chunk_start > left_samples) ? (chunk_start - left_samples) : 0;
  size_t win_right = std::min(chunk_end + right_samples, total_audio_);

  // Convert window + chunk boundaries from sample-space to mel-frame space.
  const int64_t win_left_mel = static_cast<int64_t>(win_left / hop);
  const int64_t win_right_mel = std::min<int64_t>(win_right / hop, total_mel_frames_);
  const int64_t num_mel_frames = win_right_mel - win_left_mel;

  next_chunk_start_ = chunk_end;
  if (is_last) finished_ = true;

  if (num_mel_frames <= 0) {
    current_encoder_cpu_.clear();
    current_enc_time_ = 0;
    current_t_ = 0;
    current_end_frame_ = 0;
    return;
  }

  auto& cpu_allocator = model_.allocator_cpu_;
  auto& cpu_device = *GetDeviceInterface(DeviceType::CPU);
  auto& inference_device = *model_.p_device_inputs_;

  // Slice the cached, globally-normalized mel ([num_mels, T_full]) into a
  // contiguous [1, num_mels, num_mel_frames] tensor staged on CPU, then copy
  // it onto the inference device for the encoder.
  auto signal_shape = std::array<int64_t, 3>{1, num_mels, num_mel_frames};
  auto signal_cpu = OrtValue::CreateTensor(cpu_allocator, signal_shape, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT);
  float* dst = signal_cpu->GetTensorMutableData<float>();
  for (int b = 0; b < num_mels; ++b) {
    const float* src = full_mel_.data() + static_cast<size_t>(b) * total_mel_frames_ + win_left_mel;
    std::memcpy(dst + static_cast<size_t>(b) * num_mel_frames, src,
                static_cast<size_t>(num_mel_frames) * sizeof(float));
  }
  auto processed_signal = OrtValue::CreateTensor(inference_device.GetAllocator(), signal_shape, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT);
  ByteWrapTensor(inference_device, *processed_signal).CopyFrom(ByteWrapTensor(cpu_device, *signal_cpu));

  // Tiny scalar input, keep on CPU; ORT bridges to the session's EP.
  auto len_shape = std::array<int64_t, 1>{1};
  auto signal_length = OrtValue::CreateTensor(cpu_allocator, len_shape, ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64);
  *signal_length->GetTensorMutableData<int64_t>() = num_mel_frames;

  const char* enc_input_names[] = {cfg_.enc_in_audio.c_str(), cfg_.enc_in_length.c_str()};
  OrtValue* enc_inputs[] = {processed_signal.get(), signal_length.get()};
  const char* enc_output_names[] = {cfg_.enc_out_encoded.c_str(), cfg_.enc_out_length.c_str()};

  auto run_options = OrtRunOptions::Create();
  auto enc_outputs = model_.session_encoder_->Run(
      run_options.get(),
      enc_input_names, enc_inputs, 2,
      enc_output_names, 2);

  // Encoder output is on the inference device. Mirror it to CPU once so the
  // per-frame slicing in EmitNextToken stays cheap (no per-frame D2H copies).
  auto enc_shape = enc_outputs[0]->GetTensorTypeAndShapeInfo()->GetShape();
  current_enc_time_ = enc_shape[2];
  auto enc_span = WrapTensor<float>(inference_device, *enc_outputs[0]);
  auto enc_cpu = enc_span.CopyDeviceToCpu();
  current_encoder_cpu_.assign(enc_cpu.begin(), enc_cpu.end());

  // enc_outputs[1] (lengths) is also on device; copy the single int64 to CPU.
  auto enc_len_span = WrapTensor<int64_t>(inference_device, *enc_outputs[1]);
  int64_t enc_total = enc_len_span.CopyDeviceToCpu()[0];

  // Map chunk start/end (sample-space) to encoder-frame indices within this
  // window. encoder_frame_index = mel_frame_index / subsampling_factor.
  // Drop left-context frames; on the last chunk consume every remaining
  // encoder frame (no right-context to trim).
  const int64_t chunk_start_mel = static_cast<int64_t>(chunk_start / hop) - win_left_mel;
  const int64_t chunk_end_mel = static_cast<int64_t>(chunk_end / hop) - win_left_mel;
  int64_t decode_start = chunk_start_mel / sub;
  int64_t decode_end = is_last ? enc_total : std::min(chunk_end_mel / sub, enc_total);
  if (decode_end < decode_start) decode_end = decode_start;  // last-chunk rounding

  current_t_ = decode_start;
  current_end_frame_ = decode_end;
  symbols_this_frame_ = 0;
}

int32_t ParakeetTdtState::EmitNextToken() {
  auto& cpu_allocator = model_.allocator_cpu_;
  auto& cpu_device = *GetDeviceInterface(DeviceType::CPU);
  auto& inference_device = *model_.p_device_inputs_;
  auto run_options = OrtRunOptions::Create();

  const int num_durations = static_cast<int>(cfg_.tdt_durations.size());
  const int blank_id = cfg_.blank_id;
  const int max_sym = cfg_.max_symbols_per_step;
  const int64_t hidden_dim = cfg_.hidden_dim;
  const int64_t dec_dim = cfg_.decoder_lstm_dim;

  while (true) {
    // Advance through chunks until the current encoder window has frames
    // left, or we've consumed the entire utterance.
    while (current_t_ >= current_end_frame_) {
      if (finished_) return static_cast<int32_t>(blank_id);  // eos
      EncodeNextChunk();
    }

    // Build encoder_frame [1, 1, hidden_dim] on CPU from the mirrored
    // encoder output, then copy it onto the inference device for the joiner.
    const float* enc_data = current_encoder_cpu_.data();
    const int64_t enc_time = current_enc_time_;
    auto frame_shape = std::array<int64_t, 3>{1, 1, hidden_dim};
    auto encoder_frame_cpu = OrtValue::CreateTensor(cpu_allocator, frame_shape, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT);
    float* frame_data = encoder_frame_cpu->GetTensorMutableData<float>();
    for (int64_t d = 0; d < hidden_dim; ++d) {
      frame_data[d] = enc_data[d * enc_time + current_t_];
    }
    auto encoder_frame = OrtValue::CreateTensor(inference_device.GetAllocator(), frame_shape, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT);
    ByteWrapTensor(inference_device, *encoder_frame).CopyFrom(ByteWrapTensor(cpu_device, *encoder_frame_cpu));

    // Decoder output is [1, dec_dim, 1] on the inference device; reshape to
    // [1, 1, dec_dim] for the joiner via a same-device byte copy (the
    // underlying memory layout is identical).
    auto dec_shape = std::array<int64_t, 3>{1, 1, dec_dim};
    auto decoder_frame = OrtValue::CreateTensor(inference_device.GetAllocator(), dec_shape, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT);
    ByteWrapTensor(inference_device, *decoder_frame).CopyFrom(ByteWrapTensor(inference_device, *dec_.decoder_output));

    const char* join_input_names[] = {cfg_.join_in_encoder.c_str(), cfg_.join_in_decoder.c_str()};
    OrtValue* join_inputs[] = {encoder_frame.get(), decoder_frame.get()};
    const char* join_output_names[] = {cfg_.join_out_logits.c_str()};

    auto join_outputs = model_.session_joiner_->Run(
        run_options.get(),
        join_input_names, join_inputs, 2,
        join_output_names, 1);

    // Joiner logits land on the inference device; copy to CPU for argmax.
    auto logits_span = WrapTensor<float>(inference_device, *join_outputs[0]);
    auto logits_cpu = logits_span.CopyDeviceToCpu();
    const float* logits_data = logits_cpu.data();

    // Joiner output layout: [token_logits (blank_id + 1) | duration_logits].
    // Token argmax covers indices [0..blank_id] inclusive.
    const int num_tok_logits = blank_id + 1;
    int best_token = 0;
    float best_score = logits_data[0];
    for (int i = 1; i < num_tok_logits; ++i) {
      if (logits_data[i] > best_score) {
        best_score = logits_data[i];
        best_token = i;
      }
    }

    // Duration argmax. When the predicted token is blank, forbid duration
    // index 0: (blank, 0) emits nothing AND doesn't advance current_t_, and
    // since symbols_this_frame_ is only incremented on a non-blank emit, the
    // max_symbols_per_step escape never fires either. The loop would hang on
    // the same frame forever, re-running the joiner with identical inputs.
    // Forcing dur_idx >= 1 on blank guarantees forward progress.
    int dur_idx = 0;
    {
      const int dur_off = num_tok_logits;
      const int start_i = (best_token == blank_id) ? 1 : 0;
      float best_dur_score = -std::numeric_limits<float>::infinity();
      for (int i = start_i; i < num_durations; ++i) {
        if (logits_data[dur_off + i] > best_dur_score) {
          best_dur_score = logits_data[dur_off + i];
          dur_idx = i;
        }
      }
    }
    // Map the duration argmax index to an actual encoder-frame skip count via
    // the TDT duration table (e.g. v3 uses [0,1,2,3,4], so the lookup is the
    // identity here, but the table format supports non-contiguous schedules
    // like [0,1,2,4,8]).
    int skip = cfg_.tdt_durations[dur_idx];

    bool emitted = false;
    int32_t emitted_token = 0;
    if (best_token != blank_id) {
      symbols_this_frame_++;
      emitted_token = static_cast<int32_t>(best_token);
      emitted = true;
      StepDecoder(emitted_token);
    }

    // Frame-advance bookkeeping. Two cases:
    //  - skip > 0: model voted to leave this frame, so reset the per-frame
    //    symbol counter for the next frame.
    //  - skip == 0: model wants to stay (only possible on a non-blank emit
    //    here, since blank+0 is forbidden above). Bump-counter logic already
    //    incremented symbols_this_frame_; if we've hit the cap, force a
    //    one-frame skip to escape runaway emission at this frame.
    if (skip > 0) {
      symbols_this_frame_ = 0;
    } else if (symbols_this_frame_ >= max_sym) {
      symbols_this_frame_ = 0;
      skip = 1;
    }
    current_t_ += skip;

    if (emitted) return emitted_token;
    // Pure-blank frame: nothing to return, loop and process the next frame.
  }
}

void ParakeetTdtState::SetExtraInputs(const std::vector<ExtraInput>& extra_inputs) {
  if (mel_loaded_) return;

  // Locate the pre-computed mel-features tensor produced by ParakeetTdtProcessor.
  // Expected shape: [1, num_mels, total_frames], already globally normalized.
  const Tensor* mel_tensor = nullptr;
  for (const auto& ei : extra_inputs) {
    if (ei.name == "mel_features") {
      mel_tensor = ei.tensor.get();
      break;
    }
  }
  if (!mel_tensor) {
    throw std::runtime_error("ParakeetTdtState::SetExtraInputs: 'mel_features' input is missing.");
  }

  auto info = mel_tensor->ort_tensor_->GetTensorTypeAndShapeInfo();
  auto shape = info->GetShape();
  if (shape.size() != 3 || shape[0] != 1) {
    throw std::runtime_error("ParakeetTdtState::SetExtraInputs: mel_features must have shape [1, num_mels, T].");
  }
  const int num_mels = static_cast<int>(shape[1]);
  total_mel_frames_ = static_cast<int>(shape[2]);
  if (num_mels != model_.config_->model.num_mels) {
    throw std::runtime_error("ParakeetTdtState::SetExtraInputs: mel num_mels mismatch with model config.");
  }

  const float* mel_src = mel_tensor->ort_tensor_->GetTensorData<float>();
  full_mel_.assign(mel_src,
                   mel_src + static_cast<size_t>(num_mels) * total_mel_frames_);

  // Map mel-frame extent back to audio-sample space; this is what the chunk
  // / left-context / right-context windows are expressed in.
  total_audio_ = static_cast<size_t>(total_mel_frames_) * model_.config_->model.hop_length;
  next_chunk_start_ = 0;
  finished_ = (total_audio_ == 0);
  mel_loaded_ = true;
}

DeviceSpan<float> ParakeetTdtState::Run(int /*total_length*/,
                                        DeviceSpan<int32_t>& /*next_tokens*/,
                                        DeviceSpan<int32_t> /*next_indices*/) {
  // First call: prime the decoder LSTM and encode the first chunk. Subsequent
  // calls advance the TDT loop by exactly one emitted token (or return eos
  // when the audio is fully consumed).
  if (!initialized_) {
    InitializeDecoderState();
    EncodeNextChunk();
    initialized_ = true;
  }

  int32_t next_token = EmitNextToken();

  // Encode the emitted token id as a one-hot logits row that the standard
  // search will pick up via argmax.
  std::fill(logits_buffer_.begin(), logits_buffer_.end(), 0.0f);
  if (next_token >= 0 && next_token < logits_size_) {
    logits_buffer_[static_cast<size_t>(next_token)] = 100.0f;
  }

  // Wrap the CPU buffer into a DeviceSpan<float> the search can consume.
  auto* cpu_device = GetDeviceInterface(DeviceType::CPU);
  logits_device_ = cpu_device->WrapMemory(std::span<float>{logits_buffer_.data(), logits_buffer_.size()});
  return logits_device_;
}

}  // namespace Generators
