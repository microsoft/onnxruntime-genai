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

  // Generation termination relies on the standard search path comparing the
  // last emitted token against config.model.eos_token_id. ParakeetTdtState
  // returns blank_id when finished_, so blank_id must be one of the
  // configured EOS ids or generator loops will hang.
  const auto& eos_ids = config_->model.eos_token_id;
  if (std::find(eos_ids.begin(), eos_ids.end(), parakeet_config_.blank_id) == eos_ids.end()) {
    throw std::runtime_error(
        "Parakeet TDT: model.blank_id (" + std::to_string(parakeet_config_.blank_id) +
        ") must be present in model.eos_token_id so generation terminates correctly.");
  }

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

ParakeetEncoderSubState::ParakeetEncoderSubState(const ParakeetTdtModel& model,
                                                 const GeneratorParams& params)
    : State{params, model},
      model_{model},
      cfg_{model.parakeet_config_} {
  auto& cpu_allocator = model_.allocator_cpu_;

  // signal_length is a tiny scalar input updated per chunk; allocate once.
  auto len_shape = std::array<int64_t, 1>{1};
  signal_length_ = OrtValue::CreateTensor(cpu_allocator, len_shape, ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64);

  mel_input_idx_ = inputs_.size();
  input_names_.push_back(cfg_.enc_in_audio.c_str());
  inputs_.push_back(nullptr);

  length_input_idx_ = inputs_.size();
  input_names_.push_back(cfg_.enc_in_length.c_str());
  inputs_.push_back(signal_length_.get());

  output_names_.push_back(cfg_.enc_out_encoded.c_str());
  outputs_.push_back(nullptr);
  output_names_.push_back(cfg_.enc_out_length.c_str());
  outputs_.push_back(nullptr);

  if (model_.config_->model.encoder.run_options.has_value()) {
    State::SetRunOptions(model_.config_->model.encoder.run_options.value());
  }
}

void ParakeetEncoderSubState::SetInputs(OrtValue* mel_tensor, int64_t num_mel_frames) {
  inputs_[mel_input_idx_] = mel_tensor;
  *signal_length_->GetTensorMutableData<int64_t>() = num_mel_frames;
}

DeviceSpan<float> ParakeetEncoderSubState::Run(int /*total_length*/,
                                               DeviceSpan<int32_t>& /*next_tokens*/,
                                               DeviceSpan<int32_t> /*next_indices*/) {
  State::Run(*model_.session_encoder_);
  return {};
}

ParakeetDecoderSubState::ParakeetDecoderSubState(const ParakeetTdtModel& model,
                                                 const GeneratorParams& params)
    : State{params, model},
      model_{model},
      cfg_{model.parakeet_config_} {
  auto& cpu_allocator = model_.allocator_cpu_;

  auto targets_shape = std::array<int64_t, 2>{1, 1};
  targets_ = OrtValue::CreateTensor(cpu_allocator, targets_shape, ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64);
  *targets_->GetTensorMutableData<int64_t>() = 0;

  auto tgt_len_shape = std::array<int64_t, 1>{1};
  targets_length_ = OrtValue::CreateTensor(cpu_allocator, tgt_len_shape, ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64);
  *targets_length_->GetTensorMutableData<int64_t>() = 1;

  ResetLstmState();

  targets_input_idx_ = inputs_.size();
  input_names_.push_back(cfg_.dec_in_targets.c_str());
  inputs_.push_back(targets_.get());

  targets_length_input_idx_ = inputs_.size();
  input_names_.push_back(cfg_.dec_in_targets_length.c_str());
  inputs_.push_back(targets_length_.get());

  state_h_input_idx_ = inputs_.size();
  input_names_.push_back(cfg_.dec_in_lstm_hidden_state.c_str());
  inputs_.push_back(state_h_.get());

  state_c_input_idx_ = inputs_.size();
  input_names_.push_back(cfg_.dec_in_lstm_cell_state.c_str());
  inputs_.push_back(state_c_.get());

  output_names_.push_back(cfg_.dec_out_outputs.c_str());
  outputs_.push_back(nullptr);
  output_names_.push_back(cfg_.dec_out_outputs_length.c_str());
  outputs_.push_back(nullptr);
  output_names_.push_back(cfg_.dec_out_lstm_hidden_state.c_str());
  outputs_.push_back(nullptr);
  output_names_.push_back(cfg_.dec_out_lstm_cell_state.c_str());
  outputs_.push_back(nullptr);

  if (model_.config_->model.decoder.run_options.has_value()) {
    State::SetRunOptions(model_.config_->model.decoder.run_options.value());
  }
}

void ParakeetDecoderSubState::ResetLstmState() {
  auto& cpu_allocator = model_.allocator_cpu_;
  auto state_shape = std::array<int64_t, 3>{cfg_.decoder_lstm_layers, 1, cfg_.decoder_lstm_dim};
  state_h_ = OrtValue::CreateTensor(cpu_allocator, state_shape, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT);
  std::memset(state_h_->GetTensorMutableData<float>(), 0,
              static_cast<size_t>(cfg_.decoder_lstm_layers) * cfg_.decoder_lstm_dim * sizeof(float));
  state_c_ = OrtValue::CreateTensor(cpu_allocator, state_shape, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT);
  std::memset(state_c_->GetTensorMutableData<float>(), 0,
              static_cast<size_t>(cfg_.decoder_lstm_layers) * cfg_.decoder_lstm_dim * sizeof(float));
  // input slots may not exist yet during ctor; if they do, refresh them.
  if (inputs_.size() > state_c_input_idx_) {
    inputs_[state_h_input_idx_] = state_h_.get();
    inputs_[state_c_input_idx_] = state_c_.get();
  }
}

void ParakeetDecoderSubState::StepWithToken(int32_t token_id) {
  *targets_->GetTensorMutableData<int64_t>() = token_id;

  DeviceSpan<int32_t> dummy_tokens;
  State::Run(*model_.session_decoder_);

  state_h_.reset(outputs_[2]);
  outputs_[2] = nullptr;
  state_c_.reset(outputs_[3]);
  outputs_[3] = nullptr;
  inputs_[state_h_input_idx_] = state_h_.get();
  inputs_[state_c_input_idx_] = state_c_.get();
}

std::unique_ptr<OrtValue> ParakeetDecoderSubState::TakeDecoderOutput() {
  std::unique_ptr<OrtValue> out(outputs_[0]);
  outputs_[0] = nullptr;
  if (outputs_[1] != nullptr) {
    std::unique_ptr<OrtValue> _(outputs_[1]);
    outputs_[1] = nullptr;
  }
  return out;
}

DeviceSpan<float> ParakeetDecoderSubState::Run(int /*total_length*/,
                                               DeviceSpan<int32_t>& /*next_tokens*/,
                                               DeviceSpan<int32_t> /*next_indices*/) {
  State::Run(*model_.session_decoder_);
  return {};
}

ParakeetJoinerSubState::ParakeetJoinerSubState(const ParakeetTdtModel& model,
                                               const GeneratorParams& params)
    : State{params, model},
      model_{model},
      cfg_{model.parakeet_config_} {
  encoder_input_idx_ = inputs_.size();
  input_names_.push_back(cfg_.join_in_encoder.c_str());
  inputs_.push_back(nullptr);

  decoder_input_idx_ = inputs_.size();
  input_names_.push_back(cfg_.join_in_decoder.c_str());
  inputs_.push_back(nullptr);

  output_names_.push_back(cfg_.join_out_logits.c_str());
  outputs_.push_back(nullptr);

  if (model_.config_->model.joiner.run_options.has_value()) {
    State::SetRunOptions(model_.config_->model.joiner.run_options.value());
  }
}

void ParakeetJoinerSubState::SetInputFrames(OrtValue* encoder_frame, OrtValue* decoder_frame) {
  inputs_[encoder_input_idx_] = encoder_frame;
  inputs_[decoder_input_idx_] = decoder_frame;
}

DeviceSpan<float> ParakeetJoinerSubState::Run(int /*total_length*/,
                                              DeviceSpan<int32_t>& /*next_tokens*/,
                                              DeviceSpan<int32_t> /*next_indices*/) {
  State::Run(*model_.session_joiner_);
  return {};
}

ParakeetTdtState::ParakeetTdtState(const ParakeetTdtModel& model, const GeneratorParams& params)
    : TransducerState{params, model},
      model_{model},
      cfg_{model.parakeet_config_} {
  encoder_state_ = std::make_unique<ParakeetEncoderSubState>(model, params);
  decoder_state_ = std::make_unique<ParakeetDecoderSubState>(model, params);
  joiner_state_ = std::make_unique<ParakeetJoinerSubState>(model, params);
}

void ParakeetTdtState::InitializeDecoderState() {
  decoder_state_->StepWithToken(static_cast<int32_t>(cfg_.blank_id));
  decoder_output_ = decoder_state_->TakeDecoderOutput();
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

  // Slice the cached, globally-normalized mel ([num_mels, T_full]) into a
  // contiguous [1, num_mels, num_mel_frames] CPU tensor. CUDA EP will copy
  // to device internally if it needs to.
  auto signal_shape = std::array<int64_t, 3>{1, num_mels, num_mel_frames};
  auto processed_signal = OrtValue::CreateTensor(cpu_allocator, signal_shape, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT);
  float* dst = processed_signal->GetTensorMutableData<float>();
  for (int b = 0; b < num_mels; ++b) {
    const float* src = full_mel_.data() + static_cast<size_t>(b) * total_mel_frames_ + win_left_mel;
    std::memcpy(dst + static_cast<size_t>(b) * num_mel_frames, src,
                static_cast<size_t>(num_mel_frames) * sizeof(float));
  }

  encoder_state_->SetInputs(processed_signal.get(), num_mel_frames);
  DeviceSpan<int32_t> dummy_tokens;
  encoder_state_->Run(0, dummy_tokens, {});

  // Take ownership of the encoder outputs so the sub-state's slots are reset
  // for the next chunk (whose shapes differ).
  std::unique_ptr<OrtValue> enc_output(encoder_state_->outputs_[0]);
  encoder_state_->outputs_[0] = nullptr;
  std::unique_ptr<OrtValue> enc_length(encoder_state_->outputs_[1]);
  encoder_state_->outputs_[1] = nullptr;

  // Encoder output: read directly. Tensors that ORT placed on CUDA we'd need
  // to copy back via the session's own allocator; with our CPU-resident inputs
  // ORT typically routes outputs back to CPU via MemcpyToHost. Handle both
  // just in case.
  auto enc_shape = enc_output->GetTensorTypeAndShapeInfo()->GetShape();
  current_enc_time_ = enc_shape[2];
  size_t enc_elems = 1;
  for (auto d : enc_shape) enc_elems *= static_cast<size_t>(d);

  auto& enc_mi = enc_output->GetTensorMemoryInfo();
  bool enc_on_cpu = (enc_mi.GetDeviceType() == OrtMemoryInfoDeviceType_CPU);
  if (enc_on_cpu) {
    const float* src = enc_output->GetTensorData<float>();
    current_encoder_cpu_.assign(src, src + enc_elems);
  } else {
    auto& inference_device = *model_.p_device_inputs_;
    auto enc_span = WrapTensor<float>(inference_device, *enc_output);
    auto enc_cpu = enc_span.CopyDeviceToCpu();
    current_encoder_cpu_.assign(enc_cpu.begin(), enc_cpu.end());
  }

  auto& enc_len_mi = enc_length->GetTensorMemoryInfo();
  bool enc_len_on_cpu = (enc_len_mi.GetDeviceType() == OrtMemoryInfoDeviceType_CPU);
  int64_t enc_total;
  if (enc_len_on_cpu) {
    enc_total = enc_length->GetTensorData<int64_t>()[0];
  } else {
    auto& inference_device = *model_.p_device_inputs_;
    auto enc_len_span = WrapTensor<int64_t>(inference_device, *enc_length);
    enc_total = enc_len_span.CopyDeviceToCpu()[0];
  }

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

  const int num_durations = static_cast<int>(cfg_.tdt_durations.size());
  const int blank_id = cfg_.blank_id;
  const int max_sym = cfg_.max_symbols_per_step;
  const int64_t hidden_dim = cfg_.hidden_dim;
  const int64_t dec_dim = cfg_.decoder_lstm_dim;

  while (true) {
    while (current_t_ >= current_end_frame_) {
      if (finished_) return static_cast<int32_t>(blank_id);  // eos
      EncodeNextChunk();
    }

    // Build encoder_frame [1, 1, hidden_dim] on CPU from the mirrored
    // encoder output. Allocated once, reused.
    const float* enc_data = current_encoder_cpu_.data();
    const int64_t enc_time = current_enc_time_;
    auto frame_shape = std::array<int64_t, 3>{1, 1, hidden_dim};
    if (!encoder_frame_) {
      encoder_frame_ = OrtValue::CreateTensor(cpu_allocator, frame_shape, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT);
    }
    float* enc_frame_data = encoder_frame_->GetTensorMutableData<float>();
    for (int64_t d = 0; d < hidden_dim; ++d) {
      enc_frame_data[d] = enc_data[d * enc_time + current_t_];
    }

    // Decoder output is [1, dec_dim, 1]; reshape to [1, 1, dec_dim] for
    // the joiner via a plain memcpy (layout is identical for contiguous data).
    auto dec_shape = std::array<int64_t, 3>{1, 1, dec_dim};
    if (!decoder_frame_) {
      decoder_frame_ = OrtValue::CreateTensor(cpu_allocator, dec_shape, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT);
    }
    const float* dec_src = decoder_output_->GetTensorData<float>();
    std::memcpy(decoder_frame_->GetTensorMutableData<float>(), dec_src,
                static_cast<size_t>(dec_dim) * sizeof(float));

    joiner_state_->SetInputFrames(encoder_frame_.get(), decoder_frame_.get());
    DeviceSpan<int32_t> dummy_tokens;
    joiner_state_->Run(0, dummy_tokens, {});

    OrtValue* join_logits = joiner_state_->outputs_[0];
    const float* logits_data;
    std::vector<float> logits_buf;
    auto& mi_jout = join_logits->GetTensorMemoryInfo();
    if (mi_jout.GetDeviceType() == OrtMemoryInfoDeviceType_CPU) {
      logits_data = join_logits->GetTensorData<float>();
    } else {
      auto& inference_device = *model_.p_device_inputs_;
      auto logits_span = WrapTensor<float>(inference_device, *join_logits);
      auto logits_cpu = logits_span.CopyDeviceToCpu();
      logits_buf.assign(logits_cpu.begin(), logits_cpu.end());
      logits_data = logits_buf.data();
    }

    // Joiner output layout: [token_logits (vocab_size) | duration_logits].
    // Token argmax covers all vocab indices; blank_id is one of them.
    const int num_tok_logits = model_.config_->model.vocab_size;
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
      decoder_state_->StepWithToken(emitted_token);
      decoder_output_ = decoder_state_->TakeDecoderOutput();
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
    if (ei.name == Config::Defaults::AudioFeaturesName) {
      mel_tensor = ei.tensor.get();
      break;
    }
  }
  if (!mel_tensor) {
    throw std::runtime_error("ParakeetTdtState::SetExtraInputs: 'audio_features' input is missing.");
  }

  auto info = mel_tensor->ort_tensor_->GetTensorTypeAndShapeInfo();
  auto shape = info->GetShape();
  if (shape.size() != 3 || shape[0] != 1) {
    throw std::runtime_error("ParakeetTdtState::SetExtraInputs: audio_features must have shape [1, num_mels, T].");
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
  throw std::runtime_error(
      "ParakeetTdtState::Run() is not used directly. "
      "TDT models bypass the standard search/logits pipeline; use "
      "Generator::GenerateNextToken() with set_inputs.");
}

void ParakeetTdtState::StepToken() {
  // First call: prime the decoder LSTM and encode the first chunk.
  // Subsequent calls advance the TDT loop by exactly one emitted token (or
  // mark the chunk done when the audio is fully consumed).
  if (!initialized_) {
    InitializeDecoderState();
    EncodeNextChunk();
    initialized_ = true;
  }

  last_tokens_.clear();
  const int blank_id = cfg_.blank_id;
  int32_t next_token = EmitNextToken();

  // EmitNextToken returns blank_id as the end-of-stream marker (set when the
  // trailing chunk is consumed). The framework search would normally treat
  // blank as a real EOS token; here we just stop emitting and let the
  // generator surface this via IsChunkDone().
  if (next_token == blank_id) {
    chunk_done_ = true;
    return;
  }

  all_tokens_.push_back(next_token);
  last_tokens_.push_back(next_token);
}

}  // namespace Generators
