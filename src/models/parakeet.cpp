// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <algorithm>
#include <cstring>
#include <numeric>
#include <stdexcept>
#include <vector>

#include "../generators.h"
#include "parakeet.h"

namespace Generators {

void ParakeetConfig::PopulateFromConfig(const Config& config) {
  const auto& m = config.model;
  if (m.sample_rate > 0) sample_rate = m.sample_rate;
  if (m.num_mels > 0) num_mels = m.num_mels;
  if (m.fft_size > 0) fft_size = m.fft_size;
  if (m.hop_length > 0) hop_length = m.hop_length;
  if (m.win_length > 0) win_length = m.win_length;
  if (m.preemph != 0.0f) preemph = m.preemph;
  if (m.log_eps > 0.0f) log_eps = m.log_eps;
  if (m.subsampling_factor > 0) subsampling_factor = m.subsampling_factor;
  if (m.vocab_size > 0) vocab_size = m.vocab_size;
  if (m.max_symbols_per_step > 0) max_symbols_per_step = m.max_symbols_per_step;
  // blank_id and blank position: by sherpa convention blank is at index = vocab_size.
  blank_id = (m.blank_id > 0) ? m.blank_id : vocab_size;

  if (m.encoder.hidden_size > 0) encoder_hidden_dim = m.encoder.hidden_size;
  if (m.decoder.hidden_size > 0) decoder_lstm_dim = m.decoder.hidden_size;
  if (m.decoder.num_hidden_layers > 0) decoder_lstm_layers = m.decoder.num_hidden_layers;
}

ParakeetModel::ParakeetModel(std::unique_ptr<Config> config, OrtEnv& ort_env)
    : Model{std::move(config)} {
  parakeet_config_.PopulateFromConfig(*config_);

  encoder_session_options_ = OrtSessionOptions::Create();
  decoder_session_options_ = OrtSessionOptions::Create();
  joiner_session_options_ = OrtSessionOptions::Create();

  CreateSessionOptionsFromConfig(
      config_->model.encoder.session_options.has_value()
          ? config_->model.encoder.session_options.value()
          : config_->model.decoder.session_options,
      *encoder_session_options_, true);
  CreateSessionOptionsFromConfig(config_->model.decoder.session_options,
                                 *decoder_session_options_, true);
  CreateSessionOptionsFromConfig(
      config_->model.joiner.session_options.has_value()
          ? config_->model.joiner.session_options.value()
          : config_->model.decoder.session_options,
      *joiner_session_options_, true);

  std::string enc_fn = config_->model.encoder.filename.empty() ? "encoder.int8.onnx" : config_->model.encoder.filename;
  std::string dec_fn = config_->model.decoder.filename.empty() ? "decoder.int8.onnx" : config_->model.decoder.filename;
  std::string joi_fn = config_->model.joiner.filename.empty() ? "joiner.int8.onnx" : config_->model.joiner.filename;

  session_encoder_ = CreateSession(ort_env, enc_fn, encoder_session_options_.get());
  session_decoder_ = CreateSession(ort_env, dec_fn, decoder_session_options_.get());
  session_joiner_ = CreateSession(ort_env, joi_fn, joiner_session_options_.get());

  session_info_.Add(*session_encoder_);
  session_info_.Add(*session_decoder_);
  session_info_.Add(*session_joiner_);
}

std::unique_ptr<State> ParakeetModel::CreateState(DeviceSpan<int32_t> /*sequence_lengths*/,
                                                  const GeneratorParams& params) const {
  return std::make_unique<ParakeetState>(*this, params);
}

// ---------------- ParakeetState ----------------

ParakeetState::ParakeetState(const ParakeetModel& model, const GeneratorParams& params)
    : State{params, model},
      parakeet_model_{model},
      cfg_{model.parakeet_config_} {
  auto& alloc = model_.allocator_cpu_;

  // Decoder LSTM state buffers
  auto state_shape = std::array<int64_t, 3>{cfg_.decoder_lstm_layers * 2, 1, cfg_.decoder_lstm_dim};
  // NOTE: Parakeet decoder uses a fused LSTM state tensor with shape [num_layers * 2, B, hidden]
  // (h and c stacked as the first dim). Sherpa exports it that way as `states.1`.
  // Actually per metadata it's [2, B, 640] which corresponds to layers stacked. We use what the
  // model declares.
  state_shape = {2, 1, cfg_.decoder_lstm_dim};

  auto state_type = ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;
  if (model_.session_info_.HasInput(cfg_.dec_in_states))
    state_type = model_.session_info_.GetInputDataType(cfg_.dec_in_states);

  dec_states_ = OrtValue::CreateTensor(alloc, state_shape, state_type);
  ByteWrapTensor(*GetDeviceInterface(DeviceType::CPU), *dec_states_).Zero();

  dec_states_zero_ = OrtValue::CreateTensor(alloc, state_shape, state_type);
  ByteWrapTensor(*GetDeviceInterface(DeviceType::CPU), *dec_states_zero_).Zero();

  // Joiner per-step input buffers (channel-first to match joiner: [B, hidden, T=1])
  auto enc_frame_shape = std::array<int64_t, 3>{1, cfg_.encoder_hidden_dim, 1};
  joiner_enc_frame_ = OrtValue::CreateTensor<float>(alloc, enc_frame_shape);
  auto dec_frame_shape = std::array<int64_t, 3>{1, cfg_.decoder_lstm_dim, 1};
  joiner_dec_frame_ = OrtValue::CreateTensor<float>(alloc, dec_frame_shape);

  // Logits buffer the search will read each step. Sized = blank_id + 1 so blank is the last index.
  int64_t logits_vocab = cfg_.blank_id + 1;
  auto logits_shape = std::array<int64_t, 3>{1, 1, logits_vocab};
  logits_ = OrtValue::CreateTensor<float>(alloc, logits_shape);
  output_names_.push_back(model_.config_->model.decoder.outputs.logits.c_str());
  outputs_.push_back(logits_.get());
}

ParakeetState::~ParakeetState() = default;

void ParakeetState::SetExtraInputs(const std::vector<ExtraInput>& extra_inputs) {
  for (const auto& in : extra_inputs) {
    if (in.name == Config::Defaults::AudioFeaturesName ||
        in.name == cfg_.enc_in_audio) {
      mel_input_ = in.tensor;
    }
  }
}

DeviceSpan<float> ParakeetState::Run(int /*current_length*/,
                                     DeviceSpan<int32_t>& /*next_tokens*/,
                                     DeviceSpan<int32_t> /*next_indices*/) {
  if (!decode_started_) {
    if (!mel_input_ || !mel_input_->ort_tensor_)
      throw std::runtime_error("ParakeetState: no audio_features set. Call processor(audios=...) and generator.set_inputs(...).");
    RunEncoderAllChunks();
    decode_started_ = true;
    time_step_ = 0;
    symbol_step_ = 0;
    last_token_ = -1;
    need_decoder_run_ = true;
    exhausted_ = (T_enc_total_ == 0);
  }

  if (exhausted_) {
    return EmitLogits(cfg_.blank_id);  // EOS
  }

  // Run TDT decode steps until we either emit a real token or exhaust frames.
  while (time_step_ < T_enc_total_) {
    if (need_decoder_run_) {
      RunDecoder();
      need_decoder_run_ = false;
    }

    // Copy current encoder frame into joiner input
    {
      const float* enc_src = encoder_out_btc_.data() + static_cast<size_t>(time_step_) * cfg_.encoder_hidden_dim;
      float* dst = joiner_enc_frame_->GetTensorMutableData<float>();
      std::memcpy(dst, enc_src, sizeof(float) * cfg_.encoder_hidden_dim);
    }

    // decoder_out_ is [1, dim, 1] (channels-second). Joiner wants [B,1,dim] (length-second).
    // Both layouts are the same for U=1 because dim and 1 are interchangeable in the byte stream.
    {
      const float* dec_src = decoder_out_->GetTensorData<float>();
      float* dst = joiner_dec_frame_->GetTensorMutableData<float>();
      std::memcpy(dst, dec_src, sizeof(float) * cfg_.decoder_lstm_dim);
    }

    auto logits = RunJoiner();
    // logits layout: [vocab_size + 1 (blank) + num_durations]
    int n_vocab_with_blank = cfg_.blank_id + 1;
    int n_durations = cfg_.num_durations;
    if (static_cast<int>(logits.size()) != n_vocab_with_blank + n_durations) {
      throw std::runtime_error("ParakeetState: unexpected joiner logits size " +
                               std::to_string(logits.size()) + " (expected " +
                               std::to_string(n_vocab_with_blank + n_durations) + ")");
    }

    // Argmax token over [0, n_vocab_with_blank)
    int best_tok = 0;
    float best_tok_score = logits[0];
    for (int i = 1; i < n_vocab_with_blank; ++i) {
      if (logits[i] > best_tok_score) { best_tok_score = logits[i]; best_tok = i; }
    }
    // Argmax duration over [n_vocab_with_blank, n_vocab_with_blank + n_durations)
    int best_dur_idx = 0;
    float best_dur_score = logits[n_vocab_with_blank];
    for (int i = 1; i < n_durations; ++i) {
      float v = logits[n_vocab_with_blank + i];
      if (v > best_dur_score) { best_dur_score = v; best_dur_idx = i; }
    }
    int dur = cfg_.durations[best_dur_idx];

    if (best_tok == cfg_.blank_id) {
      // Blank: advance frame by max(dur, 1) to avoid stalling
      time_step_ += std::max(dur, 1);
      symbol_step_ = 0;
      continue;
    }

    // Emit a real token; update LSTM state
    last_token_ = best_tok;
    {
      // Move new states from decoder output (which we kept) into dec_states_
      // (RunDecoder already wrote new states into dec_states_)
    }
    // Force decoder rerun next step (input token has changed)
    need_decoder_run_ = true;
    symbol_step_++;
    if (dur > 0) {
      time_step_ += dur;
      symbol_step_ = 0;
    } else if (symbol_step_ >= cfg_.max_symbols_per_step) {
      time_step_ += 1;
      symbol_step_ = 0;
    }
    return EmitLogits(best_tok);
  }

  exhausted_ = true;
  return EmitLogits(cfg_.blank_id);
}

void ParakeetState::RunEncoderAllChunks() {
  // mel_input_ shape: per Whisper-style processor we deliver [1, num_mels, total_frames] float32
  auto mel = mel_input_->ort_tensor_.get();
  auto mel_info = mel->GetTensorTypeAndShapeInfo();
  auto mel_shape = mel_info->GetShape();
  if (mel_shape.size() != 3 || mel_shape[0] != 1 || mel_shape[1] != cfg_.num_mels) {
    throw std::runtime_error("ParakeetState: expected mel shape [1, num_mels, T], got " +
                             std::to_string(mel_shape.size()) + "D");
  }
  int64_t total_frames = mel_shape[2];

  // Frame-domain chunking. ratio frames/sec = sample_rate / hop_length.
  double frames_per_sec = static_cast<double>(cfg_.sample_rate) / cfg_.hop_length;
  int64_t chunk_frames = static_cast<int64_t>(cfg_.chunk_seconds * frames_per_sec);
  int64_t overlap_frames = static_cast<int64_t>(cfg_.overlap_seconds * frames_per_sec);
  if (chunk_frames <= overlap_frames) chunk_frames = total_frames;  // fall back to single chunk
  int64_t hop_frames = chunk_frames - overlap_frames;

  // Encoder subsampling factor → encoded frames per encoder time step
  int sub = cfg_.subsampling_factor > 0 ? cfg_.subsampling_factor : 8;
  int64_t overlap_enc = overlap_frames / sub;

  encoder_out_btc_.clear();
  T_enc_total_ = 0;

  auto& alloc = model_.allocator_cpu_;
  const float* mel_data = mel->GetTensorData<float>();

  int chunk_idx = 0;
  for (int64_t start = 0; start < total_frames; start += hop_frames) {
    int64_t end = std::min(start + chunk_frames, total_frames);
    int64_t cur_frames = end - start;
    if (cur_frames <= 0) break;

    // Build chunk mel tensor [1, num_mels, cur_frames]
    auto chunk_shape = std::array<int64_t, 3>{1, cfg_.num_mels, cur_frames};
    auto chunk_mel = OrtValue::CreateTensor<float>(alloc, chunk_shape);
    float* dst = chunk_mel->GetTensorMutableData<float>();
    // mel layout is [1, num_mels, total_frames] row-major: stride for (m, t) = m*total_frames + t
    for (int m = 0; m < cfg_.num_mels; ++m) {
      std::memcpy(dst + static_cast<size_t>(m) * cur_frames,
                  mel_data + static_cast<size_t>(m) * total_frames + start,
                  sizeof(float) * cur_frames);
    }

    // length tensor [1] int64
    auto len_shape = std::array<int64_t, 1>{1};
    auto len_tensor = OrtValue::CreateTensor<int64_t>(alloc, len_shape);
    *len_tensor->GetTensorMutableData<int64_t>() = cur_frames;

    const char* in_names[] = {cfg_.enc_in_audio.c_str(), cfg_.enc_in_length.c_str()};
    const OrtValue* in_vals[] = {chunk_mel.get(), len_tensor.get()};
    const char* out_names[] = {cfg_.enc_out.c_str(), cfg_.enc_out_length.c_str()};
    OrtValue* out_vals[2] = {nullptr, nullptr};
    parakeet_model_.session_encoder_->Run(nullptr, in_names, in_vals, 2, out_names, out_vals, 2);
    std::unique_ptr<OrtValue> enc_out{out_vals[0]};
    std::unique_ptr<OrtValue> enc_out_len{out_vals[1]};

    auto enc_shape = enc_out->GetTensorTypeAndShapeInfo()->GetShape();
    // Expect [1, hidden, T_enc_chunk]
    if (enc_shape.size() != 3 || enc_shape[0] != 1 || enc_shape[1] != cfg_.encoder_hidden_dim) {
      throw std::runtime_error("ParakeetState: unexpected encoder output shape");
    }
    int64_t T_enc = enc_shape[2];
    int64_t real_T_enc = T_enc;
    auto len_dtype = enc_out_len->GetTensorTypeAndShapeInfo()->GetElementType();
    if (len_dtype == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64) {
      real_T_enc = std::min<int64_t>(T_enc, *enc_out_len->GetTensorData<int64_t>());
    } else if (len_dtype == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32) {
      real_T_enc = std::min<int64_t>(T_enc, *enc_out_len->GetTensorData<int32_t>());
    }

    // Drop overlap frames for chunks after the first
    int64_t drop_front = (chunk_idx > 0) ? std::min<int64_t>(overlap_enc / 2, real_T_enc) : 0;
    int64_t drop_back = (end < total_frames) ? std::min<int64_t>(overlap_enc / 2, real_T_enc - drop_front) : 0;
    int64_t keep = real_T_enc - drop_front - drop_back;
    if (keep < 0) keep = 0;

    // Transpose [1, hidden, T_enc] → row-major [keep, hidden] and append
    const float* enc_data = enc_out->GetTensorData<float>();
    size_t old = encoder_out_btc_.size();
    encoder_out_btc_.resize(old + static_cast<size_t>(keep) * cfg_.encoder_hidden_dim);
    for (int64_t t = 0; t < keep; ++t) {
      int64_t src_t = drop_front + t;
      for (int h = 0; h < cfg_.encoder_hidden_dim; ++h) {
        encoder_out_btc_[old + static_cast<size_t>(t) * cfg_.encoder_hidden_dim + h] =
            enc_data[static_cast<size_t>(h) * T_enc + src_t];
      }
    }
    T_enc_total_ += keep;

    if (end >= total_frames) break;
    ++chunk_idx;
  }

  // Encoder no longer needs mel; free reference
  mel_input_.reset();
}

void ParakeetState::RunDecoder() {
  auto& alloc = model_.allocator_cpu_;

  // targets: [1, 1] int32 with last_token_ (or 0 for SOS)
  int32_t tok = (last_token_ < 0) ? 0 : last_token_;
  auto tgt_shape = std::array<int64_t, 2>{1, 1};
  auto targets = OrtValue::CreateTensor<int32_t>(alloc, tgt_shape);
  *targets->GetTensorMutableData<int32_t>() = tok;

  // target_length: [1] int32
  auto tlen_shape = std::array<int64_t, 1>{1};
  auto target_length = OrtValue::CreateTensor<int32_t>(alloc, tlen_shape);
  *target_length->GetTensorMutableData<int32_t>() = 1;

  const char* in_names[] = {
      cfg_.dec_in_targets.c_str(),
      cfg_.dec_in_target_length.c_str(),
      cfg_.dec_in_states.c_str(),
      cfg_.dec_in_states_init.c_str(),
  };
  const OrtValue* in_vals[] = {
      targets.get(),
      target_length.get(),
      dec_states_.get(),
      dec_states_zero_.get(),
  };
  const char* out_names[] = {
      cfg_.dec_out.c_str(),
      cfg_.dec_out_length.c_str(),
      cfg_.dec_out_states.c_str(),
      cfg_.dec_out_states_extra.c_str(),
  };
  OrtValue* out_vals[4] = {nullptr, nullptr, nullptr, nullptr};
  parakeet_model_.session_decoder_->Run(nullptr, in_names, in_vals, 4, out_names, out_vals, 4);

  decoder_out_.reset(out_vals[0]);
  std::unique_ptr<OrtValue> _len{out_vals[1]};
  std::unique_ptr<OrtValue> new_states{out_vals[2]};
  std::unique_ptr<OrtValue> _extra{out_vals[3]};

  // Adopt new states for next call (only when we actually emit a non-blank)
  // We always overwrite for simplicity; if we don't emit, last_token_ unchanged so next
  // RunDecoder call will produce same output anyway.
  dec_states_ = std::move(new_states);
}

std::span<const float> ParakeetState::RunJoiner() {
  const char* in_names[] = {cfg_.join_in_encoder.c_str(), cfg_.join_in_decoder.c_str()};
  const OrtValue* in_vals[] = {joiner_enc_frame_.get(), joiner_dec_frame_.get()};
  const char* out_names[] = {cfg_.join_out.c_str()};
  OrtValue* out_vals[1] = {nullptr};
  parakeet_model_.session_joiner_->Run(nullptr, in_names, in_vals, 2, out_names, out_vals, 1);
  std::unique_ptr<OrtValue> joi_out{out_vals[0]};

  auto info = joi_out->GetTensorTypeAndShapeInfo();
  size_t n = info->GetElementCount();
  // Copy into a stable buffer because joi_out goes out of scope.
  static thread_local std::vector<float> buf;
  buf.assign(joi_out->GetTensorData<float>(), joi_out->GetTensorData<float>() + n);
  return std::span<const float>(buf.data(), buf.size());
}

DeviceSpan<float> ParakeetState::EmitLogits(int32_t tok_id) {
  float* p = logits_->GetTensorMutableData<float>();
  int n = cfg_.blank_id + 1;
  // Negative-infinity-ish for non-selected, large positive for selected.
  std::fill(p, p + n, -1e30f);
  if (tok_id >= 0 && tok_id < n) {
    p[tok_id] = 1.0f;
  }
  return WrapTensor<float>(*GetDeviceInterface(DeviceType::CPU), *logits_);
}

}  // namespace Generators
