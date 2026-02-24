// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
//
// ParakeetStreamingASR — streaming ASR for Parakeet FastConformer + TDT models.
//
// Implements the same streaming algorithm as NVIDIA NeMo's
// speech_to_text_streaming_infer_rnnt.py, adapted for TDT (Token-and-Duration
// Transducer) decoding where the joiner predicts both a token and a duration.
//
// Key difference from NemoStreamingASR (RNNT):
//   - Non-cache-aware encoder: re-encodes [left_context | chunk | right_context]
//     each iteration, with per-window mel normalization
//   - TDT: joiner output has vocab_size + num_durations logits; the duration
//     controls how many encoder frames to advance (not just blank=1)

#include <algorithm>
#include <cmath>
#include <cstring>
#include <fstream>
#include <numeric>

#include "generators.h"
#include "parakeet_streaming_asr.h"
#include "parakeet_mel.h"

namespace Generators {

// ─── Vocabulary loading ─────────────────────────────────────────────────────

void ParakeetStreamingASR::LoadVocab() {
  if (vocab_loaded_) return;

  // Load vocabulary from vocab.txt (one token per line, 0-indexed).
  // We read directly from the file rather than using the tokenizer's Decode()
  // because SPM's Decode strips the leading ▁ space marker for individual tokens,
  // which causes word boundaries to be lost during streaming concatenation.
  auto vocab_path = model_.config_->config_path / "vocab.txt";
  std::ifstream f(vocab_path.string());
  if (f.is_open()) {
    vocab_.resize(config_.vocab_size);
    std::string line;
    int idx = 0;
    while (std::getline(f, line) && idx < config_.vocab_size) {
      // Trim trailing whitespace
      while (!line.empty() && (line.back() == '\r' || line.back() == '\n'))
        line.pop_back();
      vocab_[idx] = line;
      idx++;
    }
    // Fill remaining with empty strings
    for (; idx < config_.vocab_size; idx++) {
      vocab_[idx] = "";
    }
  } else {
    // Fallback: use tokenizer Decode (may lose spaces)
    auto tokenizer = model_.CreateTokenizer();
    vocab_.resize(config_.vocab_size);
    for (int i = 0; i < config_.vocab_size; ++i) {
      try {
        std::vector<int32_t> ids = {static_cast<int32_t>(i)};
        vocab_[i] = tokenizer->Decode(ids);
      } catch (...) {
        vocab_[i] = "";
      }
    }
  }
  vocab_loaded_ = true;
}

// ─── Constructor / Reset ────────────────────────────────────────────────────

ParakeetStreamingASR::ParakeetStreamingASR(Model& model)
    : model_{model} {
  auto* parakeet_model = dynamic_cast<ParakeetSpeechModel*>(&model);
  if (!parakeet_model) {
    throw std::runtime_error("ParakeetStreamingASR requires a parakeet_tdt model type. Got: " + model.config_->model.type);
  }

  encoder_session_ = parakeet_model->session_encoder_.get();
  decoder_session_ = parakeet_model->session_decoder_.get();
  joiner_session_ = parakeet_model->session_joiner_.get();
  config_ = parakeet_model->parakeet_config_;

  // Initialize decoder state (LSTM zeros)
  auto& allocator = model_.allocator_cpu_;
  decoder_state_.Initialize(config_, allocator);
  decoder_initialized_ = false;
}

ParakeetStreamingASR::~ParakeetStreamingASR() = default;

void ParakeetStreamingASR::Reset() {
  auto& allocator = model_.allocator_cpu_;
  decoder_state_.Reset(config_, allocator);
  decoder_initialized_ = false;
  full_transcript_.clear();
  all_audio_.clear();
  processed_audio_samples_ = 0;
  chunk_index_ = 0;
}

// ─── Initialize decoder with blank token ────────────────────────────────────

void ParakeetStreamingASR::InitializeDecoderState() {
  if (decoder_initialized_) return;

  auto& allocator = model_.allocator_cpu_;
  auto run_options = OrtRunOptions::Create();

  // Run decoder with blank_id to get initial decoder_output
  auto targets_shape = std::array<int64_t, 2>{1, 1};
  auto targets = OrtValue::CreateTensor(allocator, targets_shape, ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64);
  *targets->GetTensorMutableData<int64_t>() = static_cast<int64_t>(config_.blank_id);

  auto tgt_len_shape = std::array<int64_t, 1>{1};
  auto target_length = OrtValue::CreateTensor(allocator, tgt_len_shape, ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64);
  *target_length->GetTensorMutableData<int64_t>() = 1;

  const char* dec_input_names[] = {
      config_.dec_in_targets.c_str(), config_.dec_in_target_length.c_str(),
      config_.dec_in_states_1.c_str(), config_.dec_in_states_2.c_str()};
  OrtValue* dec_inputs[] = {
      targets.get(), target_length.get(),
      decoder_state_.state_h.get(), decoder_state_.state_c.get()};

  const char* dec_output_names[] = {
      config_.dec_out_outputs.c_str(), config_.dec_out_prednet_lengths.c_str(),
      config_.dec_out_states_1.c_str(), config_.dec_out_states_2.c_str()};

  auto dec_outputs = decoder_session_->Run(
      run_options.get(),
      dec_input_names, dec_inputs, 4,
      dec_output_names, 4);

  // decoder_output shape: [1, 640, 1] — take last time step
  auto dec_out_shape = dec_outputs[0]->GetTensorTypeAndShapeInfo()->GetShape();
  int64_t dec_dim = dec_out_shape[1];

  // Extract last time step: [1, 640, 1]
  auto frame_shape = std::array<int64_t, 3>{1, dec_dim, 1};
  decoder_state_.decoder_output = OrtValue::CreateTensor(allocator, frame_shape, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT);
  const float* src = dec_outputs[0]->GetTensorData<float>();
  float* dst = decoder_state_.decoder_output->GetTensorMutableData<float>();
  int64_t t_last = dec_out_shape[2] - 1;
  for (int64_t d = 0; d < dec_dim; ++d) {
    dst[d] = src[d * dec_out_shape[2] + t_last];
  }

  decoder_state_.state_h = std::move(dec_outputs[2]);
  decoder_state_.state_c = std::move(dec_outputs[3]);

  decoder_initialized_ = true;
}

// ─── TranscribeChunk ────────────────────────────────────────────────────────

std::string ParakeetStreamingASR::TranscribeChunk(const float* audio_data, size_t num_samples) {
  LoadVocab();
  InitializeDecoderState();

  // Append incoming audio to full audio buffer
  all_audio_.insert(all_audio_.end(), audio_data, audio_data + num_samples);

  const size_t chunk_sz = static_cast<size_t>(config_.chunk_samples);
  const size_t right_ctx = static_cast<size_t>(config_.right_context_samples);
  std::string result;

  // Process chunks only when we have enough audio for right context lookahead.
  // This matches the Python reference which has the full audio available:
  //   win_right = min(chunk_end + right_samples, len(audio))
  // By waiting for right_ctx more samples, we ensure the encoder window
  // includes the right context audio.
  while (processed_audio_samples_ + chunk_sz + right_ctx <= all_audio_.size()) {
    size_t chunk_start = processed_audio_samples_;
    size_t chunk_end = chunk_start + chunk_sz;
    bool is_last = false;  // Not last — there's at least right_ctx of future audio

    result += ProcessChunk(chunk_start, chunk_end, is_last);
    processed_audio_samples_ = chunk_end;
    chunk_index_++;
  }

  return result;
}

std::string ParakeetStreamingASR::Flush() {
  LoadVocab();
  InitializeDecoderState();

  std::string result;
  const size_t chunk_sz = static_cast<size_t>(config_.chunk_samples);

  // Process any remaining complete chunks (without requiring right context)
  while (processed_audio_samples_ + chunk_sz <= all_audio_.size()) {
    size_t chunk_start = processed_audio_samples_;
    size_t chunk_end = chunk_start + chunk_sz;
    bool is_last = (chunk_end + chunk_sz > all_audio_.size());

    result += ProcessChunk(chunk_start, chunk_end, is_last);
    processed_audio_samples_ = chunk_end;
    chunk_index_++;
  }

  // Process final partial chunk if any audio remains
  if (processed_audio_samples_ < all_audio_.size()) {
    size_t chunk_start = processed_audio_samples_;
    size_t chunk_end = all_audio_.size();

    result += ProcessChunk(chunk_start, chunk_end, /*is_last=*/true);
    processed_audio_samples_ = chunk_end;
    chunk_index_++;
  }

  return result;
}

// ─── ProcessChunk: build window, mel, normalize, encode, TDT decode ────────

std::string ParakeetStreamingASR::ProcessChunk(size_t chunk_start, size_t chunk_end, bool is_last) {
  auto& allocator = model_.allocator_cpu_;

  const int hop = config_.hop_length;
  const int sub = config_.subsampling_factor;
  const int encoder_frame_samples = hop * sub;  // 1280

  // Streaming context sizes from config (already aligned to encoder frame boundaries)
  const size_t left_samples = static_cast<size_t>(config_.left_context_samples);
  const size_t right_samples = static_cast<size_t>(config_.right_context_samples);
  // Chunk size aligned
  const size_t chunk_samples_aligned = static_cast<size_t>(
      (static_cast<int>(chunk_end - chunk_start) / encoder_frame_samples) * encoder_frame_samples);

  // Build encoder window: [left_context | chunk | right_context]
  size_t win_left = (chunk_start > left_samples) ? (chunk_start - left_samples) : 0;
  size_t win_right = std::min(chunk_end + right_samples, all_audio_.size());

  // For last chunk, try to include as much context as possible
  if (is_last) {
    size_t target_buf_size = left_samples + chunk_samples_aligned + right_samples;
    size_t actual_size = win_right - win_left;
    if (actual_size < target_buf_size) {
      win_left = (win_right > target_buf_size) ? (win_right - target_buf_size) : 0;
    }
  }

  const float* window_audio = all_audio_.data() + win_left;
  size_t window_len = win_right - win_left;

  // ── Compute mel features on the window ──
  // Use parakeet_mel::ComputeLogMel which matches NeMo's pipeline exactly:
  // preemphasis, symmetric Hann, center-padded STFT, librosa mel filterbank, log(x + 2^-24)
  parakeet_mel::ParakeetMelConfig mel_cfg;
  mel_cfg.num_mels = config_.num_mels;
  mel_cfg.fft_size = config_.fft_size;
  mel_cfg.hop_length = config_.hop_length;
  mel_cfg.win_length = config_.win_length;
  mel_cfg.sample_rate = config_.sample_rate;
  mel_cfg.preemph = config_.preemph;

  int num_mel_frames = 0;
  auto raw_mel = parakeet_mel::ComputeLogMel(window_audio, window_len, mel_cfg, num_mel_frames);
  // raw_mel is [num_mels, num_mel_frames] row-major

  if (num_mel_frames <= 0) return "";

  // ── Per-window normalization (Bessel's correction, per-feature) ──
  // This matches NeMo's normalize_batch with per_feature mode
  const int num_mels = config_.num_mels;
  std::vector<float> mel_normalized(num_mels * num_mel_frames);

  for (int m = 0; m < num_mels; ++m) {
    const float* row = raw_mel.data() + m * num_mel_frames;

    // Compute mean
    double sum = 0.0;
    for (int t = 0; t < num_mel_frames; ++t) sum += row[t];
    float mean = static_cast<float>(sum / num_mel_frames);

    // Compute variance with Bessel's correction
    double var_sum = 0.0;
    for (int t = 0; t < num_mel_frames; ++t) {
      double diff = row[t] - mean;
      var_sum += diff * diff;
    }
    float std_val;
    if (num_mel_frames > 1) {
      std_val = std::sqrt(static_cast<float>(var_sum / (num_mel_frames - 1))) + 1e-5f;
    } else {
      std_val = 1e-5f;
    }

    // Normalize
    float* out_row = mel_normalized.data() + m * num_mel_frames;
    for (int t = 0; t < num_mel_frames; ++t) {
      out_row[t] = (row[t] - mean) / std_val;
    }
  }

  // ── Run encoder ──
  auto signal_shape = std::array<int64_t, 3>{1, static_cast<int64_t>(num_mels), static_cast<int64_t>(num_mel_frames)};
  auto processed_signal = OrtValue::CreateTensor(allocator, signal_shape, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT);
  std::memcpy(processed_signal->GetTensorMutableData<float>(), mel_normalized.data(),
              num_mels * num_mel_frames * sizeof(float));

  auto len_shape = std::array<int64_t, 1>{1};
  auto signal_length = OrtValue::CreateTensor(allocator, len_shape, ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64);
  *signal_length->GetTensorMutableData<int64_t>() = static_cast<int64_t>(num_mel_frames);

  const char* enc_input_names[] = {
      config_.enc_in_audio.c_str(), config_.enc_in_length.c_str()};
  OrtValue* enc_inputs[] = {processed_signal.get(), signal_length.get()};

  const char* enc_output_names[] = {
      config_.enc_out_encoded.c_str(), config_.enc_out_length.c_str()};

  auto run_options = OrtRunOptions::Create();
  auto enc_outputs = encoder_session_->Run(
      run_options.get(),
      enc_input_names, enc_inputs, 2,
      enc_output_names, 2);

  auto* encoded = enc_outputs[0].get();
  int64_t enc_total = *enc_outputs[1]->GetTensorData<int64_t>();

  // ── Strip left context encoder frames ──
  size_t left_ctx_samples = chunk_start - win_left;
  int64_t left_ctx_mel = static_cast<int64_t>(left_ctx_samples) / hop;
  int64_t left_enc = left_ctx_mel / sub;

  // ── Determine decode range (only decode chunk frames, not left/right context) ──
  int64_t decode_start = left_enc;
  int64_t decode_end;

  if (is_last) {
    decode_end = enc_total;
  } else {
    int64_t chunk_actual = static_cast<int64_t>(chunk_end - chunk_start);
    int64_t chunk_mel = chunk_actual / hop;
    int64_t chunk_enc = chunk_mel / sub;
    decode_end = std::min(left_enc + chunk_enc, enc_total);
  }

  if (decode_end <= decode_start) return "";

  // ── TDT greedy decode ──
  std::string chunk_text = RunTDTDecoder(encoded, enc_total, decode_start, decode_end);
  full_transcript_ += chunk_text;

  return chunk_text;
}

// ─── TDT Greedy Decoder ─────────────────────────────────────────────────────

std::string ParakeetStreamingASR::RunTDTDecoder(OrtValue* encoder_output,
                                                 int64_t encoded_len,
                                                 int64_t start_frame,
                                                 int64_t end_frame) {
  auto& allocator = model_.allocator_cpu_;
  std::string result;

  auto enc_info = encoder_output->GetTensorTypeAndShapeInfo();
  auto enc_shape = enc_info->GetShape();
  // enc_shape: [1, 1024, T'] — [batch, hidden_dim, time]
  int64_t hidden_dim = enc_shape[1];
  int64_t enc_time = enc_shape[2];
  const float* enc_data = encoder_output->GetTensorData<float>();

  auto run_options = OrtRunOptions::Create();

  const int num_durations = config_.tdt_num_extra_outputs;
  const int vocab_size = config_.vocab_size;
  const int blank_id = config_.blank_id;
  const int max_sym = config_.max_symbols_per_step;

  int symbols_this_frame = 0;
  int64_t t = start_frame;

  while (t < end_frame) {
    // ── Extract single encoder frame: [1, hidden_dim, 1] ──
    auto frame_shape = std::array<int64_t, 3>{1, hidden_dim, 1};
    auto encoder_frame_raw = OrtValue::CreateTensor(allocator, frame_shape, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT);
    float* frame_data = encoder_frame_raw->GetTensorMutableData<float>();
    for (int64_t d = 0; d < hidden_dim; ++d) {
      frame_data[d] = enc_data[d * enc_time + t];
    }

    // ── Transpose encoder frame to [1, 1, hidden_dim] for joiner ──
    auto enc_join_shape = std::array<int64_t, 3>{1, 1, hidden_dim};
    auto enc_frame_t = OrtValue::CreateTensor(allocator, enc_join_shape, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT);
    std::memcpy(enc_frame_t->GetTensorMutableData<float>(), frame_data, hidden_dim * sizeof(float));

    // ── Transpose decoder output [1, dec_dim, 1] to [1, 1, dec_dim] for joiner ──
    auto dec_shape = decoder_state_.decoder_output->GetTensorTypeAndShapeInfo()->GetShape();
    int64_t dec_dim = dec_shape[1];
    auto dec_join_shape = std::array<int64_t, 3>{1, 1, dec_dim};
    auto dec_frame_t = OrtValue::CreateTensor(allocator, dec_join_shape, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT);
    {
      const float* dec_src = decoder_state_.decoder_output->GetTensorData<float>();
      float* dec_dst = dec_frame_t->GetTensorMutableData<float>();
      // decoder_output is [1, dec_dim, 1] — just copy the dec_dim values
      for (int64_t d = 0; d < dec_dim; ++d) {
        dec_dst[d] = dec_src[d];
      }
    }

    // ── Run joiner → [1, 1, 1, vocab_size + num_durations] ──
    const char* join_input_names[] = {
        config_.join_in_encoder.c_str(), config_.join_in_decoder.c_str()};
    OrtValue* join_inputs[] = {enc_frame_t.get(), dec_frame_t.get()};

    const char* join_output_names[] = {config_.join_out_logits.c_str()};

    auto join_outputs = joiner_session_->Run(
        run_options.get(),
        join_input_names, join_inputs, 2,
        join_output_names, 1);

    const float* logits_data = join_outputs[0]->GetTensorData<float>();

    // ── Token prediction: argmax over first vocab_size logits ──
    int best_token = 0;
    float best_score = logits_data[0];
    for (int i = 1; i < vocab_size; ++i) {
      if (logits_data[i] > best_score) {
        best_score = logits_data[i];
        best_token = i;
      }
    }

    // ── Duration prediction: argmax over next num_durations logits ──
    int skip = 0;
    if (num_durations > 0) {
      float best_dur_score = logits_data[vocab_size];
      for (int i = 1; i < num_durations; ++i) {
        if (logits_data[vocab_size + i] > best_dur_score) {
          best_dur_score = logits_data[vocab_size + i];
          skip = i;
        }
      }
      // Map to actual duration value if available
      if (skip < static_cast<int>(config_.tdt_durations.size())) {
        skip = config_.tdt_durations[skip];
      }
    }

    if (best_token != blank_id) {
      // ── Emit token ──
      symbols_this_frame++;

      if (best_token < static_cast<int>(vocab_.size())) {
        std::string token_str = vocab_[best_token];
        // Replace sentencepiece space marker "▁" with space
        size_t pos = 0;
        while ((pos = token_str.find("\xe2\x96\x81", pos)) != std::string::npos) {
          token_str.replace(pos, 3, " ");
          pos += 1;
        }
        result += token_str;
      }

      // ── Update decoder state with emitted token ──
      auto targets_shape = std::array<int64_t, 2>{1, 1};
      auto targets = OrtValue::CreateTensor(allocator, targets_shape, ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64);
      *targets->GetTensorMutableData<int64_t>() = static_cast<int64_t>(best_token);

      auto tgt_len_shape = std::array<int64_t, 1>{1};
      auto target_length = OrtValue::CreateTensor(allocator, tgt_len_shape, ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64);
      *target_length->GetTensorMutableData<int64_t>() = 1;

      const char* dec_input_names[] = {
          config_.dec_in_targets.c_str(), config_.dec_in_target_length.c_str(),
          config_.dec_in_states_1.c_str(), config_.dec_in_states_2.c_str()};
      OrtValue* dec_inputs[] = {
          targets.get(), target_length.get(),
          decoder_state_.state_h.get(), decoder_state_.state_c.get()};

      const char* dec_output_names[] = {
          config_.dec_out_outputs.c_str(), config_.dec_out_prednet_lengths.c_str(),
          config_.dec_out_states_1.c_str(), config_.dec_out_states_2.c_str()};

      auto dec_outputs = decoder_session_->Run(
          run_options.get(),
          dec_input_names, dec_inputs, 4,
          dec_output_names, 4);

      // Update decoder output: extract last time step [1, dec_dim, 1]
      auto new_dec_shape = dec_outputs[0]->GetTensorTypeAndShapeInfo()->GetShape();
      int64_t new_dec_dim = new_dec_shape[1];
      int64_t new_dec_time = new_dec_shape[2];
      auto new_frame_shape = std::array<int64_t, 3>{1, new_dec_dim, 1};
      decoder_state_.decoder_output = OrtValue::CreateTensor(allocator, new_frame_shape, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT);
      {
        const float* src = dec_outputs[0]->GetTensorData<float>();
        float* dst = decoder_state_.decoder_output->GetTensorMutableData<float>();
        int64_t t_last = new_dec_time - 1;
        for (int64_t d = 0; d < new_dec_dim; ++d) {
          dst[d] = src[d * new_dec_time + t_last];
        }
      }

      decoder_state_.state_h = std::move(dec_outputs[2]);
      decoder_state_.state_c = std::move(dec_outputs[3]);
      decoder_state_.last_token = static_cast<int64_t>(best_token);
    }

    // Handle duration: skip > 0 means advance by 'skip' frames
    if (skip > 0) {
      symbols_this_frame = 0;
    }

    // Safety: force advance if too many symbols at one frame
    if (symbols_this_frame >= max_sym) {
      symbols_this_frame = 0;
      skip = 1;
    }

    // Force advance if blank with duration 0 (prevent infinite loop)
    if (best_token == blank_id && skip == 0) {
      symbols_this_frame = 0;
      skip = 1;
    }

    t += skip;
  }

  return result;
}

std::unique_ptr<StreamingASR> CreateParakeetStreamingASR(Model& model) {
  return std::make_unique<ParakeetStreamingASR>(model);
}

}  // namespace Generators
