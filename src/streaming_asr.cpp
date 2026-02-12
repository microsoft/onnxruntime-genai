// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
//
// StreamingASR implementation — high-level streaming speech recognition.

#include <algorithm>
#include <cmath>
#include <cstring>
#include <fstream>
#include <numeric>
#include <sstream>

#include "generators.h"
#include "streaming_asr.h"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

namespace Generators {

// ─── Mel spectrogram utilities (Slaney scale, matching librosa/NeMo) ────────

// Slaney mel scale: linear below 1000 Hz, logarithmic above
static constexpr float kMinLogHz = 1000.0f;
static constexpr float kMinLogMel = 15.0f;              // 1000 / (200/3)
static constexpr float kLinScale = 200.0f / 3.0f;       // Hz per mel (linear region)
static constexpr float kLogStep = 0.06875177742094912f;  // log(6.4) / 27

static float HzToMel(float hz) {
  if (hz < kMinLogHz) return hz / kLinScale;
  return kMinLogMel + std::log(hz / kMinLogHz) / kLogStep;
}
static float MelToHz(float mel) {
  if (mel < kMinLogMel) return mel * kLinScale;
  return kMinLogHz * std::exp((mel - kMinLogMel) * kLogStep);
}

void StreamingASR::InitMelFilterbank() {
  int num_bins = kFFTSize / 2 + 1;
  float mel_low = HzToMel(0.0f);
  float mel_high = HzToMel(static_cast<float>(kSampleRate) / 2.0f);

  // Compute mel center frequencies in Hz (num_mels + 2 points)
  std::vector<float> mel_f(kNumMels + 2);
  for (int i = 0; i < kNumMels + 2; ++i) {
    float mel = mel_low + (mel_high - mel_low) * i / (kNumMels + 1);
    mel_f[i] = MelToHz(mel);
  }

  // Differences between consecutive mel center frequencies (Hz)
  std::vector<float> fdiff(kNumMels + 1);
  for (int i = 0; i < kNumMels + 1; ++i) {
    fdiff[i] = mel_f[i + 1] - mel_f[i];
  }

  // FFT bin center frequencies in Hz
  std::vector<float> fft_freqs(num_bins);
  for (int k = 0; k < num_bins; ++k) {
    fft_freqs[k] = static_cast<float>(k) * kSampleRate / kFFTSize;
  }

  // Build triangular filterbank with Slaney normalization (matches librosa exactly)
  mel_filters_.resize(kNumMels, std::vector<float>(num_bins, 0.0f));
  for (int m = 0; m < kNumMels; ++m) {
    for (int k = 0; k < num_bins; ++k) {
      float lower = (fft_freqs[k] - mel_f[m]) / (fdiff[m] + 1e-10f);
      float upper = (mel_f[m + 2] - fft_freqs[k]) / (fdiff[m + 1] + 1e-10f);
      mel_filters_[m][k] = std::max(0.0f, std::min(lower, upper));
    }
    // Slaney area normalization: 2 / bandwidth
    float enorm = 2.0f / (mel_f[m + 2] - mel_f[m] + 1e-10f);
    for (int k = 0; k < num_bins; ++k) {
      mel_filters_[m][k] *= enorm;
    }
  }

  // Build Hann window (symmetric, matching torch.hann_window(periodic=False))
  hann_window_.resize(kWinLength);
  for (int i = 0; i < kWinLength; ++i) {
    hann_window_[i] = 0.5f * (1.0f - std::cos(2.0f * static_cast<float>(M_PI) * i / (kWinLength - 1)));
  }
}

std::pair<std::vector<float>, int> StreamingASR::ComputeLogMel(
    const float* audio, size_t num_samples,
    const float* right_ctx, size_t right_ctx_len) {
  // Apply pre-emphasis filter: y[n] = x[n] - 0.97 * x[n-1]
  std::vector<float> preemphasized(num_samples);
  if (num_samples > 0) {
    preemphasized[0] = audio[0] - kPreemph * preemph_last_sample_;
    for (size_t i = 1; i < num_samples; ++i) {
      preemphasized[i] = audio[i] - kPreemph * audio[i - 1];
    }
    preemph_last_sample_ = audio[num_samples - 1];
  }

  // Pre-emphasize right context audio (from start of next chunk).
  // This provides real data for the last mel frame's window instead of zeros.
  // NOTE: preemph_last_sample_ is NOT updated from right context — it stays
  // set from the main audio so the next chunk's pre-emphasis is correct.
  std::vector<float> preemph_right;
  if (right_ctx && right_ctx_len > 0 && num_samples > 0) {
    preemph_right.resize(right_ctx_len);
    preemph_right[0] = right_ctx[0] - kPreemph * audio[num_samples - 1];
    for (size_t i = 1; i < right_ctx_len; ++i) {
      preemph_right[i] = right_ctx[i] - kPreemph * right_ctx[i - 1];
    }
  }

  // Left-only center pad for streaming: prepend overlap from previous chunk.
  // For the first chunk this is zeros (matching center=True left edge).
  int pad = kFFTSize / 2;  // 256 samples
  size_t total_signal = num_samples + preemph_right.size();
  std::vector<float> padded(pad + total_signal);
  std::memcpy(padded.data(), audio_overlap_.data(), pad * sizeof(float));
  std::memcpy(padded.data() + pad, preemphasized.data(), num_samples * sizeof(float));
  if (!preemph_right.empty()) {
    std::memcpy(padded.data() + pad + num_samples, preemph_right.data(),
                preemph_right.size() * sizeof(float));
  }

  // Update overlap buffer from main audio only (not right context)
  if (num_samples >= static_cast<size_t>(pad)) {
    audio_overlap_.assign(preemphasized.data() + num_samples - pad, preemphasized.data() + num_samples);
  } else {
    size_t keep = pad - num_samples;
    std::vector<float> new_overlap(pad, 0.0f);
    std::memcpy(new_overlap.data(), audio_overlap_.data() + num_samples, keep * sizeof(float));
    std::memcpy(new_overlap.data() + keep, preemphasized.data(), num_samples * sizeof(float));
    audio_overlap_ = std::move(new_overlap);
  }

  // Window centering offset: torch.stft centers win_length window within n_fft frame.
  // This shifts the effective analysis position by (n_fft - win_length) / 2 samples.
  constexpr int kWinOffset = (kFFTSize - kWinLength) / 2;  // 56

  // Right-pad to accommodate the window offset for the last frame
  padded.resize(padded.size() + kWinOffset, 0.0f);

  if (static_cast<int>(padded.size()) < kWinOffset + kWinLength) {
    padded.resize(kWinOffset + kWinLength, 0.0f);
  }

  // Frame count from full padded array
  int num_frames = static_cast<int>((padded.size() - kWinOffset - kWinLength) / kHopLength) + 1;

  // Cap frame count: right context provides real data for boundary frames
  // but should not create additional frames beyond what main audio produces.
  int expected_main_padded = pad + static_cast<int>(num_samples) + kWinOffset;
  int expected_frames = (expected_main_padded - kWinOffset - kWinLength) / kHopLength + 1;
  num_frames = std::min(num_frames, expected_frames);

  int num_bins = kFFTSize / 2 + 1;

  std::vector<float> mel_spec(kNumMels * num_frames);

  for (int t = 0; t < num_frames; ++t) {
    std::vector<float> magnitudes(num_bins);
    const float* frame = padded.data() + t * kHopLength + kWinOffset;

    for (int k = 0; k < num_bins; ++k) {
      float real_sum = 0.0f, imag_sum = 0.0f;
      for (int n = 0; n < kWinLength; ++n) {
        float val = frame[n] * hann_window_[n];
        float angle = 2.0f * static_cast<float>(M_PI) * k * n / kFFTSize;
        real_sum += val * std::cos(angle);
        imag_sum -= val * std::sin(angle);
      }
      magnitudes[k] = real_sum * real_sum + imag_sum * imag_sum;
    }

    // Apply mel filterbank
    for (int m = 0; m < kNumMels; ++m) {
      float val = 0.0f;
      for (int k = 0; k < num_bins; ++k) {
        val += mel_filters_[m][k] * magnitudes[k];
      }
      mel_spec[m * num_frames + t] = std::log(val + 5.96046448e-08f);  // log(mel + eps), NeMo default
    }
  }

  return {mel_spec, num_frames};
}

// ─── Vocabulary loading ─────────────────────────────────────────────────────

void StreamingASR::LoadVocab() {
  if (vocab_loaded_) return;

  auto config_path = model_.config_->config_path;

  // Try tokens.txt
  auto tokens_path = config_path / "tokens.txt";
  std::ifstream tokens_file(tokens_path.string());
  if (tokens_file.is_open()) {
    std::string line;
    while (std::getline(tokens_file, line)) {
      // Format: "token_text index" (space-separated) or "token_text\tindex" (tab-separated)
      auto sep_pos = line.rfind(' ');
      if (sep_pos == std::string::npos) sep_pos = line.rfind('\t');
      if (sep_pos != std::string::npos) {
        vocab_.push_back(line.substr(0, sep_pos));
      } else {
        vocab_.push_back(line);
      }
    }
    vocab_loaded_ = true;
    return;
  }

  // Fallback: use tokenizer
  try {
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
    vocab_loaded_ = true;
    return;
  } catch (...) {
  }

  // Last resort
  vocab_.resize(cache_config_.vocab_size);
  for (int i = 0; i < cache_config_.vocab_size; ++i) {
    vocab_[i] = "<" + std::to_string(i) + ">";
  }
  vocab_loaded_ = true;
}

// ─── StreamingASR ───────────────────────────────────────────────────────────

StreamingASR::StreamingASR(Model& model)
    : model_{model} {
  // Get the NemotronSpeechModel to access its sessions
  auto* nemotron_model = dynamic_cast<NemotronSpeechModel*>(&model);
  if (!nemotron_model) {
    throw std::runtime_error("StreamingASR requires a nemotron_speech model type. Got: " + model.config_->model.type);
  }

  encoder_session_ = nemotron_model->session_encoder_.get();
  decoder_session_ = nemotron_model->session_decoder_.get();
  joiner_session_ = nemotron_model->session_joiner_.get();
  cache_config_ = nemotron_model->cache_config_;

  // Initialize mel filterbank
  InitMelFilterbank();

  // Initialize audio overlap buffer for center-padded STFT
  audio_overlap_.assign(kFFTSize / 2, 0.0f);

  // Initialize streaming state
  auto& allocator = model_.allocator_cpu_;
  encoder_cache_.Initialize(cache_config_, allocator);
  decoder_state_.Initialize(cache_config_, allocator);
}

StreamingASR::~StreamingASR() = default;

void StreamingASR::Reset() {
  auto& allocator = model_.allocator_cpu_;
  encoder_cache_.Reset(cache_config_, allocator);
  decoder_state_.Reset(cache_config_, allocator);
  full_transcript_.clear();
  audio_overlap_.assign(kFFTSize / 2, 0.0f);
  preemph_last_sample_ = 0.0f;
  pending_audio_.clear();
  has_pending_ = false;
  chunk_index_ = 0;
}

std::string StreamingASR::TranscribeChunk(const float* audio_data, size_t num_samples) {
  LoadVocab();

  std::string result;

  if (has_pending_) {
    // Process the pending (previous) chunk, using the start of the current
    // chunk as right context.  This gives the last mel frame of the pending
    // chunk real audio data instead of zeros, matching NeMo's full-audio mel.
    size_t right_ctx_len = std::min(num_samples, static_cast<size_t>(kWinLength));
    result = ProcessPendingChunk(audio_data, right_ctx_len);
  }

  // Buffer the current chunk — it will be processed when the next chunk arrives
  pending_audio_.assign(audio_data, audio_data + num_samples);
  has_pending_ = true;

  return result;
}

std::string StreamingASR::ProcessPendingChunk(const float* right_ctx_audio, size_t right_ctx_len) {
  // Compute log-mel spectrogram for the pending audio with right context
  auto [mel_data, num_frames] = ComputeLogMel(pending_audio_.data(), pending_audio_.size(),
                                               right_ctx_audio, right_ctx_len);

  auto& allocator = model_.allocator_cpu_;

  // ── Debug: dump mel and raw audio to /tmp/mel_dumps/ ──
  {
    if (chunk_index_ == 0) {
      std::system("mkdir -p /tmp/mel_dumps");
    }
    std::string mel_path = "/tmp/mel_dumps/mel_chunk_" + std::to_string(chunk_index_) + ".npy";
    SaveNpy(mel_path, mel_data.data(), {static_cast<int64_t>(kNumMels), static_cast<int64_t>(num_frames)});

    std::string audio_path = "/tmp/mel_dumps/audio_chunk_" + std::to_string(chunk_index_) + ".npy";
    SaveNpy(audio_path, pending_audio_.data(), {static_cast<int64_t>(pending_audio_.size())});
  }

  // Create processed_signal: [1, num_mels, num_frames]
  auto signal_shape = std::array<int64_t, 3>{1, kNumMels, num_frames};
  auto processed_signal = OrtValue::CreateTensor(allocator, signal_shape, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT);
  std::memcpy(processed_signal->GetTensorMutableRawData(), mel_data.data(), mel_data.size() * sizeof(float));

  // Create processed_signal_length: [1]
  auto len_shape = std::array<int64_t, 1>{1};
  auto signal_length = OrtValue::CreateTensor(allocator, len_shape, ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64);
  *signal_length->GetTensorMutableData<int64_t>() = static_cast<int64_t>(num_frames);

  // Encoder inputs
  const char* enc_input_names[] = {
      "audio_signal", "length",
      "cache_last_channel", "cache_last_time", "cache_last_channel_len"};
  OrtValue* enc_inputs[] = {
      processed_signal.get(), signal_length.get(),
      encoder_cache_.cache_last_channel.get(),
      encoder_cache_.cache_last_time.get(),
      encoder_cache_.cache_last_channel_len.get()};

  // Encoder outputs — let ORT allocate
  const char* enc_output_names[] = {
      "outputs", "encoded_lengths",
      "cache_last_channel_next", "cache_last_time_next", "cache_last_channel_next_len"};

  // Run encoder
  auto run_options = OrtRunOptions::Create();
  auto enc_outputs = encoder_session_->Run(
      run_options.get(),
      enc_input_names, enc_inputs, 5,
      enc_output_names, 5);

  // Parse encoder outputs
  auto* encoded = enc_outputs[0].get();
  int64_t encoded_len = *enc_outputs[1]->GetTensorData<int64_t>();

  // ── Debug: dump encoder output ──
  {
    auto enc_info2 = encoded->GetTensorTypeAndShapeInfo();
    auto enc_shape2 = enc_info2->GetShape();
    // Shape is [1, hidden_dim, time_steps]
    int64_t total_enc = 1;
    for (auto d : enc_shape2) total_enc *= d;
    std::string enc_path = "/tmp/mel_dumps/encoder_out_" + std::to_string(chunk_index_) + ".npy";
    SaveNpy(enc_path, encoded->GetTensorData<float>(),
            {enc_shape2.begin(), enc_shape2.end()});

    std::string len_path = "/tmp/mel_dumps/encoder_len_" + std::to_string(chunk_index_) + ".txt";
    std::ofstream(len_path) << encoded_len << " shape:";
    for (auto d : enc_shape2) std::ofstream(len_path, std::ios::app) << " " << d;
    std::ofstream(len_path, std::ios::app) << "\n";
  }

  // Update cache
  encoder_cache_.cache_last_channel = std::move(enc_outputs[2]);
  encoder_cache_.cache_last_time = std::move(enc_outputs[3]);
  encoder_cache_.cache_last_channel_len = std::move(enc_outputs[4]);

  // Run RNNT decoder
  std::string chunk_text = RunRNNTDecoder(encoded, encoded_len);
  full_transcript_ += chunk_text;
  chunk_index_++;

  return chunk_text;
}

std::string StreamingASR::RunRNNTDecoder(OrtValue* encoder_output, int64_t encoded_len) {
  auto& allocator = model_.allocator_cpu_;
  std::string result;

  auto enc_info = encoder_output->GetTensorTypeAndShapeInfo();
  auto enc_shape = enc_info->GetShape();
  int64_t hidden_dim = enc_shape[1];
  int64_t total_frames = std::min(enc_shape[2], encoded_len);
  // Drop the last encoder frame per chunk — it has subsampling boundary artifacts
  // (no right context for the conv subsampling layer at the chunk edge).
  // This improves word accuracy (e.g. captures "career path, actually" vs just "career.").
  int64_t time_steps = (total_frames > 1) ? total_frames - 1 : total_frames;
  const float* enc_data = encoder_output->GetTensorData<float>();

  auto run_options = OrtRunOptions::Create();

  // ── Debug: dump decoder LSTM states at start of this chunk ──
  {
    auto s1_shape = decoder_state_.state_1->GetTensorTypeAndShapeInfo()->GetShape();
    SaveNpy("/tmp/mel_dumps/decoder_state1_" + std::to_string(chunk_index_) + ".npy",
            decoder_state_.state_1->GetTensorData<float>(),
            {s1_shape.begin(), s1_shape.end()});
    auto s2_shape = decoder_state_.state_2->GetTensorTypeAndShapeInfo()->GetShape();
    SaveNpy("/tmp/mel_dumps/decoder_state2_" + std::to_string(chunk_index_) + ".npy",
            decoder_state_.state_2->GetTensorData<float>(),
            {s2_shape.begin(), s2_shape.end()});
  }

  // Collect token IDs for this chunk (for debug dump)
  std::vector<int> chunk_tokens;
  // Collect per-step joiner logits info
  std::ofstream step_log("/tmp/mel_dumps/decoder_steps_" + std::to_string(chunk_index_) + ".txt");
  step_log << "# t sym last_token best_token best_score blank_score total_logits\n";

  for (int64_t t = 0; t < time_steps; ++t) {
    // Extract single encoder frame: [1, hidden_dim, 1]
    auto frame_shape = std::array<int64_t, 3>{1, hidden_dim, 1};
    auto encoder_frame = OrtValue::CreateTensor(allocator, frame_shape, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT);
    float* frame_data = encoder_frame->GetTensorMutableData<float>();
    for (int64_t d = 0; d < hidden_dim; ++d) {
      frame_data[d] = enc_data[d * enc_shape[2] + t];
    }

    constexpr int kMaxSymbolsPerStep = 10;
    for (int sym = 0; sym < kMaxSymbolsPerStep; ++sym) {
      // ── Step 1: Run decoder (prediction network) ──
      auto targets_shape = std::array<int64_t, 2>{1, 1};
      auto targets = OrtValue::CreateTensor(allocator, targets_shape, ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32);
      *targets->GetTensorMutableData<int32_t>() = decoder_state_.last_token;

      auto tgt_len_shape = std::array<int64_t, 1>{1};
      auto target_length = OrtValue::CreateTensor(allocator, tgt_len_shape, ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32);
      *target_length->GetTensorMutableData<int32_t>() = 1;

      const char* dec_input_names[] = {
          "targets", "target_length",
          "states.1", "onnx::Slice_3"};
      OrtValue* dec_inputs[] = {
          targets.get(), target_length.get(),
          decoder_state_.state_1.get(), decoder_state_.state_2.get()};

      const char* dec_output_names[] = {
          "outputs", "prednet_lengths", "states", "162"};

      auto dec_outputs = decoder_session_->Run(
          run_options.get(),
          dec_input_names, dec_inputs, 4,
          dec_output_names, 4);

      // dec_outputs[0] = decoder hidden [1, 640, 1]
      // dec_outputs[1] = prednet_lengths [1]
      // dec_outputs[2] = new states h [2, ?, 640]
      // dec_outputs[3] = new states c [2, ?, 640]

      // ── Step 2: Run joiner (joint network) ──
      const char* join_input_names[] = {
          "encoder_outputs", "decoder_outputs"};
      OrtValue* join_inputs[] = {
          encoder_frame.get(), dec_outputs[0].get()};

      const char* join_output_names[] = {"outputs"};

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

      // ── Debug: log this step and dump joiner logits for first chunks ──
      {
        float blank_score = (cache_config_.blank_id < total_logits) ? logits_data[cache_config_.blank_id] : -999.0f;
        step_log << t << " " << sym << " " << decoder_state_.last_token << " "
                 << best_token << " " << best_score << " " << blank_score << " " << total_logits << "\n";

        // Dump full joiner logits for first 5 chunks
        if (chunk_index_ < 5) {
          std::string logit_path = "/tmp/mel_dumps/joiner_logits_" + std::to_string(chunk_index_)
                                   + "_t" + std::to_string(t) + "_s" + std::to_string(sym) + ".npy";
          SaveNpy(logit_path, logits_data, {static_cast<int64_t>(total_logits)});
        }
      }

      // Blank => next time step
      if (best_token == cache_config_.blank_id || best_token >= cache_config_.vocab_size) {
        break;
      }

      // Emit token & update state
      decoder_state_.last_token = best_token;
      decoder_state_.state_1 = std::move(dec_outputs[2]);
      decoder_state_.state_2 = std::move(dec_outputs[3]);
      chunk_tokens.push_back(best_token);

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
    }
  }

  // ── Debug: dump emitted token IDs for this chunk ──
  {
    std::ofstream tf("/tmp/mel_dumps/decoder_tokens_" + std::to_string(chunk_index_) + ".txt");
    for (size_t i = 0; i < chunk_tokens.size(); ++i) {
      tf << chunk_tokens[i];
      if (i + 1 < chunk_tokens.size()) tf << " ";
    }
    tf << "\n";
  }

  return result;
}

void StreamingASR::SaveNpy(const std::string& path, const float* data,
                               const std::vector<int64_t>& shape) {
    std::ofstream f(path, std::ios::binary);
    if (!f) return;

    // Build header string: "{'descr': '<f4', 'fortran_order': False, 'shape': (D0, D1, ...), }\n"
    std::string shape_str = "(";
    for (size_t i = 0; i < shape.size(); ++i) {
      shape_str += std::to_string(shape[i]);
      if (i + 1 < shape.size()) shape_str += ", ";
      else if (shape.size() == 1) shape_str += ",";
    }
    shape_str += ")";
    std::string header = "{'descr': '<f4', 'fortran_order': False, 'shape': " + shape_str + ", }";

    // Pad header to 64-byte alignment (magic=6 + version=2 + header_len=2 + header + '\n')
    size_t prefix_len = 6 + 2 + 2;  // magic + version + header_len
    size_t total = prefix_len + header.size() + 1;  // +1 for trailing '\n'
    size_t pad = (64 - (total % 64)) % 64;
    header.append(pad, ' ');
    header += '\n';

    uint16_t header_len = static_cast<uint16_t>(header.size());

    // Write magic
    f.write("\x93NUMPY", 6);
    // Write version 1.0
    char ver[] = {1, 0};
    f.write(ver, 2);
    // Write header length (little-endian)
    f.write(reinterpret_cast<const char*>(&header_len), 2);
    // Write header
    f.write(header.data(), header.size());
    // Write data
    size_t num_elements = 1;
    for (auto d : shape) num_elements *= static_cast<size_t>(d);
    f.write(reinterpret_cast<const char*>(data), num_elements * sizeof(float));
  }

  std::unique_ptr<StreamingASR> CreateStreamingASR(Model& model) {
  return std::make_unique<StreamingASR>(model);
}

}  // namespace Generators
