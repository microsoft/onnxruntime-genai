// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
//
// Nemotron Speech Streaming ASR — cache-aware encoder + RNNT decoder_joint.

#include <algorithm>
#include <cmath>
#include <cstring>
#include <fstream>
#include <numeric>
#include <sstream>
#include <vector>

#include "../generators.h"
#include "nemotron_speech.h"

namespace Generators {

// ─── NemotronEncoderCache ───────────────────────────────────────────────────

void NemotronEncoderCache::Initialize(const NemotronCacheConfig& cfg, OrtAllocator& allocator) {
  // cache_last_channel: [batch, num_layers, left_context, hidden_dim]
  auto ch_shape = std::array<int64_t, 4>{1, cfg.num_encoder_layers, cfg.left_context, cfg.hidden_dim};
  cache_last_channel = OrtValue::CreateTensor(allocator, ch_shape, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT);
  std::memset(cache_last_channel->GetTensorMutableRawData(), 0,
              1 * cfg.num_encoder_layers * cfg.left_context * cfg.hidden_dim * sizeof(float));

  // cache_last_time: [batch, num_layers, hidden_dim, conv_context]
  auto tm_shape = std::array<int64_t, 4>{1, cfg.num_encoder_layers, cfg.hidden_dim, cfg.conv_context};
  cache_last_time = OrtValue::CreateTensor(allocator, tm_shape, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT);
  std::memset(cache_last_time->GetTensorMutableRawData(), 0,
              1 * cfg.num_encoder_layers * cfg.hidden_dim * cfg.conv_context * sizeof(float));

  // cache_last_channel_len: [1]
  auto len_shape = std::array<int64_t, 1>{1};
  cache_last_channel_len = OrtValue::CreateTensor(allocator, len_shape, ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64);
  *cache_last_channel_len->GetTensorMutableData<int64_t>() = 0;
}

void NemotronEncoderCache::Reset(const NemotronCacheConfig& cfg, OrtAllocator& allocator) {
  Initialize(cfg, allocator);
}

// ─── NemotronDecoderState ───────────────────────────────────────────────────

void NemotronDecoderState::Initialize(const NemotronCacheConfig& cfg, OrtAllocator& allocator) {
  // LSTM states: [lstm_layers, 1, lstm_dim]
  auto state_shape = std::array<int64_t, 3>{cfg.decoder_lstm_layers, 1, cfg.decoder_lstm_dim};
  state_1 = OrtValue::CreateTensor(allocator, state_shape, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT);
  std::memset(state_1->GetTensorMutableRawData(), 0,
              cfg.decoder_lstm_layers * 1 * cfg.decoder_lstm_dim * sizeof(float));

  state_2 = OrtValue::CreateTensor(allocator, state_shape, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT);
  std::memset(state_2->GetTensorMutableRawData(), 0,
              cfg.decoder_lstm_layers * 1 * cfg.decoder_lstm_dim * sizeof(float));

  last_token = cfg.blank_id;  // Start with blank/SOS token (not token 0)
}

void NemotronDecoderState::Reset(const NemotronCacheConfig& cfg, OrtAllocator& allocator) {
  Initialize(cfg, allocator);
}

// ─── Simple log-mel spectrogram (CPU) ───────────────────────────────────────

namespace {

// Hann window centered in n_fft frame (matching torch.stft)
// Window of win_length is placed at offset (fft_size - win_length) / 2
std::vector<float> HannWindow(int fft_size, int win_length) {
  std::vector<float> window(fft_size, 0.0f);
  int offset = (fft_size - win_length) / 2;
  for (int i = 0; i < win_length; ++i) {
    window[offset + i] = 0.5f * (1.0f - std::cos(2.0f * static_cast<float>(M_PI) * i / win_length));
  }
  return window;
}

// Simple DFT-based STFT for a single frame (n_fft samples, centered window)
void ComputeSTFTFrame(const float* frame, int fft_size, const float* window,
                      std::vector<float>& magnitudes) {
  int num_bins = fft_size / 2 + 1;
  magnitudes.resize(num_bins);

  for (int k = 0; k < num_bins; ++k) {
    float real_sum = 0.0f, imag_sum = 0.0f;
    for (int n = 0; n < fft_size; ++n) {
      float val = frame[n] * window[n];
      float angle = 2.0f * static_cast<float>(M_PI) * k * n / fft_size;
      real_sum += val * std::cos(angle);
      imag_sum -= val * std::sin(angle);
    }
    magnitudes[k] = real_sum * real_sum + imag_sum * imag_sum;
  }
}

// Mel filter bank (Slaney scale, matching librosa/NeMo)
// Slaney mel scale: linear below 1000 Hz, logarithmic above
static constexpr float kMinLogHz = 1000.0f;
static constexpr float kMinLogMel = 15.0f;              // 1000 / (200/3)
static constexpr float kLinScale = 200.0f / 3.0f;       // Hz per mel (linear region)
static constexpr float kLogStep = 0.06875177742094912f;  // log(6.4) / 27

float HzToMel(float hz) {
  if (hz < kMinLogHz) return hz / kLinScale;
  return kMinLogMel + std::log(hz / kMinLogHz) / kLogStep;
}
float MelToHz(float mel) {
  if (mel < kMinLogMel) return mel * kLinScale;
  return kMinLogHz * std::exp((mel - kMinLogMel) * kLogStep);
}

std::vector<std::vector<float>> CreateMelFilterbank(int num_mels, int fft_size, int sample_rate) {
  int num_bins = fft_size / 2 + 1;
  float mel_low = HzToMel(0.0f);
  float mel_high = HzToMel(static_cast<float>(sample_rate) / 2.0f);

  // Compute mel center frequencies in Hz (num_mels + 2 points)
  std::vector<float> mel_f(num_mels + 2);
  for (int i = 0; i < num_mels + 2; ++i) {
    float mel = mel_low + (mel_high - mel_low) * i / (num_mels + 1);
    mel_f[i] = MelToHz(mel);
  }

  // Differences between consecutive mel center frequencies (Hz)
  std::vector<float> fdiff(num_mels + 1);
  for (int i = 0; i < num_mels + 1; ++i) {
    fdiff[i] = mel_f[i + 1] - mel_f[i];
  }

  // FFT bin center frequencies in Hz
  std::vector<float> fft_freqs(num_bins);
  for (int k = 0; k < num_bins; ++k) {
    fft_freqs[k] = static_cast<float>(k) * sample_rate / fft_size;
  }

  // Build triangular filterbank with Slaney normalization (matches librosa exactly)
  std::vector<std::vector<float>> filterbank(num_mels, std::vector<float>(num_bins, 0.0f));
  for (int m = 0; m < num_mels; ++m) {
    for (int k = 0; k < num_bins; ++k) {
      float lower = (fft_freqs[k] - mel_f[m]) / (fdiff[m] + 1e-10f);
      float upper = (mel_f[m + 2] - fft_freqs[k]) / (fdiff[m + 1] + 1e-10f);
      filterbank[m][k] = std::max(0.0f, std::min(lower, upper));
    }
    // Slaney area normalization: 2 / bandwidth
    float enorm = 2.0f / (mel_f[m + 2] - mel_f[m] + 1e-10f);
    for (int k = 0; k < num_bins; ++k) {
      filterbank[m][k] *= enorm;
    }
  }
  return filterbank;
}

// Compute log-mel spectrogram for an audio chunk
// Output shape: [1, num_mels, num_frames]
std::vector<float> ComputeLogMel(const float* audio, size_t num_samples,
                                 int num_mels, int fft_size, int hop_length, int win_length,
                                 int sample_rate, int& out_num_frames) {
  static auto mel_filters = CreateMelFilterbank(num_mels, fft_size, sample_rate);
  static auto window = HannWindow(fft_size, win_length);

  // Apply pre-emphasis: y[n] = x[n] - 0.97 * x[n-1]
  constexpr float preemph = 0.97f;
  int n = static_cast<int>(num_samples);
  std::vector<float> preemphasized(n);
  if (n > 0) {
    preemphasized[0] = audio[0];  // No previous sample for first sample
    for (int i = 1; i < n; ++i) {
      preemphasized[i] = audio[i] - preemph * audio[i - 1];
    }
  }

  // Center-pad both sides: fft_size/2 zeros on each side (matching torch.stft center=True)
  int pad = fft_size / 2;
  std::vector<float> padded(pad + n + pad, 0.0f);
  if (n > 0) {
    std::memcpy(padded.data() + pad, preemphasized.data(), n * sizeof(float));
  }

  if (static_cast<int>(padded.size()) < fft_size) {
    padded.resize(fft_size, 0.0f);
  }

  // Frame count using n_fft as frame size (matching torch.stft)
  int num_frames = static_cast<int>((padded.size() - fft_size) / hop_length) + 1;
  out_num_frames = num_frames;

  std::vector<float> magnitudes;
  std::vector<float> mel_spec(num_mels * num_frames);

  for (int t = 0; t < num_frames; ++t) {
    const float* frame = padded.data() + t * hop_length;
    ComputeSTFTFrame(frame, fft_size, window.data(), magnitudes);

    for (int m = 0; m < num_mels; ++m) {
      float val = 0.0f;
      int num_bins = fft_size / 2 + 1;
      for (int k = 0; k < num_bins; ++k) {
        val += mel_filters[m][k] * magnitudes[k];
      }
      // Log-mel: log(mel + eps), NeMo default (log_zero_guard_type=add)
      mel_spec[m * num_frames + t] = std::log(val + 5.96046448e-08f);
    }
  }

  return mel_spec;
}

}  // anonymous namespace

// ─── NemotronSpeechModel ────────────────────────────────────────────────────

NemotronSpeechModel::NemotronSpeechModel(std::unique_ptr<Config> config, OrtEnv& ort_env)
    : Model{std::move(config)} {
  // Parse Nemotron-specific config from genai_config.json if provided
  // Defaults match the 0.6B streaming model
  cache_config_ = NemotronCacheConfig{};

  // Create session options
  encoder_session_options_ = OrtSessionOptions::Create();
  decoder_session_options_ = OrtSessionOptions::Create();
  joiner_session_options_ = OrtSessionOptions::Create();

  if (config_->model.encoder.session_options.has_value()) {
    CreateSessionOptionsFromConfig(config_->model.encoder.session_options.value(),
                                   *encoder_session_options_, true, false);
  } else {
    CreateSessionOptionsFromConfig(config_->model.decoder.session_options,
                                   *encoder_session_options_, true, false);
  }
  CreateSessionOptionsFromConfig(config_->model.decoder.session_options,
                                 *decoder_session_options_, true, false);
  if (config_->model.joiner.session_options.has_value()) {
    CreateSessionOptionsFromConfig(config_->model.joiner.session_options.value(),
                                   *joiner_session_options_, true, false);
  } else {
    CreateSessionOptionsFromConfig(config_->model.decoder.session_options,
                                   *joiner_session_options_, true, false);
  }

  // Load the three ONNX models
  // encoder.onnx - cache-aware streaming encoder
  std::string encoder_filename = config_->model.encoder.filename;
  if (encoder_filename.empty()) encoder_filename = "encoder.onnx";

  // decoder.onnx - RNNT prediction network
  std::string decoder_filename = config_->model.decoder.filename;
  if (decoder_filename.empty()) decoder_filename = "decoder.onnx";

  // joiner.onnx - RNNT joint network
  std::string joiner_filename = config_->model.joiner.filename;
  if (joiner_filename.empty()) joiner_filename = "joiner.onnx";

  session_encoder_ = CreateSession(ort_env, encoder_filename, encoder_session_options_.get());
  session_decoder_ = CreateSession(ort_env, decoder_filename, decoder_session_options_.get());
  session_joiner_ = CreateSession(ort_env, joiner_filename, joiner_session_options_.get());

  session_info_.Add(*session_encoder_);
  session_info_.Add(*session_decoder_);
  session_info_.Add(*session_joiner_);
}

std::unique_ptr<State> NemotronSpeechModel::CreateState(DeviceSpan<int32_t> sequence_lengths,
                                                        const GeneratorParams& params) const {
  return std::make_unique<NemotronSpeechState>(*this, params, sequence_lengths);
}

// ─── NemotronSpeechState ────────────────────────────────────────────────────

NemotronSpeechState::NemotronSpeechState(const NemotronSpeechModel& model,
                                         const GeneratorParams& params,
                                         DeviceSpan<int32_t> sequence_lengths)
    : State{params, model},
      model_{model} {
  // Initialize encoder cache and decoder states
  auto& allocator = model_.p_device_inputs_->GetAllocator();
  encoder_cache_.Initialize(model_.cache_config_, allocator);
  decoder_state_.Initialize(model_.cache_config_, allocator);
}

void NemotronSpeechState::SetExtraInputs(const std::vector<ExtraInput>& extra_inputs) {
  // Audio features can be passed through extra inputs for batch mode
  for (const auto& [name, value] : extra_inputs) {
    if (name == "audio_features" || name == "processed_signal") {
      audio_features_ = std::unique_ptr<OrtValue>(reinterpret_cast<OrtValue*>(value.get()));
    }
  }
}

void NemotronSpeechState::LoadVocab() {
  if (vocab_loaded_) return;

  // Try to load tokenizer vocabulary for RNNT decoding
  // The vocab is typically a sentencepiece model, but we need the token list
  // We'll try to use the existing tokenizer infrastructure
  auto config_path = model_.config_->config_path;

  // Try loading vocab from a simple text file first (tokens.txt)
  auto tokens_path = (config_path / "tokens.txt").string();
  std::ifstream tokens_file(tokens_path);
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

  // If no tokens.txt, create a basic vocab (will use tokenizer for decoding)
  vocab_.resize(model_.cache_config_.vocab_size);
  for (int i = 0; i < model_.cache_config_.vocab_size; ++i) {
    vocab_[i] = "<" + std::to_string(i) + ">";
  }
  vocab_loaded_ = true;
}

void NemotronSpeechState::ResetStreaming() {
  auto& allocator = model_.p_device_inputs_->GetAllocator();
  encoder_cache_.Reset(model_.cache_config_, allocator);
  decoder_state_.Reset(model_.cache_config_, allocator);
  full_transcript_.clear();
  chunk_transcript_.clear();
}

void NemotronSpeechState::RunEncoder(const float* audio_data, size_t num_samples) {
  const auto& cfg = model_.cache_config_;

  // Compute log-mel spectrogram
  int num_frames = 0;
  auto mel_data = ComputeLogMel(audio_data, num_samples,
                                kNumMels, kFFTSize, kHopLength, kWinLength,
                                cfg.sample_rate, num_frames);

  // Create processed_signal tensor: [1, num_mels, num_frames]
  auto signal_shape = std::array<int64_t, 3>{1, kNumMels, num_frames};
  auto& allocator = model_.p_device_inputs_->GetAllocator();
  auto processed_signal = OrtValue::CreateTensor(allocator, signal_shape,
                                                  ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT);
  std::memcpy(processed_signal->GetTensorMutableRawData(), mel_data.data(),
              mel_data.size() * sizeof(float));

  // Create processed_signal_length tensor: [1]
  auto len_shape = std::array<int64_t, 1>{1};
  auto signal_length = OrtValue::CreateTensor(allocator, len_shape,
                                               ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64);
  *signal_length->GetTensorMutableData<int64_t>() = static_cast<int64_t>(num_frames);

  // Set up encoder inputs
  std::vector<const char*> input_names = {
      "audio_signal",
      "length",
      "cache_last_channel",
      "cache_last_time",
      "cache_last_channel_len"};

  std::vector<OrtValue*> inputs = {
      processed_signal.get(),
      signal_length.get(),
      encoder_cache_.cache_last_channel.get(),
      encoder_cache_.cache_last_time.get(),
      encoder_cache_.cache_last_channel_len.get()};

  // Set up encoder outputs
  std::vector<const char*> output_names = {
      "outputs",
      "encoded_lengths",
      "cache_last_channel_next",
      "cache_last_time_next",
      "cache_last_channel_next_len"};

  // Create output tensors (sizes determined by the model)
  // encoded: [1, hidden_dim, time_out]
  // For streaming, time_out is typically num_frames / subsampling_factor
  int time_out = std::max(1, num_frames / 8);  // Approximate: FastConformer uses 8x subsampling
  auto encoded_shape = std::array<int64_t, 3>{1, cfg.hidden_dim, time_out};
  last_encoder_output_ = OrtValue::CreateTensor(allocator, encoded_shape,
                                                 ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT);

  auto enc_len_shape = std::array<int64_t, 1>{1};
  last_encoded_len_ = OrtValue::CreateTensor(allocator, enc_len_shape,
                                              ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64);

  // Allocate output cache tensors
  auto ch_shape = std::array<int64_t, 4>{1, cfg.num_encoder_layers, cfg.left_context, cfg.hidden_dim};
  auto new_cache_channel = OrtValue::CreateTensor(allocator, ch_shape,
                                                   ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT);

  auto tm_shape = std::array<int64_t, 4>{1, cfg.num_encoder_layers, cfg.hidden_dim, cfg.conv_context};
  auto new_cache_time = OrtValue::CreateTensor(allocator, tm_shape,
                                                ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT);

  auto cl_shape = std::array<int64_t, 1>{1};
  auto new_cache_len = OrtValue::CreateTensor(allocator, cl_shape,
                                               ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64);

  std::vector<OrtValue*> outputs = {
      last_encoder_output_.get(),
      last_encoded_len_.get(),
      new_cache_channel.get(),
      new_cache_time.get(),
      new_cache_len.get()};

  // Clear and set State I/O for the encoder run
  input_names_.clear();
  inputs_.clear();
  output_names_.clear();
  outputs_.clear();

  for (size_t i = 0; i < input_names.size(); ++i) {
    input_names_.push_back(input_names[i]);
    inputs_.push_back(inputs[i]);
  }
  for (size_t i = 0; i < output_names.size(); ++i) {
    output_names_.push_back(output_names[i]);
    outputs_.push_back(outputs[i]);
  }

  // Run encoder
  State::Run(*model_.session_encoder_);

  // Update encoder cache with new values
  encoder_cache_.cache_last_channel = std::move(new_cache_channel);
  encoder_cache_.cache_last_time = std::move(new_cache_time);
  encoder_cache_.cache_last_channel_len = std::move(new_cache_len);
}

std::string NemotronSpeechState::RunRNNTDecoder(OrtValue* encoder_output, int64_t encoded_len) {
  const auto& cfg = model_.cache_config_;
  auto& allocator = model_.p_device_inputs_->GetAllocator();

  LoadVocab();

  std::string result;

  // Get encoder output data
  auto enc_info = encoder_output->GetTensorTypeAndShapeInfo();
  auto enc_shape = enc_info->GetShape();
  // enc_shape: [1, hidden_dim, time_steps]
  int64_t hidden_dim = enc_shape[1];
  int64_t time_steps = enc_shape[2];
  const float* enc_data = encoder_output->GetTensorData<float>();

  // Iterate over each encoder time step
  for (int64_t t = 0; t < std::min(time_steps, encoded_len); ++t) {
    // Extract single frame: [1, hidden_dim, 1]
    auto frame_shape = std::array<int64_t, 3>{1, hidden_dim, 1};
    auto encoder_frame = OrtValue::CreateTensor(allocator, frame_shape,
                                                 ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT);
    float* frame_data = encoder_frame->GetTensorMutableData<float>();
    for (int64_t d = 0; d < hidden_dim; ++d) {
      frame_data[d] = enc_data[d * time_steps + t];
    }

    // Greedy RNNT decoding loop for this time step
    constexpr int kMaxSymbolsPerStep = 10;  // Safety limit
    for (int sym = 0; sym < kMaxSymbolsPerStep; ++sym) {
      // ── Step 1: Run decoder (prediction network) ──
      // targets: [1, 1] - last emitted token
      auto targets_shape = std::array<int64_t, 2>{1, 1};
      auto targets = OrtValue::CreateTensor(allocator, targets_shape,
                                             ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32);
      *targets->GetTensorMutableData<int32_t>() = decoder_state_.last_token;

      // target_length: [1]
      auto tgt_len_shape = std::array<int64_t, 1>{1};
      auto target_length = OrtValue::CreateTensor(allocator, tgt_len_shape,
                                                   ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32);
      *target_length->GetTensorMutableData<int32_t>() = 1;

      // Decoder outputs
      // outputs (prediction): [1, decoder_lstm_dim, 1]
      auto dec_out_shape = std::array<int64_t, 3>{1, cfg.decoder_lstm_dim, 1};
      auto decoder_output = OrtValue::CreateTensor(allocator, dec_out_shape,
                                                    ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT);

      // prednet_lengths: [1]
      auto prednet_len_shape = std::array<int64_t, 1>{1};
      auto prednet_lengths = OrtValue::CreateTensor(allocator, prednet_len_shape,
                                                     ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32);

      // New LSTM states: [lstm_layers, 1, lstm_dim]
      auto state_shape = std::array<int64_t, 3>{cfg.decoder_lstm_layers, 1, cfg.decoder_lstm_dim};
      auto new_state_1 = OrtValue::CreateTensor(allocator, state_shape,
                                                  ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT);
      auto new_state_2 = OrtValue::CreateTensor(allocator, state_shape,
                                                  ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT);

      // Decoder I/O names (matching the ONNX graph)
      std::vector<const char*> dec_input_names = {
          "targets", "target_length", "states.1", "onnx::Slice_3"};
      std::vector<OrtValue*> dec_inputs = {
          targets.get(), target_length.get(),
          decoder_state_.state_1.get(), decoder_state_.state_2.get()};

      std::vector<const char*> dec_output_names = {
          "outputs", "prednet_lengths", "states", "162"};
      std::vector<OrtValue*> dec_outputs = {
          decoder_output.get(), prednet_lengths.get(),
          new_state_1.get(), new_state_2.get()};

      // Set State I/O for decoder run
      input_names_.clear();
      inputs_.clear();
      output_names_.clear();
      outputs_.clear();

      for (size_t i = 0; i < dec_input_names.size(); ++i) {
        input_names_.push_back(dec_input_names[i]);
        inputs_.push_back(dec_inputs[i]);
      }
      for (size_t i = 0; i < dec_output_names.size(); ++i) {
        output_names_.push_back(dec_output_names[i]);
        outputs_.push_back(dec_outputs[i]);
      }

      // Run decoder (prediction network)
      State::Run(*model_.session_decoder_);

      // ── Step 2: Run joiner (joint network) ──
      // Joiner takes encoder frame + decoder output → logits
      // Output shape: [1, 1, 1, vocab_size+1]
      auto logits_shape = std::array<int64_t, 4>{1, 1, 1, cfg.vocab_size + 1};
      auto logits = OrtValue::CreateTensor(allocator, logits_shape,
                                            ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT);

      std::vector<const char*> join_input_names = {
          "encoder_outputs", "decoder_outputs"};
      std::vector<OrtValue*> join_inputs = {
          encoder_frame.get(), decoder_output.get()};

      std::vector<const char*> join_output_names = {"outputs"};
      std::vector<OrtValue*> join_outputs = {logits.get()};

      input_names_.clear();
      inputs_.clear();
      output_names_.clear();
      outputs_.clear();

      for (size_t i = 0; i < join_input_names.size(); ++i) {
        input_names_.push_back(join_input_names[i]);
        inputs_.push_back(join_inputs[i]);
      }
      for (size_t i = 0; i < join_output_names.size(); ++i) {
        output_names_.push_back(join_output_names[i]);
        outputs_.push_back(join_outputs[i]);
      }

      // Run joiner (joint network)
      State::Run(*model_.session_joiner_);

      // Find argmax of logits
      const float* logits_data = logits->GetTensorData<float>();
      int total_logits = cfg.vocab_size + 1;
      int best_token = 0;
      float best_score = logits_data[0];
      for (int i = 1; i < total_logits; ++i) {
        if (logits_data[i] > best_score) {
          best_score = logits_data[i];
          best_token = i;
        }
      }

      // Check if it's blank => move to next time step
      if (best_token == cfg.blank_id || best_token >= cfg.vocab_size) {
        break;
      }

      // Emit token — update decoder state for next iteration
      decoder_state_.last_token = best_token;
      decoder_state_.state_1 = std::move(new_state_1);
      decoder_state_.state_2 = std::move(new_state_2);

      // SentencePiece tokens: "▁" prefix means space
      if (best_token < static_cast<int>(vocab_.size())) {
        std::string token_str = vocab_[best_token];
        // Replace sentencepiece space marker with actual space
        size_t pos = 0;
        while ((pos = token_str.find("\xe2\x96\x81", pos)) != std::string::npos) {
          token_str.replace(pos, 3, " ");
          pos += 1;
        }
        result += token_str;
      }
    }
  }

  return result;
}

std::string NemotronSpeechState::TranscribeChunk(const float* audio_data, size_t num_samples) {
  // Run encoder on the audio chunk
  RunEncoder(audio_data, num_samples);

  // Get encoded length
  int64_t encoded_len = *last_encoded_len_->GetTensorData<int64_t>();

  // Run RNNT decoder
  chunk_transcript_ = RunRNNTDecoder(last_encoder_output_.get(), encoded_len);

  // Accumulate
  full_transcript_ += chunk_transcript_;

  return chunk_transcript_;
}

DeviceSpan<float> NemotronSpeechState::Run(int current_length, DeviceSpan<int32_t>& next_tokens,
                                           DeviceSpan<int32_t> next_indices) {
  // In streaming mode, Run() processes audio that was set via SetExtraInputs
  // or TranscribeChunk. For the standard generator pipeline, we return empty.
  if (audio_features_) {
    auto info = audio_features_->GetTensorTypeAndShapeInfo();
    auto shape = info->GetShape();
    const float* audio_data = audio_features_->GetTensorData<float>();
    size_t total = 1;
    for (auto d : shape) total *= d;

    TranscribeChunk(audio_data, total);
    audio_features_.reset();
  }
  return {};
}

OrtValue* NemotronSpeechState::GetOutput(const char* name) {
  if (std::strcmp(name, "encoded") == 0 && last_encoder_output_) {
    return last_encoder_output_.get();
  }
  if (std::strcmp(name, "encoded_len") == 0 && last_encoded_len_) {
    return last_encoded_len_.get();
  }
  return State::GetOutput(name);
}

}  // namespace Generators
