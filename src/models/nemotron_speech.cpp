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
  // cache_last_channel: [num_layers, 1, left_context, hidden_dim]
  auto ch_shape = std::array<int64_t, 4>{cfg.num_encoder_layers, 1, cfg.left_context, cfg.hidden_dim};
  cache_last_channel = OrtValue::CreateTensor(allocator, ch_shape, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT);
  std::memset(cache_last_channel->GetTensorMutableRawData(), 0,
              cfg.num_encoder_layers * 1 * cfg.left_context * cfg.hidden_dim * sizeof(float));

  // cache_last_time: [num_layers, 1, hidden_dim, conv_context]
  auto tm_shape = std::array<int64_t, 4>{cfg.num_encoder_layers, 1, cfg.hidden_dim, cfg.conv_context};
  cache_last_time = OrtValue::CreateTensor(allocator, tm_shape, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT);
  std::memset(cache_last_time->GetTensorMutableRawData(), 0,
              cfg.num_encoder_layers * 1 * cfg.hidden_dim * cfg.conv_context * sizeof(float));

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

  last_token = 0;
}

void NemotronDecoderState::Reset(const NemotronCacheConfig& cfg, OrtAllocator& allocator) {
  Initialize(cfg, allocator);
}

// ─── Simple log-mel spectrogram (CPU) ───────────────────────────────────────

namespace {

// Hann window
std::vector<float> HannWindow(int length) {
  std::vector<float> window(length);
  for (int i = 0; i < length; ++i) {
    window[i] = 0.5f * (1.0f - std::cos(2.0f * static_cast<float>(M_PI) * i / length));
  }
  return window;
}

// Simple DFT-based STFT for a single frame
void ComputeSTFTFrame(const float* frame, int fft_size, const float* window, int win_length,
                      std::vector<float>& magnitudes) {
  int num_bins = fft_size / 2 + 1;
  magnitudes.resize(num_bins);

  for (int k = 0; k < num_bins; ++k) {
    float real_sum = 0.0f, imag_sum = 0.0f;
    for (int n = 0; n < win_length; ++n) {
      float val = frame[n] * window[n];
      float angle = 2.0f * static_cast<float>(M_PI) * k * n / fft_size;
      real_sum += val * std::cos(angle);
      imag_sum -= val * std::sin(angle);
    }
    magnitudes[k] = real_sum * real_sum + imag_sum * imag_sum;
  }
}

// Mel filter bank (HTK style)
float HzToMel(float hz) { return 2595.0f * std::log10(1.0f + hz / 700.0f); }
float MelToHz(float mel) { return 700.0f * (std::pow(10.0f, mel / 2595.0f) - 1.0f); }

std::vector<std::vector<float>> CreateMelFilterbank(int num_mels, int fft_size, int sample_rate) {
  int num_bins = fft_size / 2 + 1;
  float mel_low = HzToMel(0.0f);
  float mel_high = HzToMel(static_cast<float>(sample_rate) / 2.0f);

  std::vector<float> mel_points(num_mels + 2);
  for (int i = 0; i < num_mels + 2; ++i) {
    mel_points[i] = MelToHz(mel_low + (mel_high - mel_low) * i / (num_mels + 1));
  }

  // Convert to FFT bin indices
  std::vector<float> bin_points(num_mels + 2);
  for (int i = 0; i < num_mels + 2; ++i) {
    bin_points[i] = (fft_size + 1) * mel_points[i] / sample_rate;
  }

  std::vector<std::vector<float>> filterbank(num_mels, std::vector<float>(num_bins, 0.0f));
  for (int m = 0; m < num_mels; ++m) {
    for (int k = 0; k < num_bins; ++k) {
      float fk = static_cast<float>(k);
      if (fk >= bin_points[m] && fk <= bin_points[m + 1]) {
        filterbank[m][k] = (fk - bin_points[m]) / (bin_points[m + 1] - bin_points[m] + 1e-10f);
      } else if (fk >= bin_points[m + 1] && fk <= bin_points[m + 2]) {
        filterbank[m][k] = (bin_points[m + 2] - fk) / (bin_points[m + 2] - bin_points[m + 1] + 1e-10f);
      }
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
  static auto window = HannWindow(win_length);

  // Pad audio if needed
  std::vector<float> padded(audio, audio + num_samples);
  if (static_cast<int>(num_samples) < win_length) {
    padded.resize(win_length, 0.0f);
  }

  int num_frames = static_cast<int>((padded.size() - win_length) / hop_length) + 1;
  out_num_frames = num_frames;

  std::vector<float> magnitudes;
  std::vector<float> mel_spec(num_mels * num_frames);

  for (int t = 0; t < num_frames; ++t) {
    const float* frame = padded.data() + t * hop_length;
    ComputeSTFTFrame(frame, fft_size, window.data(), win_length, magnitudes);

    for (int m = 0; m < num_mels; ++m) {
      float val = 0.0f;
      int num_bins = fft_size / 2 + 1;
      for (int k = 0; k < num_bins; ++k) {
        val += mel_filters[m][k] * magnitudes[k];
      }
      // Log-mel with floor
      mel_spec[m * num_frames + t] = std::log(std::max(val, 1e-10f));
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

  if (config_->model.encoder.session_options.has_value()) {
    CreateSessionOptionsFromConfig(config_->model.encoder.session_options.value(),
                                   *encoder_session_options_, true, false);
  } else {
    CreateSessionOptionsFromConfig(config_->model.decoder.session_options,
                                   *encoder_session_options_, true, false);
  }
  CreateSessionOptionsFromConfig(config_->model.decoder.session_options,
                                 *decoder_session_options_, true, false);

  // Load the two ONNX models
  // encoder.onnx - cache-aware streaming encoder
  std::string encoder_filename = config_->model.encoder.filename;
  if (encoder_filename.empty()) encoder_filename = "encoder.onnx";

  // decoder_joint.onnx - RNNT decoder + joint network
  std::string decoder_filename = config_->model.decoder.filename;
  if (decoder_filename.empty()) decoder_filename = "decoder_joint.onnx";

  session_encoder_ = CreateSession(ort_env, encoder_filename, encoder_session_options_.get());
  session_decoder_joint_ = CreateSession(ort_env, decoder_filename, decoder_session_options_.get());

  session_info_.Add(*session_encoder_);
  session_info_.Add(*session_decoder_joint_);
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
      // Format: "token index" or just "token"
      auto space_pos = line.find('\t');
      if (space_pos != std::string::npos) {
        vocab_.push_back(line.substr(0, space_pos));
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
      "processed_signal",
      "processed_signal_length",
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
      "encoded",
      "encoded_len",
      "cache_last_channel_next",
      "cache_last_time_next",
      "cache_last_channel_len_next"};

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
  auto ch_shape = std::array<int64_t, 4>{cfg.num_encoder_layers, 1, cfg.left_context, cfg.hidden_dim};
  auto new_cache_channel = OrtValue::CreateTensor(allocator, ch_shape,
                                                   ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT);

  auto tm_shape = std::array<int64_t, 4>{cfg.num_encoder_layers, 1, cfg.hidden_dim, cfg.conv_context};
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
      // Prepare decoder inputs
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

      // Set up decoder inputs
      std::vector<const char*> dec_input_names = {
          "encoder_outputs", "targets", "target_length",
          "input_states_1", "input_states_2"};

      std::vector<OrtValue*> dec_inputs = {
          encoder_frame.get(), targets.get(), target_length.get(),
          decoder_state_.state_1.get(), decoder_state_.state_2.get()};

      // Set up decoder outputs
      // outputs: [1, 1, vocab_size+1]
      auto logits_shape = std::array<int64_t, 3>{1, 1, cfg.vocab_size + 1};
      auto logits = OrtValue::CreateTensor(allocator, logits_shape,
                                            ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT);

      // New LSTM states
      auto state_shape = std::array<int64_t, 3>{cfg.decoder_lstm_layers, 1, cfg.decoder_lstm_dim};
      auto new_state_1 = OrtValue::CreateTensor(allocator, state_shape,
                                                  ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT);
      auto new_state_2 = OrtValue::CreateTensor(allocator, state_shape,
                                                  ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT);

      std::vector<const char*> dec_output_names = {
          "outputs", "output_states_1", "output_states_2"};

      std::vector<OrtValue*> dec_outputs = {
          logits.get(), new_state_1.get(), new_state_2.get()};

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

      // Run decoder_joint
      State::Run(*model_.session_decoder_joint_);

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

      // Emit token
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
