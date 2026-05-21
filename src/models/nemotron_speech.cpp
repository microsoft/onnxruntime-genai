// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <algorithm>
#include <cstring>
#include <stdexcept>
#include <vector>

#include "../generators.h"
#include "nemo_mel_spectrogram.h"
#include "nemotron_speech.h"

namespace Generators {

void NemotronConfig::PopulateFromConfig(const Config& config) {
  const auto& enc = config.model.encoder;
  const auto& dec = config.model.decoder;
  const auto& jo = config.model.joiner;

  // Encoder dimensions
  hidden_dim = enc.hidden_size;
  num_encoder_layers = enc.num_hidden_layers;

  // Decoder dimensions (LSTM)
  decoder_lstm_dim = dec.hidden_size;
  decoder_lstm_layers = dec.num_hidden_layers;

  // Speech / mel feature config (now at model level)
  num_mels = config.model.num_mels;
  fft_size = config.model.fft_size;
  hop_length = config.model.hop_length;
  win_length = config.model.win_length;
  preemph = config.model.preemph;
  log_eps = config.model.log_eps;
  subsampling_factor = config.model.subsampling_factor;
  left_context = config.model.left_context;
  conv_context = config.model.conv_context;
  pre_encode_cache_size = config.model.pre_encode_cache_size;
  sample_rate = config.model.sample_rate;
  chunk_samples = config.model.chunk_samples;
  blank_id = config.model.blank_id;
  max_symbols_per_step = config.model.max_symbols_per_step;
  blank_penalty = config.search.blank_penalty;

  // Vocab size from top-level config
  vocab_size = config.model.vocab_size;

  // Encoder I/O names
  enc_in_audio = enc.inputs.audio_features;
  enc_out_encoded = enc.outputs.encoder_outputs;

  // Encoder cache I/O names (from encoder inputs/outputs)
  enc_in_length = enc.inputs.input_lengths;
  enc_in_cache_channel = enc.inputs.cache_last_channel;
  enc_in_cache_time = enc.inputs.cache_last_time;
  enc_in_cache_channel_len = enc.inputs.cache_last_channel_len;
  enc_in_lang_id = enc.inputs.lang_id;
  enc_out_length = enc.outputs.output_lengths;
  enc_out_cache_channel = enc.outputs.cache_last_channel_next;
  enc_out_cache_time = enc.outputs.cache_last_time_next;
  enc_out_cache_channel_len = enc.outputs.cache_last_channel_len_next;

  // Joiner I/O names
  join_in_encoder = jo.inputs.encoder_outputs;
  join_in_decoder = jo.inputs.decoder_outputs;
  join_out_logits = jo.outputs.logits;

  // Decoder I/O names (RNNT prediction network)
  dec_in_targets = dec.inputs.targets;
  dec_in_lstm_hidden = dec.inputs.lstm_hidden_state;
  dec_in_lstm_cell = dec.inputs.lstm_cell_state;
  dec_out_outputs = dec.outputs.outputs;
  dec_out_lstm_hidden = dec.outputs.lstm_hidden_state;
  dec_out_lstm_cell = dec.outputs.lstm_cell_state;
}

void NemotronEncoderCache::Initialize(const NemotronConfig& cfg, const SessionInfo& session_info, OrtAllocator& allocator, DeviceInterface& device) {
  auto cache_channel_type = session_info.GetInputDataType(cfg.enc_in_cache_channel);
  auto cache_time_type = session_info.GetInputDataType(cfg.enc_in_cache_time);
  auto cache_channel_len_type = session_info.GetInputDataType(cfg.enc_in_cache_channel_len);

  // cache_last_channel: [batch, num_layers, left_context, hidden_dim] (device memory)
  auto ch_shape = std::array<int64_t, 4>{1, cfg.num_encoder_layers, cfg.left_context, cfg.hidden_dim};
  cache_last_channel = OrtValue::CreateTensor(allocator, ch_shape, cache_channel_type);
  ByteWrapTensor(device, *cache_last_channel).Zero();

  // cache_last_time: [batch, num_layers, hidden_dim, conv_context] (device memory)
  auto tm_shape = std::array<int64_t, 4>{1, cfg.num_encoder_layers, cfg.hidden_dim, cfg.conv_context};
  cache_last_time = OrtValue::CreateTensor(allocator, tm_shape, cache_time_type);
  ByteWrapTensor(device, *cache_last_time).Zero();

  // cache_last_channel_len: [1] int64 scalar; keep on CPU (host-written each step,
  // ORT will copy to device automatically if the session expects it there).
  auto& cpu_alloc = GetDeviceInterface(DeviceType::CPU)->GetAllocator();
  auto len_shape = std::array<int64_t, 1>{1};
  cache_last_channel_len = OrtValue::CreateTensor(cpu_alloc, len_shape, cache_channel_len_type);
  *cache_last_channel_len->GetTensorMutableData<int64_t>() = 0;
}

void NemotronEncoderCache::Reset(const NemotronConfig& cfg, const SessionInfo& session_info, OrtAllocator& allocator, DeviceInterface& device) {
  Initialize(cfg, session_info, allocator, device);
}

void NemotronDecoderState::Initialize(const NemotronConfig& cfg, const SessionInfo& session_info, OrtAllocator& allocator, DeviceInterface& device) {
  auto lstm_hidden_type = session_info.GetInputDataType(cfg.dec_in_lstm_hidden);
  auto lstm_cell_type = session_info.GetInputDataType(cfg.dec_in_lstm_cell);

  // LSTM states: [lstm_layers, 1, lstm_dim] (device memory)
  auto state_shape = std::array<int64_t, 3>{cfg.decoder_lstm_layers, 1, cfg.decoder_lstm_dim};
  lstm_hidden_state = OrtValue::CreateTensor(allocator, state_shape, lstm_hidden_type);
  ByteWrapTensor(device, *lstm_hidden_state).Zero();

  lstm_cell_state = OrtValue::CreateTensor(allocator, state_shape, lstm_cell_type);
  ByteWrapTensor(device, *lstm_cell_state).Zero();

  last_token = cfg.blank_id;  // Start with blank/SOS token
}

void NemotronDecoderState::Reset(const NemotronConfig& cfg, const SessionInfo& session_info, OrtAllocator& allocator, DeviceInterface& device) {
  Initialize(cfg, session_info, allocator, device);
}

NemotronSpeechModel::NemotronSpeechModel(std::unique_ptr<Config> config, OrtEnv& ort_env)
    : Model{std::move(config)} {
  nemotron_config_ = NemotronConfig{};
  nemotron_config_.PopulateFromConfig(*config_);

  // Create session options
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

  // Load the three ONNX models
  std::string encoder_filename = config_->model.encoder.filename;
  if (encoder_filename.empty()) encoder_filename = "encoder.onnx";

  std::string decoder_filename = config_->model.decoder.filename;
  if (decoder_filename.empty()) decoder_filename = "decoder.onnx";

  std::string joiner_filename = config_->model.joiner.filename;
  if (joiner_filename.empty()) joiner_filename = "joiner.onnx";

  session_encoder_ = CreateSession(ort_env, encoder_filename, encoder_session_options_.get());
  session_decoder_ = CreateSession(ort_env, decoder_filename, decoder_session_options_.get());
  session_joiner_ = CreateSession(ort_env, joiner_filename, joiner_session_options_.get());

  session_info_.Add(*session_encoder_);
  session_info_.Add(*session_decoder_);
  session_info_.Add(*session_joiner_);
}

std::unique_ptr<State> NemotronSpeechModel::CreateState(DeviceSpan<int32_t> /*sequence_lengths*/,
                                                        const GeneratorParams& params) const {
  return std::make_unique<NemotronSpeechState>(*this, params);
}

NemotronEncoderSubState::NemotronEncoderSubState(const NemotronSpeechModel& model, const GeneratorParams& params)
    : State{params, model},
      model_{model} {
  auto& cfg = model_.nemotron_config_;
  auto& cpu_dev = *GetDeviceInterface(DeviceType::CPU);
  auto& cpu_alloc = cpu_dev.GetAllocator();
  auto& inf_dev = *model_.p_device_inputs_;
  auto& inf_alloc = inf_dev.GetAllocator();

  // Allocate cache on inference device so ORT consumes it without an
  // implicit CPU→device copy on every Run.
  cache_.Initialize(cfg, model_.session_info_, inf_alloc, inf_dev);

  has_length_input_ = model_.session_info_.HasInput(cfg.enc_in_length);

  if (has_length_input_) {
    auto len_type = model_.session_info_.GetInputDataType(cfg.enc_in_length);
    auto len_shape = std::array<int64_t, 1>{1};
    signal_length_ = OrtValue::CreateTensor(cpu_alloc, len_shape, len_type);
  }

  // Register inputs: mel, [length], cache_channel, cache_time, cache_channel_len
  mel_input_idx_ = inputs_.size();
  input_names_.push_back(cfg.enc_in_audio.c_str());
  inputs_.push_back(nullptr);

  if (has_length_input_) {
    length_input_idx_ = inputs_.size();
    input_names_.push_back(cfg.enc_in_length.c_str());
    inputs_.push_back(signal_length_.get());
  }

  cache_channel_input_idx_ = inputs_.size();
  input_names_.push_back(cfg.enc_in_cache_channel.c_str());
  inputs_.push_back(cache_.cache_last_channel.get());

  cache_time_input_idx_ = inputs_.size();
  input_names_.push_back(cfg.enc_in_cache_time.c_str());
  inputs_.push_back(cache_.cache_last_time.get());

  cache_channel_len_input_idx_ = inputs_.size();
  input_names_.push_back(cfg.enc_in_cache_channel_len.c_str());
  inputs_.push_back(cache_.cache_last_channel_len.get());

  // Optional lang_id input (multilingual prompt-conditioned encoder).
  // Shape: [1] int64 scalar carrying the language index; the encoder graph
  // builds the one-hot prompt tensor internally.
  has_lang_id_input_ = !cfg.enc_in_lang_id.empty() && model_.session_info_.HasInput(cfg.enc_in_lang_id);
  if (has_lang_id_input_) {
    auto lang_id_type = model_.session_info_.GetInputDataType(cfg.enc_in_lang_id);
    if (lang_id_type != ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64)
      throw std::runtime_error("Encoder lang_id input must be int64");
    auto lang_id_shape = std::array<int64_t, 1>{1};
    lang_id_tensor_ = OrtValue::CreateTensor(cpu_alloc, lang_id_shape, lang_id_type);
    // Default to language index 0; callers can change it per-generator at runtime
    // via OgaGenerator_SetRuntimeOption(gen, "lang_id", "<int>").
    *lang_id_tensor_->GetTensorMutableData<int64_t>() = 0;
    lang_id_input_idx_ = inputs_.size();
    input_names_.push_back(cfg.enc_in_lang_id.c_str());
    inputs_.push_back(lang_id_tensor_.get());
  }

  output_names_.push_back(cfg.enc_out_encoded.c_str());
  outputs_.push_back(nullptr);
  encoded_output_idx_ = outputs_.size() - 1;

  output_names_.push_back(cfg.enc_out_length.c_str());
  outputs_.push_back(nullptr);
  length_output_idx_ = outputs_.size() - 1;

  output_names_.push_back(cfg.enc_out_cache_channel.c_str());
  outputs_.push_back(nullptr);
  cache_channel_output_idx_ = outputs_.size() - 1;

  output_names_.push_back(cfg.enc_out_cache_time.c_str());
  outputs_.push_back(nullptr);
  cache_time_output_idx_ = outputs_.size() - 1;

  output_names_.push_back(cfg.enc_out_cache_channel_len.c_str());
  outputs_.push_back(nullptr);
  cache_channel_len_output_idx_ = outputs_.size() - 1;

  // Pre-allocate all encoder outputs at known shapes so ORT writes them
  // directly to the desired device (no implicit CPU fallback when an output
  // slot is null).
  //
  // T_enc per streaming chunk is deterministic: each chunk feeds
  // (chunk_samples / hop_length) new mel frames; the encoder subsamples by
  // `subsampling_factor` and emits only the frames corresponding to that new
  // audio (the `pre_encode_cache_size` frames carried over from the previous
  // chunk are consumed for receptive-field padding and don't produce extra
  // output frames).
  const int mel_frames_per_chunk = cfg.chunk_samples / cfg.hop_length;
  const int t_enc_per_chunk = mel_frames_per_chunk / cfg.subsampling_factor;

  auto enc_out_type = model_.session_info_.GetOutputDataType(cfg.enc_out_encoded);
  auto encoded_shape =
      std::array<int64_t, 3>{1, t_enc_per_chunk, cfg.hidden_dim};
  encoded_out_ = OrtValue::CreateTensor(inf_alloc, encoded_shape, enc_out_type);
  outputs_[encoded_output_idx_] = encoded_out_.get();

  // output_length: int64[1] on CPU so the host can read it directly after Run.
  auto length_out_type = model_.session_info_.GetOutputDataType(cfg.enc_out_length);
  auto length_out_shape = std::array<int64_t, 1>{1};
  output_length_ = OrtValue::CreateTensor(cpu_alloc, length_out_shape, length_out_type);
  outputs_[length_output_idx_] = output_length_.get();

  // Cache "next" buffers: same shapes as the input caches, allocated on the
  // same device. After each Run we swap them with the input cache buffers
  // (ping-pong) so we never round-trip through CPU.
  auto ch_type = model_.session_info_.GetOutputDataType(cfg.enc_out_cache_channel);
  auto ch_shape = std::array<int64_t, 4>{1, cfg.num_encoder_layers, cfg.left_context, cfg.hidden_dim};
  cache_last_channel_next_ = OrtValue::CreateTensor(inf_alloc, ch_shape, ch_type);
  outputs_[cache_channel_output_idx_] = cache_last_channel_next_.get();

  auto tm_type = model_.session_info_.GetOutputDataType(cfg.enc_out_cache_time);
  auto tm_shape = std::array<int64_t, 4>{1, cfg.num_encoder_layers, cfg.hidden_dim, cfg.conv_context};
  cache_last_time_next_ = OrtValue::CreateTensor(inf_alloc, tm_shape, tm_type);
  outputs_[cache_time_output_idx_] = cache_last_time_next_.get();

  // cache_last_channel_len_next: int64[1] on CPU (matches the input cache
  // length tensor placement; ORT will copy across if the session needs it
  // on the device).
  auto chlen_type = model_.session_info_.GetOutputDataType(cfg.enc_out_cache_channel_len);
  auto chlen_shape = std::array<int64_t, 1>{1};
  cache_last_channel_len_next_ = OrtValue::CreateTensor(cpu_alloc, chlen_shape, chlen_type);
  outputs_[cache_channel_len_output_idx_] = cache_last_channel_len_next_.get();

  // Set run options from config
  if (model_.config_->model.encoder.run_options.has_value()) {
    State::SetRunOptions(model_.config_->model.encoder.run_options.value());
  }
}

void NemotronEncoderSubState::SetMelInput(OrtValue* mel_tensor, int64_t total_mel_frames) {
  inputs_[mel_input_idx_] = mel_tensor;
  if (has_length_input_) {
    *signal_length_->GetTensorMutableData<int64_t>() = total_mel_frames;
  }
}

void NemotronEncoderSubState::UpdateCacheInputs() {
  inputs_[cache_channel_input_idx_] = cache_.cache_last_channel.get();
  inputs_[cache_time_input_idx_] = cache_.cache_last_time.get();
  inputs_[cache_channel_len_input_idx_] = cache_.cache_last_channel_len.get();
}

void NemotronEncoderSubState::UpdateOutputs() {
  outputs_[encoded_output_idx_] = encoded_out_.get();
  outputs_[length_output_idx_] = output_length_.get();
  outputs_[cache_channel_output_idx_] = cache_last_channel_next_.get();
  outputs_[cache_time_output_idx_] = cache_last_time_next_.get();
  outputs_[cache_channel_len_output_idx_] = cache_last_channel_len_next_.get();
}

void NemotronEncoderSubState::RotateCaches() {
  // After Run, the freshly-computed cache lives in *_next_. Swap with the
  // input cache buffers so the next Run sees it as input, and re-register
  // both input and output pointers (the old input buffer becomes the new
  // "next" output target).
  std::swap(cache_.cache_last_channel, cache_last_channel_next_);
  std::swap(cache_.cache_last_time, cache_last_time_next_);
  std::swap(cache_.cache_last_channel_len, cache_last_channel_len_next_);
  UpdateCacheInputs();
  UpdateOutputs();
}

void NemotronEncoderSubState::SetLangId(int lang_id) {
  if (!has_lang_id_input_ || !lang_id_tensor_) return;
  *lang_id_tensor_->GetTensorMutableData<int64_t>() = static_cast<int64_t>(lang_id);
}

DeviceSpan<float> NemotronEncoderSubState::Run(int /*total_length*/, DeviceSpan<int32_t>& /*next_tokens*/, DeviceSpan<int32_t> /*next_indices*/) {
  State::Run(*model_.session_encoder_);
  return {};
}

NemotronPredictionSubState::NemotronPredictionSubState(const NemotronSpeechModel& model, const GeneratorParams& params)
    : State{params, model},
      model_{model} {
  auto& cfg = model_.nemotron_config_;
  auto& cpu_dev = *GetDeviceInterface(DeviceType::CPU);
  auto& cpu_alloc = cpu_dev.GetAllocator();
  auto& inf_dev = *model_.p_device_inputs_;
  auto& inf_alloc = inf_dev.GetAllocator();

  lstm_state_.Initialize(cfg, model_.session_info_, cpu_alloc, cpu_dev);

  auto targets_type = model_.session_info_.GetInputDataType(cfg.dec_in_targets);
  auto targets_shape = std::array<int64_t, 2>{1, 1};
  targets_ = OrtValue::CreateTensor(cpu_alloc, targets_shape, targets_type);

  // Register inputs
  targets_input_idx_ = inputs_.size();
  input_names_.push_back(cfg.dec_in_targets.c_str());
  inputs_.push_back(targets_.get());

  lstm_hidden_input_idx_ = inputs_.size();
  input_names_.push_back(cfg.dec_in_lstm_hidden.c_str());
  inputs_.push_back(lstm_state_.lstm_hidden_state.get());

  lstm_cell_input_idx_ = inputs_.size();
  input_names_.push_back(cfg.dec_in_lstm_cell.c_str());
  inputs_.push_back(lstm_state_.lstm_cell_state.get());

  // Register outputs: outputs, lstm_hidden, lstm_cell
  output_names_.push_back(cfg.dec_out_outputs.c_str());
  outputs_.push_back(nullptr);
  dec_output_idx_ = outputs_.size() - 1;

  output_names_.push_back(cfg.dec_out_lstm_hidden.c_str());
  outputs_.push_back(nullptr);
  lstm_hidden_output_idx_ = outputs_.size() - 1;

  output_names_.push_back(cfg.dec_out_lstm_cell.c_str());
  outputs_.push_back(nullptr);
  lstm_cell_output_idx_ = outputs_.size() - 1;

  // Pre-allocate prediction-net outputs so ORT writes them directly to the
  // correct device. `dec_output_` is [1, lstm_dim] and lives on the inference
  // device (it's consumed by the joiner). The two LSTM-state "next" buffers
  // mirror the input lstm_state_ buffers and form a ping-pong pair.
  auto dec_out_type = model_.session_info_.GetOutputDataType(cfg.dec_out_outputs);
  // decoder_output shape is [B=1, D=lstm_dim, T=1]
  auto dec_out_shape = std::array<int64_t, 3>{1, cfg.decoder_lstm_dim, 1};
  dec_output_ = OrtValue::CreateTensor(inf_alloc, dec_out_shape, dec_out_type);
  outputs_[dec_output_idx_] = dec_output_.get();

  auto lstm_state_shape = std::array<int64_t, 3>{cfg.decoder_lstm_layers, 1, cfg.decoder_lstm_dim};
  auto lstm_hidden_type = model_.session_info_.GetOutputDataType(cfg.dec_out_lstm_hidden);
  lstm_hidden_next_ = OrtValue::CreateTensor(cpu_alloc, lstm_state_shape, lstm_hidden_type);
  outputs_[lstm_hidden_output_idx_] = lstm_hidden_next_.get();

  auto lstm_cell_type = model_.session_info_.GetOutputDataType(cfg.dec_out_lstm_cell);
  lstm_cell_next_ = OrtValue::CreateTensor(cpu_alloc, lstm_state_shape, lstm_cell_type);
  outputs_[lstm_cell_output_idx_] = lstm_cell_next_.get();

  // Set run options from config
  if (model_.config_->model.decoder.run_options.has_value()) {
    State::SetRunOptions(model_.config_->model.decoder.run_options.value());
  }
}

void NemotronPredictionSubState::UpdateInputs() {
  *targets_->GetTensorMutableData<int64_t>() = lstm_state_.last_token;
  inputs_[lstm_hidden_input_idx_] = lstm_state_.lstm_hidden_state.get();
  inputs_[lstm_cell_input_idx_] = lstm_state_.lstm_cell_state.get();
}

void NemotronPredictionSubState::UpdateOutputs() {
  outputs_[dec_output_idx_] = dec_output_.get();
  outputs_[lstm_hidden_output_idx_] = lstm_hidden_next_.get();
  outputs_[lstm_cell_output_idx_] = lstm_cell_next_.get();
}

void NemotronPredictionSubState::RotateLstmState() {
  // The freshly-computed LSTM state lives in *_next_. Swap with the input
  // state so the next Run consumes it; the old input buffers become the new
  // "next" output targets (their stale content will be overwritten).
  std::swap(lstm_state_.lstm_hidden_state, lstm_hidden_next_);
  std::swap(lstm_state_.lstm_cell_state, lstm_cell_next_);
  inputs_[lstm_hidden_input_idx_] = lstm_state_.lstm_hidden_state.get();
  inputs_[lstm_cell_input_idx_] = lstm_state_.lstm_cell_state.get();
  UpdateOutputs();
}

DeviceSpan<float> NemotronPredictionSubState::Run(int /*total_length*/, DeviceSpan<int32_t>& /*next_tokens*/, DeviceSpan<int32_t> /*next_indices*/) {
  State::Run(*model_.session_decoder_);
  return {};
}

NemotronJoinerSubState::NemotronJoinerSubState(const NemotronSpeechModel& model, const GeneratorParams& params)
    : State{params, model},
      model_{model} {
  auto& cfg = model_.nemotron_config_;
  auto& inf_dev = *model_.p_device_inputs_;
  auto& inf_alloc = inf_dev.GetAllocator();

  // Register inputs
  encoder_input_idx_ = inputs_.size();
  input_names_.push_back(cfg.join_in_encoder.c_str());
  inputs_.push_back(nullptr);  // Set before each run

  decoder_input_idx_ = inputs_.size();
  input_names_.push_back(cfg.join_in_decoder.c_str());
  inputs_.push_back(nullptr);  // Set before each run

  // Register output
  output_names_.push_back(cfg.join_out_logits.c_str());
  outputs_.push_back(nullptr);
  logits_output_idx_ = outputs_.size() - 1;

  // Pre-allocate logits output on the inference device. RNNT joiner shape is
  // [B=1, T=1, U=1, vocab_size].
  auto logits_type = model_.session_info_.GetOutputDataType(cfg.join_out_logits);
  auto logits_shape = std::array<int64_t, 4>{1, 1, 1, cfg.vocab_size};
  logits_ = OrtValue::CreateTensor(inf_alloc, logits_shape, logits_type);
  outputs_[logits_output_idx_] = logits_.get();

  // Set run options from config
  if (model_.config_->model.joiner.run_options.has_value()) {
    State::SetRunOptions(model_.config_->model.joiner.run_options.value());
  }
}

void NemotronJoinerSubState::SetInputFrames(OrtValue* encoder_frame, OrtValue* decoder_frame) {
  inputs_[encoder_input_idx_] = encoder_frame;
  inputs_[decoder_input_idx_] = decoder_frame;
}

DeviceSpan<float> NemotronJoinerSubState::Run(int /*total_length*/, DeviceSpan<int32_t>& /*next_tokens*/, DeviceSpan<int32_t> /*next_indices*/) {
  State::Run(*model_.session_joiner_);
  return {};
}

NemotronSpeechState::NemotronSpeechState(const NemotronSpeechModel& model,
                                         const GeneratorParams& params)
    : State{params, model},
      nemotron_model_{model} {
  nemotron_config_ = model.nemotron_config_;

  encoder_state_ = std::make_unique<NemotronEncoderSubState>(model, params);
  prediction_state_ = std::make_unique<NemotronPredictionSubState>(model, params);
  joiner_state_ = std::make_unique<NemotronJoinerSubState>(model, params);

  // Pre-allocate encoder frame for joiner input
  auto enc_out_type = model_.session_info_.GetOutputDataType(nemotron_config_.enc_out_encoded);
  auto frame_shape = std::array<int64_t, 3>{1, 1, nemotron_config_.hidden_dim};
  encoder_frame_ = OrtValue::CreateTensor(model_.allocator_cpu_, frame_shape, enc_out_type);
}

NemotronSpeechState::~NemotronSpeechState() = default;

DeviceSpan<float> NemotronSpeechState::Run(int /*total_length*/,
                                           DeviceSpan<int32_t>& /*next_tokens*/,
                                           DeviceSpan<int32_t> /*next_indices*/) {
  throw std::runtime_error(
      "NemotronSpeechState::Run() is not used directly. "
      "Use Generator::GenerateNextToken() with set_inputs.");
}

void NemotronSpeechState::SetExtraInputs(const std::vector<ExtraInput>& extra_inputs) {
  for (const auto& input : extra_inputs) {
    if (input.name == Config::Defaults::AudioFeaturesName || input.name == nemotron_config_.enc_in_audio) {
      current_mel_ = input.tensor;
      need_encoder_run_ = true;
      chunk_done_ = false;
    }
  }
}

OrtValue* NemotronSpeechState::GetInput(const char* name) {
  if (auto* val = encoder_state_->GetInput(name)) return val;
  if (auto* val = prediction_state_->GetInput(name)) return val;
  if (auto* val = joiner_state_->GetInput(name)) return val;
  return State::GetInput(name);
}

OrtValue* NemotronSpeechState::GetOutput(const char* name) {
  if (auto* val = encoder_state_->GetOutput(name)) return val;
  if (auto* val = prediction_state_->GetOutput(name)) return val;
  if (auto* val = joiner_state_->GetOutput(name)) return val;
  return State::GetOutput(name);
}

void NemotronSpeechState::SetLangId(int lang_id) {
  if (!encoder_state_->HasLangIdInput()) {
    throw std::runtime_error(
        "Cannot set lang_id: encoder graph has no lang_id input "
        "(this model is not a prompt-conditioned multilingual model).");
  }
  encoder_state_->SetLangId(lang_id);
}

void NemotronSpeechState::ResetStreamingState() {
  auto& cpu_dev = *GetDeviceInterface(DeviceType::CPU);
  auto& cpu_alloc = cpu_dev.GetAllocator();

  auto& inf_dev = *model_.p_device_inputs_;
  auto& inf_alloc = inf_dev.GetAllocator();
  encoder_state_->cache_.Reset(nemotron_config_, model_.session_info_, inf_alloc, inf_dev);
  encoder_state_->UpdateCacheInputs();
  encoder_state_->first_run_ = true;

  prediction_state_->lstm_state_.Reset(nemotron_config_, model_.session_info_, cpu_alloc, cpu_dev);
  prediction_state_->UpdateInputs();
  prediction_state_->first_run_ = true;

  joiner_state_->first_run_ = true;

  current_mel_.reset();
  encoded_len_ = 0;
  time_step_ = 0;
  symbol_step_ = 0;
  need_encoder_run_ = false;
  chunk_done_ = true;
  last_tokens_.clear();
}

void NemotronSpeechState::RunEncoder() {
  if (!current_mel_ || !current_mel_->ort_tensor_)
    throw std::runtime_error("No mel input set. Call generator.set_model_input(\"audio_features\", mel) first.");

  OrtValue* mel_tensor = current_mel_->ort_tensor_.get();
  int64_t total_mel_frames = mel_tensor->GetTensorTypeAndShapeInfo()->GetShape()[1];

  encoder_state_->SetMelInput(mel_tensor, total_mel_frames);
  encoder_state_->UpdateCacheInputs();
  encoder_state_->UpdateOutputs();

  DeviceSpan<int32_t> dummy_tokens;
  encoder_state_->Run(0, dummy_tokens);

  // All encoder outputs are pre-allocated in encoder_state_; ORT wrote
  // results directly into them. Read length from the CPU output_length_
  // buffer, then rotate cache buffers so the next Run consumes the newly
  // produced cache as input.
  encoded_len_ = encoder_state_->EncodedLength();
  encoder_state_->RotateCaches();

  current_mel_.reset();
}

std::span<const int32_t> NemotronSpeechState::StepToken() {
  if (need_encoder_run_) {
    RunEncoder();
    need_encoder_run_ = false;
    time_step_ = 0;
    symbol_step_ = 0;
  }

  last_tokens_.clear();

  OrtValue* encoded = encoder_state_->EncodedOutput();
  auto enc_shape = encoded->GetTensorTypeAndShapeInfo()->GetShape();
  int64_t time_steps = std::min(enc_shape[1], encoded_len_);
  int64_t hidden_dim = enc_shape[2];
  size_t frame_bytes = static_cast<size_t>(hidden_dim) * sizeof(float);

  auto& cpu_dev = *GetDeviceInterface(DeviceType::CPU);
  auto& cpu_alloc_step = cpu_dev.GetAllocator();
  auto& inf_dev = *model_.p_device_inputs_;

  // All session outputs (encoded, prediction net, joiner) are pre-allocated
  // on `inf_dev` at startup, so we know the source device for every copy
  // and no DeviceFor probing is needed.
  auto enc_span = ByteWrapTensor(inf_dev, *encoded);
  auto frame_span = ByteWrapTensor(cpu_dev, *encoder_frame_);

  // Pre-allocate the decoder frame ([1, 1, lstm_dim]) and a CPU mirror of
  // the joiner logits ([1, 1, 1, vocab_size]) once per StepToken call so we
  // don't re-allocate inside the inner loop.
  auto dec_out_type = model_.session_info_.GetOutputDataType(nemotron_config_.dec_out_outputs);
  auto decoder_frame_shape = std::array<int64_t, 3>{1, 1, nemotron_config_.decoder_lstm_dim};
  auto decoder_frame = OrtValue::CreateTensor(cpu_alloc_step, decoder_frame_shape, dec_out_type);

  OrtValue* logits_dev = joiner_state_->LogitsOutput();
  auto logits_shape = logits_dev->GetTensorTypeAndShapeInfo()->GetShape();
  auto logits_type = logits_dev->GetTensorTypeAndShapeInfo()->GetElementType();
  auto logits_cpu = OrtValue::CreateTensor(cpu_alloc_step, logits_shape, logits_type);
  int total_logits = 1;
  for (auto d : logits_shape) total_logits *= static_cast<int>(d);

  OrtValue* dec_output = prediction_state_->DecoderOutput();

  DeviceSpan<int32_t> dummy_tokens;

  while (time_step_ < time_steps) {
    // Copy current encoder frame
    auto src_frame = enc_span.subspan(static_cast<size_t>(time_step_) * frame_bytes, frame_bytes);
    frame_span.CopyFrom(src_frame);

    // Run prediction network. Output lands in prediction_state_->dec_output_
    // (inf_dev) and lstm_*_next_ buffers; no ownership transfer needed.
    prediction_state_->UpdateInputs();
    prediction_state_->Run(0, dummy_tokens);

    // Copy decoder hidden vector into [1, 1, lstm_dim] joiner-input shape.
    ByteWrapTensor(cpu_dev, *decoder_frame)
        .CopyFrom(ByteWrapTensor(inf_dev, *dec_output));

    // Run joiner. Logits land in pre-allocated joiner_state_->logits_ on inf_dev.
    joiner_state_->SetInputFrames(encoder_frame_.get(), decoder_frame.get());
    joiner_state_->Run(0, dummy_tokens);

    // Argmax over logits on CPU.
    ByteWrapTensor(cpu_dev, *logits_cpu)
        .CopyFrom(ByteWrapTensor(inf_dev, *logits_dev));
    const float* logits = logits_cpu->GetTensorData<float>();

    // Apply blank penalty virtually during argmax to avoid mutating ORT output buffer
    int best_token = 0;
    float best_score = logits[0] - (nemotron_config_.blank_id == 0 ? nemotron_config_.blank_penalty : 0.0f);
    for (int i = 1; i < total_logits; ++i) {
      float score = (i == nemotron_config_.blank_id) ? logits[i] - nemotron_config_.blank_penalty : logits[i];
      if (score > best_score) {
        best_score = score;
        best_token = i;
      }
    }

    if (best_token == nemotron_config_.blank_id) {
      // Blank: discard the new LSTM state (don't rotate); the next prediction
      // Run will overwrite the *_next_ buffers anyway.
      time_step_++;
      symbol_step_ = 0;
      continue;
    }

    // Non-blank: emit token, promote the new LSTM state.
    prediction_state_->lstm_state_.last_token = best_token;
    prediction_state_->RotateLstmState();

    symbol_step_++;
    if (symbol_step_ >= nemotron_config_.max_symbols_per_step) {
      time_step_++;
      symbol_step_ = 0;
    }

    last_tokens_.push_back(static_cast<int32_t>(best_token));
    token_count_++;
    return last_tokens_;
  }

  // Exhausted all time steps
  chunk_done_ = true;
  return last_tokens_;
}

}  // namespace Generators
