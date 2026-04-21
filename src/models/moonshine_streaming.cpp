// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <cstring>
#include <stdexcept>
#include <vector>
#include <string>

#include "../generators.h"
#include "moonshine_streaming.h"

namespace Generators {

void MoonshineConfig::PopulateFromConfig(const Config& config) {
  const auto& enc = config.model.encoder;
  const auto& dec = config.model.decoder;

  encoder_hidden_size = enc.hidden_size;
  num_decoder_layers = dec.num_hidden_layers;
  num_attention_heads = dec.num_attention_heads;
  head_size = dec.head_size;
  vocab_size = config.model.vocab_size;

  sample_rate = config.model.sample_rate;
  chunk_samples = config.model.chunk_samples;

  bos_token_id = config.model.bos_token_id;
  eos_token_id = config.model.eos_token_id.empty() ? 2 : config.model.eos_token_id[0];

  // Encoder I/O names from config
  enc_in_audio = enc.inputs.audio_features;
  if (enc_in_audio.empty()) enc_in_audio = "input_values";
  enc_out_hidden_states = enc.outputs.hidden_states;
  if (enc_out_hidden_states.empty()) enc_out_hidden_states = "encoder_hidden_states";

  // Decoder I/O names
  dec_in_input_ids = dec.inputs.input_ids;
  if (dec_in_input_ids.empty()) dec_in_input_ids = "decoder_input_ids";
  dec_in_encoder_hidden_states = dec.inputs.encoder_hidden_states;
  if (dec_in_encoder_hidden_states.empty()) dec_in_encoder_hidden_states = "encoder_hidden_states";

  // Decoder with past filename
  if (!dec.pipeline.empty()) {
    dec_past_filename = dec.pipeline[0].filename;
  }
}

MoonshineStreamingModel::MoonshineStreamingModel(std::unique_ptr<Config> config, OrtEnv& ort_env)
    : Model{std::move(config)} {
  moonshine_config_ = MoonshineConfig{};
  moonshine_config_.PopulateFromConfig(*config_);

  // Create session options
  encoder_session_options_ = OrtSessionOptions::Create();
  decoder_session_options_ = OrtSessionOptions::Create();
  decoder_past_session_options_ = OrtSessionOptions::Create();

  if (config_->model.encoder.session_options.has_value()) {
    CreateSessionOptionsFromConfig(config_->model.encoder.session_options.value(),
                                   *encoder_session_options_, true);
  } else {
    CreateSessionOptionsFromConfig(config_->model.decoder.session_options,
                                   *encoder_session_options_, true);
  }
  CreateSessionOptionsFromConfig(config_->model.decoder.session_options,
                                 *decoder_session_options_, true);
  CreateSessionOptionsFromConfig(config_->model.decoder.session_options,
                                 *decoder_past_session_options_, true);

  // Load sessions
  std::string encoder_filename = config_->model.encoder.filename;
  if (encoder_filename.empty()) encoder_filename = "encoder_model_int8.onnx";

  std::string decoder_filename = config_->model.decoder.filename;
  if (decoder_filename.empty()) decoder_filename = "decoder_model_int8.onnx";

  std::string decoder_past_filename = moonshine_config_.dec_past_filename;
  if (decoder_past_filename.empty()) decoder_past_filename = "decoder_with_past_model_int8.onnx";

  session_encoder_ = CreateSession(ort_env, encoder_filename, encoder_session_options_.get());
  session_decoder_ = CreateSession(ort_env, decoder_filename, decoder_session_options_.get());
  session_decoder_past_ = CreateSession(ort_env, decoder_past_filename, decoder_past_session_options_.get());

  session_info_.Add(*session_encoder_);
  session_info_.Add(*session_decoder_);
  session_info_.Add(*session_decoder_past_);
}

std::unique_ptr<State> MoonshineStreamingModel::CreateState(DeviceSpan<int32_t> /*sequence_lengths*/,
                                                            const GeneratorParams& params) const {
  return std::make_unique<MoonshineStreamingState>(*this, params);
}

MoonshineStreamingState::MoonshineStreamingState(const MoonshineStreamingModel& model,
                                                 const GeneratorParams& params)
    : State{params, model},
      moonshine_model_{model} {
  config_ = model.moonshine_config_;

  // Pre-allocate decoder_input_ids
  auto ids_shape = std::array<int64_t, 2>{1, 1};
  decoder_input_ids_ = OrtValue::CreateTensor(model.allocator_cpu_, ids_shape,
                                              ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64);
  *decoder_input_ids_->GetTensorMutableData<int64_t>() = config_.bos_token_id;
}

MoonshineStreamingState::~MoonshineStreamingState() = default;

DeviceSpan<float> MoonshineStreamingState::Run(int /*total_length*/,
                                               DeviceSpan<int32_t>& /*next_tokens*/,
                                               DeviceSpan<int32_t> /*next_indices*/) {
  // Not used for streaming ASR - StepToken drives execution
  return {};
}

void MoonshineStreamingState::SetExtraInputs(const std::vector<ExtraInput>& extra_inputs) {
  for (const auto& input : extra_inputs) {
    if (input.name == config_.enc_out_hidden_states ||
        input.name == "encoder_hidden_states") {
      encoder_hidden_states_tensor_ = input.tensor;
      decode_mode_ = true;
      chunk_done_ = false;
      first_decode_ = true;
      last_token_ = config_.bos_token_id;

      // Reset KV caches
      self_kv_cache_.clear();
      cross_kv_cache_.clear();

      // Reset decoder_input_ids to BOS
      *decoder_input_ids_->GetTensorMutableData<int64_t>() = config_.bos_token_id;
      break;
    }
  }
}

std::span<const int32_t> MoonshineStreamingState::StepToken() {
  last_tokens_.clear();

  if (!decode_mode_ || !encoder_hidden_states_tensor_) {
    chunk_done_ = true;
    return last_tokens_;
  }

  if (first_decode_) {
    RunInitialDecoder();
    first_decode_ = false;
  } else {
    RunDecoderWithPast();
  }

  // Extract token from logits (greedy argmax)
  auto logits_data = logits_->GetTensorData<float>();
  auto logits_shape = logits_->GetTensorTypeAndShapeInfo()->GetShape();
  int64_t vocab = logits_shape.back();

  float max_val = logits_data[0];
  int max_idx = 0;
  for (int i = 1; i < vocab; ++i) {
    if (logits_data[i] > max_val) {
      max_val = logits_data[i];
      max_idx = i;
    }
  }

  last_token_ = max_idx;

  if (last_token_ == config_.eos_token_id) {
    chunk_done_ = true;
    decode_mode_ = false;
  } else {
    last_tokens_.push_back(static_cast<int32_t>(last_token_));
    token_count_++;

    // Update decoder_input_ids for next step
    *decoder_input_ids_->GetTensorMutableData<int64_t>() = last_token_;
  }

  return last_tokens_;
}

void MoonshineStreamingState::RunInitialDecoder() {
  auto* enc_ort = encoder_hidden_states_tensor_->ort_tensor_.get();

  // Build input arrays
  std::vector<const char*> input_names = {
      config_.dec_in_input_ids.c_str(),
      config_.dec_in_encoder_hidden_states.c_str()};
  std::vector<OrtValue*> input_values = {
      decoder_input_ids_.get(),
      enc_ort};

  // Build output names: logits + self KV (present_self_key/value_0..N-1) + cross KV (present_cross_key/value_0..N-1)
  std::vector<std::string> output_name_strings;
  output_name_strings.push_back("logits");
  for (int i = 0; i < config_.num_decoder_layers; ++i) {
    output_name_strings.push_back("present_self_key_" + std::to_string(i));
    output_name_strings.push_back("present_self_value_" + std::to_string(i));
  }
  for (int i = 0; i < config_.num_decoder_layers; ++i) {
    output_name_strings.push_back("present_cross_key_" + std::to_string(i));
    output_name_strings.push_back("present_cross_value_" + std::to_string(i));
  }

  std::vector<const char*> output_names;
  for (const auto& s : output_name_strings) output_names.push_back(s.c_str());

  // Allocate output values
  size_t num_outputs = output_names.size();
  std::vector<OrtValue*> output_values(num_outputs, nullptr);

  // Run initial decoder
  moonshine_model_.session_decoder_->Run(
      nullptr,
      input_names.data(), input_values.data(), input_names.size(),
      output_names.data(), output_values.data(), num_outputs);

  // Store outputs
  logits_ = std::unique_ptr<OrtValue>(output_values[0]);

  self_kv_cache_.clear();
  cross_kv_cache_.clear();

  // Self KV: indices [1, 2*num_decoder_layers]
  for (int i = 0; i < config_.num_decoder_layers * 2; ++i) {
    self_kv_cache_.push_back(std::unique_ptr<OrtValue>(output_values[1 + i]));
  }
  // Cross KV: indices [1 + 2*num_decoder_layers, ...]
  int cross_start = 1 + config_.num_decoder_layers * 2;
  for (int i = 0; i < config_.num_decoder_layers * 2; ++i) {
    cross_kv_cache_.push_back(std::unique_ptr<OrtValue>(output_values[cross_start + i]));
  }
}

void MoonshineStreamingState::RunDecoderWithPast() {
  auto* enc_ort = encoder_hidden_states_tensor_->ort_tensor_.get();

  // Build input names and values
  std::vector<std::string> input_name_strings;
  std::vector<OrtValue*> input_values;

  // decoder_input_ids
  input_name_strings.push_back(config_.dec_in_input_ids);
  input_values.push_back(decoder_input_ids_.get());

  // encoder_hidden_states
  input_name_strings.push_back(config_.dec_in_encoder_hidden_states);
  input_values.push_back(enc_ort);

  // Self KV past: past_self_key_0, past_self_value_0, ...
  for (int i = 0; i < config_.num_decoder_layers; ++i) {
    input_name_strings.push_back("past_self_key_" + std::to_string(i));
    input_values.push_back(self_kv_cache_[i * 2].get());
    input_name_strings.push_back("past_self_value_" + std::to_string(i));
    input_values.push_back(self_kv_cache_[i * 2 + 1].get());
  }

  // Cross KV: present_cross_key_0_orig, present_cross_value_0_orig, ...
  for (int i = 0; i < config_.num_decoder_layers; ++i) {
    input_name_strings.push_back("present_cross_key_" + std::to_string(i) + "_orig");
    input_values.push_back(cross_kv_cache_[i * 2].get());
    input_name_strings.push_back("present_cross_value_" + std::to_string(i) + "_orig");
    input_values.push_back(cross_kv_cache_[i * 2 + 1].get());
  }

  std::vector<const char*> input_names;
  for (const auto& s : input_name_strings) input_names.push_back(s.c_str());

  // Build output names: logits + self KV + cross KV
  std::vector<std::string> output_name_strings;
  output_name_strings.push_back("logits");
  for (int i = 0; i < config_.num_decoder_layers; ++i) {
    output_name_strings.push_back("present_self_key_" + std::to_string(i));
    output_name_strings.push_back("present_self_value_" + std::to_string(i));
  }
  for (int i = 0; i < config_.num_decoder_layers; ++i) {
    output_name_strings.push_back("present_cross_key_" + std::to_string(i));
    output_name_strings.push_back("present_cross_value_" + std::to_string(i));
  }

  std::vector<const char*> output_names;
  for (const auto& s : output_name_strings) output_names.push_back(s.c_str());

  size_t num_outputs = output_names.size();
  std::vector<OrtValue*> output_values(num_outputs, nullptr);

  // Run decoder with past
  moonshine_model_.session_decoder_past_->Run(
      nullptr,
      input_names.data(), input_values.data(), input_names.size(),
      output_names.data(), output_values.data(), num_outputs);

  // Store outputs
  logits_ = std::unique_ptr<OrtValue>(output_values[0]);

  // Update self KV (grown by 1)
  for (int i = 0; i < config_.num_decoder_layers * 2; ++i) {
    self_kv_cache_[i] = std::unique_ptr<OrtValue>(output_values[1 + i]);
  }
  // Cross KV is pass-through but we take ownership of the new outputs
  int cross_start = 1 + config_.num_decoder_layers * 2;
  for (int i = 0; i < config_.num_decoder_layers * 2; ++i) {
    cross_kv_cache_[i] = std::unique_ptr<OrtValue>(output_values[cross_start + i]);
  }
}

OrtValue* MoonshineStreamingState::GetInput(const char* /*name*/) {
  return nullptr;
}

OrtValue* MoonshineStreamingState::GetOutput(const char* /*name*/) {
  return nullptr;
}

}  // namespace Generators
