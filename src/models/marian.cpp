// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#include "../generators.h"
#include "marian.h"

namespace Generators {

MarianModel::MarianModel(std::unique_ptr<Config> config, OrtEnv& ort_env)
    : Model{std::move(config)} {
  session_encoder_ = OrtSession::Create(ort_env, (config_->config_path / fs::path(config_->model.encoder.filename)).c_str(), session_options_.get());
  session_decoder_ = OrtSession::Create(ort_env, (config_->config_path / fs::path(config_->model.decoder.filename)).c_str(), session_options_.get());

  session_info_.Add(*session_decoder_);
  session_info_.Add(*session_encoder_);
}

MarianLogits::MarianLogits(State& state)
    : state_{state},
      shape_{static_cast<int64_t>(state_.params_->BatchBeamSize()), model_.config_->model.vocab_size},
      type_{model_.session_info_.GetOutputDataType(model_.config_->model.decoder.outputs.logits)} {
  output_raw_ = std::make_unique<Tensor>(model_.p_device_inputs_, type_);
}

std::unique_ptr<State> MarianModel::CreateState(DeviceSpan<int32_t> sequence_lengths, const GeneratorParams& params) const {
  return std::make_unique<MarianState>(*this, sequence_lengths, params);
}

MarianState::MarianState(const MarianModel& model, DeviceSpan<int32_t> sequence_lengths_unk, const GeneratorParams& params)
    : State{params, model},
      model_{model},
      encoder_attention_mask_{model, *this, sequence_lengths_unk, model_.config_->model.encoder.inputs.attention_mask},
      attention_mask_{model, *this, sequence_lengths_unk, model_.config_->model.decoder.inputs.encoder_attention_mask} {
}

DeviceSpan<float> MarianLogits::Get() {
  // The model's output logits are {batch_size*num_beams, vocab_size}

  // TODO(apsonawane): Fix the issue with output_raw not getting updated properly
  // OrtValue* logits_of_last_token = output_raw_->GetOrtTensor();
  OrtValue* logits_of_last_token = state_.outputs_[output_index_];

  // Convert from float16 to float32 if necessary
  if (type_ == Ort::TypeToTensorType<Ort::Float16_t>) {
    Cast(*logits_of_last_token, logits_of_last_token_fp32_, *model_.p_device_inputs_, Ort::TypeToTensorType<float>);
    logits_of_last_token = logits_of_last_token_fp32_.get();
  }

  if (logits_.empty() || logits_of_last_token->GetTensorMutableRawData() != logits_.Span().data())
    logits_ = WrapTensor<float>(*model_.p_device_inputs_, *logits_of_last_token);

  return logits_;
}

void MarianLogits::Update(const DeviceSpan<int32_t>& next_tokens, size_t new_kv_length) {
  if (output_raw_->ort_tensor_ && static_cast<size_t>(output_raw_->GetShape()[1]) == new_kv_length && new_kv_length == 1) {
    return;
  }

  output_raw_->CreateTensor(shape_, state_.params_->use_graph_capture);
  state_.outputs_[output_index_] = output_raw_->GetOrtTensor();
}

void MarianLogits::Add() {
  output_index_ = state_.outputs_.size();

  state_.output_names_.push_back(model_.config_->model.decoder.outputs.logits.c_str());
  state_.outputs_.push_back(output_raw_->GetOrtTensor());
}

MarianInputIDs::MarianInputIDs(State& state)
    : state_{state} {
  name_ = model_.config_->model.decoder.inputs.input_ids.c_str();
  shape_ = {state_.params_->BatchBeamSize()};
  type_ = model_.session_info_.GetInputDataType(name_);

  value_ = std::make_unique<Tensor>(model_.p_device_inputs_, Ort::TypeToTensorType<int32_t>);
  cast_value_ = std::make_unique<Tensor>(model_.p_device_inputs_, Ort::TypeToTensorType<int64_t>);
}

void MarianInputIDs::Add() {
  input_index_ = state_.inputs_.size();

  state_.inputs_.push_back(value_->GetOrtTensor());
  state_.input_names_.push_back(name_);
}

void MarianInputIDs::Update(DeviceSpan<int32_t> new_tokens) {
  value_->CreateTensor(shape_, state_.params_->use_graph_capture);
  state_.inputs_[input_index_] = value_->GetOrtTensor();

  // Update input_ids with next tokens
  auto data_span = value_->GetDeviceSpan<int32_t>();
  data_span.CopyFrom(new_tokens);

  if (type_ == Ort::TypeToTensorType<int64_t>) {
    if (!cast_value_->GetOrtTensor())
      cast_value_->CreateTensor(shape_, state_.params_->use_graph_capture);
    Cast(*value_->GetOrtTensor(), cast_value_->ort_tensor_, *model_.p_device_inputs_, type_);
    state_.inputs_[input_index_] = cast_value_->GetOrtTensor();
  }
}

DeviceSpan<float> MarianState::Run(int current_length, DeviceSpan<int32_t>& next_tokens, DeviceSpan<int32_t> next_indices) {
  const int64_t encoder_hidden_size = model_.config_->model.encoder.hidden_size;
  if (first_run_) {
    // Initialize the encoder and run it once
    encoder_input_ids_.name_ = model_.config_->model.encoder.inputs.input_ids.c_str();
    encoder_input_ids_.Add();

    // encoder_attention_mask_.attention_mask_name_ = model_.config_->model.encoder.inputs.attention_mask;
    encoder_attention_mask_.Add();

    encoder_input_ids_.Update(next_tokens);
    const size_t new_length = static_cast<size_t>(encoder_input_ids_.GetShape()[1]);
    encoder_attention_mask_.Update(next_tokens, current_length, static_cast<int>(new_length));

    const auto encoder_outputs_type = model_.session_info_.GetOutputDataType(model_.config_->model.encoder.outputs.encoder_outputs.c_str());
    const std::array<int64_t, 3> encoder_outputs_shape{encoder_input_ids_.GetShape()[0], encoder_input_ids_.GetShape()[1], encoder_hidden_size};
    encoder_outputs_ = OrtValue::CreateTensor(model_.p_device_inputs_->GetAllocator(), encoder_outputs_shape, encoder_outputs_type);

    output_names_.push_back(model_.config_->model.encoder.outputs.encoder_outputs.c_str());
    outputs_.push_back(encoder_outputs_.get());

    State::Run(*model_.session_encoder_);

    // Clear inputs and outputs for the decoder
    ClearIO();

    // Initialize the decoder inputs and outputs
    decoder_input_ids_.name_ = model_.config_->model.decoder.inputs.input_ids.c_str();
    decoder_input_ids_.Add();

    next_tokens.CpuSpan()[next_tokens.size() - 1] = model_.config_->model.bos_token_id;

    decoder_input_ids_.Update(next_tokens.subspan(next_tokens.size() - 1, 1));

    const std::array<int64_t, 1> past_key_values_length_shape{1};
    past_key_values_length_ = OrtValue::CreateTensor(model_.allocator_cpu_, past_key_values_length_shape, model_.session_info_.GetInputDataType(model_.config_->model.decoder.inputs.past_key_values_length));
    input_names_.push_back(model_.config_->model.decoder.inputs.past_key_values_length.c_str());
    inputs_.push_back(past_key_values_length_.get());

    *past_key_values_length_->GetTensorMutableData<int64_t>() = -1;

    // attention_mask_.attention_mask_name_ = model_.config_->model.decoder.inputs.encoder_attention_mask;
    attention_mask_.Add();
    attention_mask_.Update(next_tokens, current_length, static_cast<int>(new_length));

    const auto hidden_states_type = model_.session_info_.GetInputDataType(model_.config_->model.decoder.inputs.encoder_hidden_states.c_str());
    const std::array<int64_t, 3> encoder_hidden_states_shape{decoder_input_ids_.GetMarianInputsShape()[0], encoder_input_ids_.GetShape()[1], encoder_hidden_size};
    encoder_hidden_states_ = OrtValue::CreateTensor(model_.p_device_inputs_->GetAllocator(), encoder_hidden_states_shape, hidden_states_type);

    input_names_.push_back(model_.config_->model.decoder.inputs.encoder_hidden_states.c_str());
    inputs_.push_back(encoder_outputs_.get());

    const auto rnn_states_prev_type = model_.session_info_.GetInputDataType(model_.config_->model.decoder.inputs.rnn_prev_states.c_str());
    const std::array<int64_t, 3> rnn_states_prev_shape{3, decoder_input_ids_.GetMarianInputsShape()[0], encoder_hidden_size};
    rnn_states_prev_ = std::make_unique<Tensor>(model_.p_device_inputs_, rnn_states_prev_type);
    rnn_states_prev_->CreateTensor(rnn_states_prev_shape);

    input_names_.push_back(model_.config_->model.decoder.inputs.rnn_prev_states.c_str());

    auto device_span = rnn_states_prev_->GetDeviceSpan<int32_t>();
    device_span.Zero();
    inputs_.push_back(rnn_states_prev_->GetOrtTensor());

    const auto rnn_states_type = model_.session_info_.GetOutputDataType(model_.config_->model.decoder.outputs.rnn_states.c_str());
    const std::array<int64_t, 3> rnn_states_shape{3, decoder_input_ids_.GetMarianInputsShape()[0], encoder_hidden_size};
    rnn_states_ = std::make_unique<Tensor>(model_.p_device_inputs_, rnn_states_type);
    rnn_states_->CreateTensor(rnn_states_shape);

    output_names_.push_back(model_.config_->model.decoder.outputs.rnn_states.c_str());
    outputs_.push_back(rnn_states_->GetOrtTensor());
    *past_key_values_length_->GetTensorMutableData<int64_t>() += 1;

    logits_.Add();

    logits_.Update(next_tokens.subspan(next_tokens.size() - 1, 1), 1);

    State::Run(*model_.session_decoder_);
    first_run_ = false;
    return logits_.Get();
  }

  // Update the decoder inputs with the next tokens
  decoder_input_ids_.Update(next_tokens);
  const std::array<int64_t, 3> rnn_states_prev_shape{3, decoder_input_ids_.GetMarianInputsShape()[0], encoder_hidden_size};

  auto rnn_states_prev_data = rnn_states_prev_->GetDeviceSpan<int32_t>();
  auto rnn_states_data = rnn_states_->GetDeviceSpan<int32_t>();
  rnn_states_prev_data.CopyFrom(rnn_states_data);

  auto data = past_key_values_length_->GetTensorMutableData<int64_t>();
  *data += 1;

  logits_.Update(next_tokens, 1);

  // Run the decoder
  State::Run(*model_.session_decoder_);
  return logits_.Get();
}

}  // namespace Generators
