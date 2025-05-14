// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#include "../generators.h"
#include "marian.h"
#include <vector>
#include "../sequences.h"

namespace Generators {

MarianModel::MarianModel(std::unique_ptr<Config> config, OrtEnv& ort_env)
    : Model{std::move(config)} {
  session_encoder_ = OrtSession::Create(ort_env, (config_->config_path / fs::path(config_->model.encoder.filename)).c_str(), session_options_.get());
  session_decoder_ = OrtSession::Create(ort_env, (config_->config_path / fs::path(config_->model.decoder.filename)).c_str(), session_options_.get());

  // InitDeviceAllocator(*session_decoder_);
  session_info_.Add(*session_decoder_);
  session_info_.Add(*session_encoder_);
}

std::unique_ptr<State> MarianModel::CreateState(DeviceSpan<int32_t> sequence_lengths, const GeneratorParams& params) const {
  return std::make_unique<MarianState>(*this, sequence_lengths, params);
}

MarianState::MarianState(const MarianModel& model, DeviceSpan<int32_t> sequence_lengths_unk, const GeneratorParams& params)
    : State{params, model},
      model_{model},
      encoder_attention_mask_{model, *this, sequence_lengths_unk},
      attention_mask_{model, *this, sequence_lengths_unk} {
}

DeviceSpan<float> MarianState::Run(int current_length, DeviceSpan<int32_t>& next_tokens, DeviceSpan<int32_t> next_indices) {
  if (first_run_) {
    // INITIALIZE THE ENCODER AND RUN IT ONCE

    encoder_input_ids_.name_ = "input_ids";
    encoder_input_ids_.Add();

    encoder_attention_mask_.name_ = model_.config_->model.encoder.inputs.attention_mask;
    encoder_attention_mask_.Add();

    encoder_input_ids_.Update(next_tokens);
    size_t new_length = static_cast<size_t>(encoder_input_ids_.GetShape()[1]);
    encoder_attention_mask_.Update(next_tokens, current_length, static_cast<int>(new_length));

    auto encoder_outputs_type = model_.session_info_.GetOutputDataType("encoder_outputs");
    auto encoder_outputs_shape = std::array<int64_t, 3>{encoder_input_ids_.GetShape()[0], encoder_input_ids_.GetShape()[1], 512};
    encoder_outputs_ = OrtValue::CreateTensor(model_.p_device_->GetAllocator(), encoder_outputs_shape, encoder_outputs_type);

    output_names_.push_back("encoder_outputs");
    outputs_.push_back(encoder_outputs_.get());

    State::Run(*model_.session_encoder_);

    // CLEAR INPUTS AND OUTPUTS
    ClearIO();

    // INITIALIZE THE DECODER
    decoder_input_ids_.name_ = "input_ids";
    decoder_input_ids_.AddDecoderInputs(static_cast<int>(model_.config_->model.bos_token_id));

    const std::array<int64_t, 1> past_key_values_length_shape{1};
    past_key_values_length_ = OrtValue::CreateTensor(model_.allocator_cpu_, past_key_values_length_shape, model_.session_info_.GetInputDataType(model_.config_->model.decoder.inputs.past_key_values_length));
    input_names_.push_back(model_.config_->model.decoder.inputs.past_key_values_length.c_str());
    inputs_.push_back(past_key_values_length_.get());

    *past_key_values_length_->GetTensorMutableData<int64_t>() = -1;

    attention_mask_.name_ = model_.config_->model.decoder.inputs.encoder_attention_mask;
    attention_mask_.Add();
    attention_mask_.Update(next_tokens, current_length, static_cast<int>(new_length));

    auto hidden_states_type = model_.session_info_.GetInputDataType("encoder_hidden_states");
    auto encoder_hidden_states_shape = std::array<int64_t, 3>{decoder_input_ids_.GetDecoderInputShape()[0], encoder_input_ids_.GetShape()[1], 512};
    encoder_hidden_states_ = OrtValue::CreateTensor(model_.p_device_->GetAllocator(), encoder_hidden_states_shape, hidden_states_type);

    input_names_.push_back("encoder_hidden_states");
    inputs_.push_back(encoder_outputs_.get());

    auto rnn_states_prev_type = model_.session_info_.GetInputDataType("rnn_states_prev");
    auto rnn_states_prev_shape = std::array<int64_t, 3>{3, decoder_input_ids_.GetDecoderInputShape()[0], 512};
    rnn_states_prev_ = OrtValue::CreateTensor(model_.p_device_->GetAllocator(), rnn_states_prev_shape, rnn_states_prev_type);

    input_names_.push_back("rnn_states_prev");
    for (int i = 0; i < rnn_states_prev_shape[0]; i++) {
      for (int j = 0; j < rnn_states_prev_shape[1]; j++) {
        for (int k = 0; k < rnn_states_prev_shape[2]; k++) {
          auto data = rnn_states_prev_->GetTensorMutableData<int32_t>();
          data[i * rnn_states_prev_shape[1] * rnn_states_prev_shape[2] + j * rnn_states_prev_shape[2] + k] = 0;
        }
      }
    }
    inputs_.push_back(rnn_states_prev_.get());

    auto rnn_states_type = model_.session_info_.GetOutputDataType("rnn_states");
    auto rnn_states_shape = std::array<int64_t, 3>{3, decoder_input_ids_.GetDecoderInputShape()[0], 512};
    rnn_states_ = OrtValue::CreateTensor(model_.p_device_->GetAllocator(), rnn_states_shape, rnn_states_type);

    output_names_.push_back("rnn_states");
    outputs_.push_back(rnn_states_.get());
    *past_key_values_length_->GetTensorMutableData<int64_t>() += 1;

    logits_.Add();

    next_tokens.CpuSpan()[next_tokens.size() - 1] = 32000;

    logits_.Update(next_tokens.subspan(next_tokens.size() - 1, 1), 1);

    State::Run(*model_.session_decoder_);
    first_run_ = false;
    return logits_.Get();
  }

  // UPDATE THE DECODER
  decoder_input_ids_.Update(next_tokens);

  auto rnn_states_prev_shape = std::array<int64_t, 3>{3, decoder_input_ids_.GetDecoderInputShape()[0], 512};

  for (int i = 0; i < rnn_states_prev_shape[0]; i++) {
    for (int j = 0; j < rnn_states_prev_shape[1]; j++) {
      for (int k = 0; k < rnn_states_prev_shape[2]; k++) {
        auto data = rnn_states_prev_->GetTensorMutableData<int32_t>();
        auto rnn_states_data = rnn_states_->GetTensorMutableData<int32_t>();
        data[i * rnn_states_prev_shape[1] * rnn_states_prev_shape[2] + j * rnn_states_prev_shape[2] + k] = rnn_states_data[i * rnn_states_prev_shape[1] * rnn_states_prev_shape[2] + j * rnn_states_prev_shape[2] + k];
      }
    }
  }

  auto data = past_key_values_length_->GetTensorMutableData<int64_t>();
  *data += 1;

  logits_.Update(next_tokens, 1);

  // RUN THE DECODER
  State::Run(*model_.session_decoder_);
  return logits_.Get();
}

}  // namespace Generators