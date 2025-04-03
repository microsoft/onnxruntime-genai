// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#include "../generators.h"
#include "encoder_decoder.h"
#include <vector>
#include "../sequences.h"

namespace Generators {

EncoderDecoderModel::EncoderDecoderModel(std::unique_ptr<Config> config, OrtEnv& ort_env)
    : Model{std::move(config)} {
  session_encoder_ = OrtSession::Create(ort_env, (config_->config_path / fs::path(config_->model.encoder_decoder_init.filename)).c_str(), session_options_.get());
  session_decoder_ = OrtSession::Create(ort_env, (config_->config_path / fs::path(config_->model.decoder.filename)).c_str(), session_options_.get());

  InitDeviceAllocator(*session_decoder_);
  session_info_->Add(*session_encoder_);
}

std::unique_ptr<State> EncoderDecoderModel::CreateState(DeviceSpan<int32_t> sequence_lengths, const GeneratorParams& params) const {
  return std::make_unique<EncoderDecoderState>(*this, sequence_lengths, params);
}

EncoderDecoderState::EncoderDecoderState(const EncoderDecoderModel& model, DeviceSpan<int32_t> sequence_lengths_unk, const GeneratorParams& params)
    : State{params, model},
      model_{model},
      encoder_attention_mask_{model, *this, sequence_lengths_unk}  {
}

DeviceSpan<float> EncoderDecoderState::Run(int current_length, DeviceSpan<int32_t>& next_tokens, DeviceSpan<int32_t> next_indices) {
  if(first_run_) {
    //INITIALIZE THE ENCODER AND RUN IT ONCE

    encoder_input_ids_.name_ = "encoder_input_ids";
    encoder_input_ids_.Add();

    encoder_attention_mask_.Add();
    
    cross_cache_ = std::make_unique<CrossCache>(*this, next_tokens.size());
    AddEncoderCrossCache(cross_cache_);
  
    encoder_input_ids_.Update(next_tokens);
    size_t new_length = static_cast<size_t>(encoder_input_ids_.GetShape()[1]);
    encoder_attention_mask_.Update(next_tokens, current_length, static_cast<int>(new_length));
    State::Run(*model_.session_encoder_);

    // CLEAR INPUTS AND OUTPUTS
    input_names_.clear();
    output_names_.clear();
    inputs_.clear();
    outputs_.clear();

    //INITIALIZE THE DECODER
    input_ids_.name_ = "input_ids";
    input_ids_.Add();

    encoder_attention_mask_.Add();
  
    logits_.Add();
    kv_cache_.Add();
    cross_cache_ = std::make_unique<CrossCache>(*this, next_tokens.size());
    AddDecoderCrossCache(cross_cache_);
    std::cout<<"FIRST RUN"<<std::endl;
  }
    first_run_ = false;

    // UPDATE THE DECODER

    input_ids_.Update(next_tokens);
    size_t new_length = static_cast<size_t>(input_ids_.GetShape()[1]);
    encoder_attention_mask_.Update(next_tokens, current_length, static_cast<int>(new_length));
  
    kv_cache_.Update(next_indices, current_length);
    logits_.Update(next_tokens, new_length);

    // RUN THE DECODER
    State::Run(*model_.session_decoder_);
    return logits_.Get();

}

}  // namespace Generators
