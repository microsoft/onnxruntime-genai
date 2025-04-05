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
    // input_ids_.name_ = "input_ids";
    // input_ids_.AddDecoderInputs();
    // input_ids_ = OrtValue::CreateTensor<int32_t>(model_.p_device_->GetAllocator(), std::array<int64_t, 2>{params_->search.batch_size, 1});
    // input_ids_index_ = inputs_.size();
    // input_names_.push_back("input_ids");
    // inputs_.push_back(input_ids_.get());
    decoder_input_ids_ = OrtValue::CreateTensor<int32_t>(model_.allocator_cpu_, std::array<int64_t, 2>{params_->search.batch_size, 1});
    // *decoder_input_ids_->GetTensorMutableData<int32_t>() = 0;
    auto* p_data = decoder_input_ids_->GetTensorMutableData<int32_t>();
    // auto* input_ids_data = decoder_input_ids_data;
    std::cout<<"FILLING DECODER INPUTS BEFORE "<<std::endl;
    for (int i = 0; i < params_->search.batch_size; i++) {
        std::cout<<"FILLING DECODER INPUTS = "<<i<<"   "<<p_data<<std::endl;
        std::cout<<"Batch size = "<<params_->search.batch_size<<std::endl;
        p_data[i] = 0;
      // p_data[i] = 0;
      // *p_data = 0;
    }
    // for (int i = 0; i < params_->BatchBeamSize(); i++) {
    //   for (int j = 0; j < 1; j++, input_ids_data++) {
    //     *input_ids_data = 0;
    //   }
    // }
    std::cout<<"FILLED DECODER INPUTS"<<std::endl;
    // expanded_decoder_input_ids_ = std::move(decoder_input_ids_);
    if (params_->search.num_beams == 1) {
      expanded_decoder_input_ids_ = std::move(decoder_input_ids_);
    } else {
      expanded_decoder_input_ids_ = model_.ExpandInputs(decoder_input_ids_, params_->search.num_beams);
    }
    input_names_.push_back("input_ids");
    inputs_.push_back(expanded_decoder_input_ids_.get());
    // inputs_.push_back(0);

    encoder_attention_mask_.Add();

    logits_.Add();
    kv_cache_.Add();

    AddDecoderCrossCache(cross_cache_);
    std::cout<<"FIRST RUN"<<std::endl;
    auto& stream = Log("INITIALIZED model_input_values");
    stream << std::endl;
    DumpTensors(model_, stream, inputs_.data(), input_names_.data(), input_names_.size(), true);
  }
    first_run_ = false;

    // UPDATE THE DECODER

    // input_ids_.Update(next_tokens);
    int batch_beam_size = static_cast<int>(next_tokens.size());
    std::unique_ptr<OrtValue> input_ids = OrtValue::CreateTensor<int32_t>(model_.allocator_cpu_, std::array<int64_t, 2>{params_->BatchBeamSize(), batch_beam_size});
    int32_t* input_ids_data = input_ids->GetTensorMutableData<int32_t>();
    auto next_tokens_value = next_tokens.CpuSpan();
    for (int i = 0; i < batch_beam_size; i++) {
      input_ids_data[i] = next_tokens_value[i];
    }
    expanded_decoder_input_ids_ = std::move(input_ids);
    inputs_[0] = expanded_decoder_input_ids_.get();

    // size_t new_length = static_cast<size_t>(input_ids_.GetShape()[1]);
    size_t new_length = expanded_decoder_input_ids_->GetTensorTypeAndShapeInfo()->GetShape()[1];
    encoder_attention_mask_.Update(next_tokens, current_length, static_cast<int>(new_length));

    kv_cache_.Update(next_indices, current_length);
    logits_.Update(next_tokens, new_length);

    // RUN THE DECODER
    State::Run(*model_.session_decoder_);
    std::cout<<"RUNNING DECODER"<<std::endl;
    return logits_.Get();

}

}  // namespace Generators
