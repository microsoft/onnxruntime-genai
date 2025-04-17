// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "simple_decoder.h"

#include <typeinfo>

namespace Generators {

StaticBatchDecoderIO::StaticBatchDecoderIO(std::shared_ptr<DecoderOnly_Model> model,
                                           ScheduledRequests& scheduled_requests,
                                           std::shared_ptr<CacheManager> cache_manager)
    : ModelIO(*scheduled_requests.Params(), *model) {
  PrepareInputIds(model, scheduled_requests);
  PrepareAttentionMask(model, scheduled_requests);
  PreparePositionIds(model, scheduled_requests);
  PrepareLogits(model, scheduled_requests);
}

void StaticBatchDecoderIO::PrepareInputIds(std::shared_ptr<DecoderOnly_Model> model, ScheduledRequests& scheduled_requests) {
  auto request_with_max_sequence_length =
      std::max_element(
          scheduled_requests.Requests().begin(), scheduled_requests.Requests().end(),
          [](const std::shared_ptr<Request>& a, const std::shared_ptr<Request>& b) {
            return a->UnprocessedTokens().size() < b->UnprocessedTokens().size();
          });

  const int64_t max_sequence_length = (*request_with_max_sequence_length)->UnprocessedTokens().size();
  const int64_t batch_size = scheduled_requests.Requests().size();
  const std::vector<int64_t> input_ids_shape = {batch_size, max_sequence_length};
  auto input_ids_tensor = std::make_unique<Tensor>(model->p_device_inputs_, Ort::TypeToTensorType<int32_t>);
  auto device_span = input_ids_tensor->GetDeviceSpan<int32_t>();
  auto cpu_span = device_span.CpuSpan();

  for (size_t i = 0; i < batch_size; ++i) {
    auto request = scheduled_requests.Requests()[i];
    auto input_ids = request->UnprocessedTokens().CopyDeviceToCpu();
    for (size_t j = 0; j < max_sequence_length; ++j) {
      cpu_span[i * max_sequence_length + j] = (j < input_ids.size()) ? input_ids[j] : model->config_->model.pad_token_id;
    }
  }

  input_names_.push_back(model->config_->model.decoder.inputs.input_ids.c_str());
  inputs_.push_back(input_ids_tensor->GetOrtTensor());
  owned_inputs_.push_back(std::move(input_ids_tensor));
}

void StaticBatchDecoderIO::PrepareAttentionMask(std::shared_ptr<DecoderOnly_Model> model, ScheduledRequests& scheduled_requests) {
  auto request_with_max_sequence_length =
      std::max_element(
          scheduled_requests.Requests().begin(), scheduled_requests.Requests().end(),
          [](const std::shared_ptr<Request>& a, const std::shared_ptr<Request>& b) {
            return a->UnprocessedTokens().size() < b->UnprocessedTokens().size();
          });

  const int64_t max_sequence_length = (*request_with_max_sequence_length)->UnprocessedTokens().size();
  const int64_t batch_size = scheduled_requests.Requests().size();
  const std::vector<int64_t> attention_mask_shape = {batch_size, max_sequence_length};
  auto attention_mask_tensor = std::make_unique<Tensor>(model->p_device_inputs_, Ort::TypeToTensorType<int32_t>);
  auto device_span = attention_mask_tensor->GetDeviceSpan<int32_t>();
  auto cpu_span = device_span.CpuSpan();

  for (size_t i = 0; i < batch_size; ++i) {
    auto request = scheduled_requests.Requests()[i];
    auto input_ids = request->UnprocessedTokens().CopyDeviceToCpu();

    for (size_t j = 0; j < max_sequence_length; ++j) {
      cpu_span[i * max_sequence_length + j] = (j < input_ids.size() && input_ids[j] != model->config_->model.pad_token_id) ? 1 : 0;
    }
  }

  device_span.CopyCpuToDevice();

  input_names_.push_back(model->config_->model.decoder.inputs.attention_mask.c_str());
  inputs_.push_back(attention_mask_tensor->GetOrtTensor());
  owned_inputs_.push_back(std::move(attention_mask_tensor));
}

void StaticBatchDecoderIO::PreparePositionIds(std::shared_ptr<DecoderOnly_Model> model, ScheduledRequests& scheduled_requests) {
  auto request_with_max_sequence_length =
      std::max_element(
          scheduled_requests.Requests().begin(), scheduled_requests.Requests().end(),
          [](const std::shared_ptr<Request>& a, const std::shared_ptr<Request>& b) {
            return a->UnprocessedTokens().size() < b->UnprocessedTokens().size();
          });

  const int64_t max_sequence_length = (*request_with_max_sequence_length)->UnprocessedTokens().size();
  const int64_t batch_size = scheduled_requests.Requests().size();
  const std::vector<int64_t> position_ids_shape = {batch_size, max_sequence_length};
  auto position_ids_tensor = std::make_unique<Tensor>(model->p_device_inputs_, Ort::TypeToTensorType<int32_t>);
  auto device_span = position_ids_tensor->GetDeviceSpan<int32_t>();
  auto cpu_span = device_span.CpuSpan();

  for (size_t i = 0; i < batch_size; ++i) {
    auto request = scheduled_requests.Requests()[i];
    auto input_ids = request->UnprocessedTokens().CopyDeviceToCpu();
    auto current_sequence_length = request->CurrentSequenceLength();

    for (size_t j = 0; j < max_sequence_length; ++j) {
      cpu_span[i * max_sequence_length + j] = (j < input_ids.size() && input_ids[j] != model->config_->model.pad_token_id) ? current_sequence_length - 1 + j : 0;
    }
  }

  device_span.CopyCpuToDevice();

  input_names_.push_back(model->config_->model.decoder.inputs.position_ids.c_str());
  inputs_.push_back(position_ids_tensor->GetOrtTensor());
  owned_inputs_.push_back(std::move(position_ids_tensor));
}

VarlenDecoderIO::VarlenDecoderIO(std::shared_ptr<DecoderOnly_Model> model,
                                 ScheduledRequests& scheduled_requests,
                                 std::shared_ptr<CacheManager> cache_manager)
    : ModelIO(*scheduled_requests.Params(), *model) {
}

SimpleDecoder::SimpleDecoder(std::shared_ptr<DecoderOnly_Model> model,
                             std::shared_ptr<CacheManager> cache_manager)
    : model_{model}, cache_manager_{cache_manager} {}

void SimpleDecoder::Decode(ScheduledRequests& scheduled_requests) {
  cache_manager_->Step();
  std::unique_ptr<State> decoder_state =
      cache_manager_->SupportsContinuousBatching()
          ? static_cast<std::unique_ptr<State>>(std::make_unique<VarlenDecoderIO>(model_, scheduled_requests, cache_manager_))
          : static_cast<std::unique_ptr<State>>(std::make_unique<StaticBatchDecoderIO>(model_, scheduled_requests, cache_manager_));

  auto run_options = scheduled_requests.RunOptions();
  model_->session_decoder_->Run(run_options.get(),
                                decoder_state->input_names_.data(),
                                decoder_state->inputs_.data(),
                                decoder_state->input_names_.size(),
                                decoder_state->output_names_.data(),
                                decoder_state->outputs_.data(),
                                decoder_state->output_names_.size());
}

}  // namespace Generators
