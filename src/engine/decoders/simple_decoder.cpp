// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "simple_decoder.h"

#include <typeinfo>

namespace Generators {

StaticBatchDecoderIO::StaticBatchDecoderIO(std::shared_ptr<DecoderOnly_Model> model,
                                           ScheduledRequests& scheduled_requests,
                                           std::shared_ptr<CacheManager> cache_manager)
    : DecoderIO(model, scheduled_requests, cache_manager) {
  std::cout << "StaticBatchDecoderIO created." << std::endl;
  PrepareInputIds(model, scheduled_requests);
  std::cout << "Input IDs prepared." << std::endl;
  PrepareAttentionMask(model, scheduled_requests);
  std::cout << "Attention mask prepared." << std::endl;
  PreparePositionIds(model, scheduled_requests);
  std::cout << "Position IDs prepared." << std::endl;
  PrepareLogits(model, scheduled_requests);
  std::cout << "Logits prepared." << std::endl;
}

void StaticBatchDecoderIO::PrepareInputIds(std::shared_ptr<DecoderOnly_Model> model, ScheduledRequests& scheduled_requests) {
  auto request_with_max_sequence_length =
      std::max_element(
          scheduled_requests.begin(), scheduled_requests.end(),
          [](const std::shared_ptr<Request>& a, const std::shared_ptr<Request>& b) {
            return a->UnprocessedTokens().size() < b->UnprocessedTokens().size();
          });

  const int64_t max_sequence_length = (*request_with_max_sequence_length)->UnprocessedTokens().size();
  const int64_t batch_size = scheduled_requests.size();
  const std::vector<int64_t> input_ids_shape = {batch_size, max_sequence_length};
  auto input_ids_tensor = std::make_unique<Tensor>(model->p_device_inputs_, Ort::TypeToTensorType<int32_t>);
  input_ids_tensor->CreateTensor(input_ids_shape);
  auto device_span = input_ids_tensor->GetDeviceSpan<int32_t>();
  auto cpu_span = device_span.CpuSpan();

  for (size_t i = 0; i < batch_size; ++i) {
    auto request = scheduled_requests[i];
    auto input_ids = request->UnprocessedTokens().CopyDeviceToCpu();
    for (size_t j = 0; j < max_sequence_length; ++j) {
      cpu_span[i * max_sequence_length + j] = (j < input_ids.size()) ? input_ids[j] : model->config_->model.pad_token_id;
    }
  }

  device_span.CopyCpuToDevice();

  input_names_.push_back(model->config_->model.decoder.inputs.input_ids.c_str());
  inputs_.push_back(input_ids_tensor->GetOrtTensor());
  owned_inputs_.push_back(std::move(input_ids_tensor));
}

void StaticBatchDecoderIO::PrepareAttentionMask(std::shared_ptr<DecoderOnly_Model> model, ScheduledRequests& scheduled_requests) {
  auto request_with_max_sequence_length =
      std::max_element(
          scheduled_requests.begin(), scheduled_requests.end(),
          [](const std::shared_ptr<Request>& a, const std::shared_ptr<Request>& b) {
            return a->UnprocessedTokens().size() < b->UnprocessedTokens().size();
          });

  const int64_t max_sequence_length = (*request_with_max_sequence_length)->UnprocessedTokens().size();
  const int64_t batch_size = scheduled_requests.size();
  const std::vector<int64_t> attention_mask_shape = {batch_size, max_sequence_length};
  auto attention_mask_tensor = std::make_unique<Tensor>(model->p_device_inputs_, Ort::TypeToTensorType<int32_t>);
  attention_mask_tensor->CreateTensor(attention_mask_shape);
  auto device_span = attention_mask_tensor->GetDeviceSpan<int32_t>();
  auto cpu_span = device_span.CpuSpan();

  for (size_t i = 0; i < batch_size; ++i) {
    auto request = scheduled_requests[i];
    auto input_ids = request->UnprocessedTokens().CopyDeviceToCpu();

    for (size_t j = 0; j < max_sequence_length; ++j) {
      cpu_span[i * max_sequence_length + j] = (j < input_ids.size() && input_ids[j] != model->config_->model.pad_token_id) ? 1 : 0;
    }
  }

  std::cout << "Attention mask prepared." << std::endl;
  device_span.CopyCpuToDevice();
  std::cout << "Attention mask copied to device." << std::endl;

  input_names_.push_back(model->config_->model.decoder.inputs.attention_mask.c_str());
  inputs_.push_back(attention_mask_tensor->GetOrtTensor());
  owned_inputs_.push_back(std::move(attention_mask_tensor));
}

void StaticBatchDecoderIO::PreparePositionIds(std::shared_ptr<DecoderOnly_Model> model, ScheduledRequests& scheduled_requests) {
  auto request_with_max_sequence_length =
      std::max_element(
          scheduled_requests.begin(), scheduled_requests.end(),
          [](const std::shared_ptr<Request>& a, const std::shared_ptr<Request>& b) {
            return a->UnprocessedTokens().size() < b->UnprocessedTokens().size();
          });

  const int64_t max_sequence_length = (*request_with_max_sequence_length)->UnprocessedTokens().size();
  const int64_t batch_size = scheduled_requests.size();
  const std::vector<int64_t> position_ids_shape = {batch_size, max_sequence_length};
  auto position_ids_tensor = std::make_unique<Tensor>(model->p_device_inputs_, Ort::TypeToTensorType<int32_t>);
  position_ids_tensor->CreateTensor(position_ids_shape);
  auto device_span = position_ids_tensor->GetDeviceSpan<int32_t>();
  auto cpu_span = device_span.CpuSpan();

  for (size_t i = 0; i < batch_size; ++i) {
    auto request = scheduled_requests[i];
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

void StaticBatchDecoderIO::PrepareLogits(std::shared_ptr<DecoderOnly_Model> model, ScheduledRequests& scheduled_requests) {
  auto request_with_max_sequence_length =
      std::max_element(
          scheduled_requests.begin(), scheduled_requests.end(),
          [](const std::shared_ptr<Request>& a, const std::shared_ptr<Request>& b) {
            return a->UnprocessedTokens().size() < b->UnprocessedTokens().size();
          });

  const int64_t max_sequence_length = (*request_with_max_sequence_length)->UnprocessedTokens().size();
  const int64_t batch_size = scheduled_requests.size();
  const std::vector<int64_t> logits_shape = {batch_size, max_sequence_length, model->config_->model.vocab_size};
  auto logits_ = std::make_unique<Tensor>(model->p_device_inputs_, Ort::TypeToTensorType<int32_t>);
  logits_->CreateTensor(logits_shape);

  input_names_.push_back(model->config_->model.decoder.outputs.logits.c_str());
  inputs_.push_back(logits_->GetOrtTensor());
}

VarlenDecoderIO::VarlenDecoderIO(std::shared_ptr<DecoderOnly_Model> model,
                                 ScheduledRequests& scheduled_requests,
                                 std::shared_ptr<CacheManager> cache_manager)
    : DecoderIO(model, scheduled_requests, cache_manager) {
}

SimpleDecoder::SimpleDecoder(std::shared_ptr<DecoderOnly_Model> model,
                             std::shared_ptr<CacheManager> cache_manager)
    : model_{model}, cache_manager_{cache_manager} {}

void SimpleDecoder::Decode(ScheduledRequests& scheduled_requests) {
  std::cout << "SimpleDecoder::Decode" << std::endl;
  cache_manager_->Step();
  std::cout << "Cache manager step completed" << std::endl;
  std::unique_ptr<DecoderIO> decoder_state =
      cache_manager_->SupportsDynamicBatching()
          ? static_cast<std::unique_ptr<DecoderIO>>(std::make_unique<VarlenDecoderIO>(model_, scheduled_requests, cache_manager_))
          : static_cast<std::unique_ptr<DecoderIO>>(std::make_unique<StaticBatchDecoderIO>(model_, scheduled_requests, cache_manager_));

  std::cout << "Decoder state created" << std::endl;

  auto run_options = scheduled_requests.RunOptions();
  model_->session_decoder_->Run(run_options.get(),
                                decoder_state->input_names_.data(),
                                decoder_state->inputs_.data(),
                                decoder_state->input_names_.size(),
                                decoder_state->output_names_.data(),
                                decoder_state->outputs_.data(),
                                decoder_state->output_names_.size());

  scheduled_requests.AddDecoderState(std::move(decoder_state));
}

}  // namespace Generators
