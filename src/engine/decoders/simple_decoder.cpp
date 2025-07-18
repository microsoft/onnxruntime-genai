// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "simple_decoder.h"

#include <typeinfo>

namespace Generators {

StaticBatchDecoderIO::StaticBatchDecoderIO(std::shared_ptr<DecoderOnly_Model> model,
                                           ScheduledRequests& scheduled_requests,
                                           std::shared_ptr<CacheManager> cache_manager)
    : DecoderIO(model, scheduled_requests, cache_manager) {
  PrepareInputIds(model, scheduled_requests);
  PrepareAttentionMask(model, scheduled_requests);
  PreparePositionIds(model, scheduled_requests);
  PrepareLogits(model, scheduled_requests);

  auto cache = cache_manager->Cache();
  for (size_t i = 0; i < cache->input_names_.size(); ++i) {
    input_names_.push_back(cache->input_names_[i]);
    inputs_.push_back(cache->inputs_[i]);
  }

  for (size_t i = 0; i < cache->output_names_.size(); ++i) {
    output_names_.push_back(cache->output_names_[i]);
    outputs_.push_back(cache->outputs_[i]);
  }
}

void StaticBatchDecoderIO::PrepareInputIds(std::shared_ptr<DecoderOnly_Model> model, ScheduledRequests& scheduled_requests) {
  auto request_with_max_sequence_length =
      std::max_element(
          scheduled_requests.begin(), scheduled_requests.end(),
          [](const std::shared_ptr<Request>& a, const std::shared_ptr<Request>& b) {
            return a->UnprocessedTokens().size() < b->UnprocessedTokens().size();
          });

  const size_t max_sequence_length = (*request_with_max_sequence_length)->UnprocessedTokens().size();
  const size_t batch_size = scheduled_requests.size();
  const std::vector<int64_t> input_ids_shape = {static_cast<int64_t>(batch_size), static_cast<int64_t>(max_sequence_length)};
  auto input_ids_tensor = std::make_unique<Tensor>(model->p_device_inputs_, Ort::TypeToTensorType<int64_t>);
  input_ids_tensor->CreateTensor(input_ids_shape);
  auto device_span = input_ids_tensor->GetDeviceSpan<int64_t>();
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
            return a->CurrentSequenceLength() < b->CurrentSequenceLength();
          });

  const size_t max_sequence_length = (*request_with_max_sequence_length)->CurrentSequenceLength();
  const size_t batch_size = scheduled_requests.size();
  const std::vector<int64_t> attention_mask_shape = {static_cast<int64_t>(batch_size), static_cast<int64_t>(max_sequence_length)};
  auto attention_mask_tensor = std::make_unique<Tensor>(model->p_device_inputs_, Ort::TypeToTensorType<int64_t>);
  attention_mask_tensor->CreateTensor(attention_mask_shape);
  auto device_span = attention_mask_tensor->GetDeviceSpan<int64_t>();
  auto cpu_span = device_span.CpuSpan();

  for (size_t i = 0; i < batch_size; ++i) {
    auto request = scheduled_requests[i];
    const size_t current_sequence_length = static_cast<size_t>(request->CurrentSequenceLength());

    for (size_t j = 0; j < max_sequence_length; ++j) {
      cpu_span[i * max_sequence_length + j] = (j < current_sequence_length) ? 1 : 0;
    }
  }

  device_span.CopyCpuToDevice();

  input_names_.push_back(model->config_->model.decoder.inputs.attention_mask.c_str());
  inputs_.push_back(attention_mask_tensor->GetOrtTensor());
  owned_inputs_.push_back(std::move(attention_mask_tensor));
}

void StaticBatchDecoderIO::PreparePositionIds(std::shared_ptr<DecoderOnly_Model> model, ScheduledRequests& scheduled_requests) {
  if (!model->session_info_.HasInput(model->config_->model.decoder.inputs.position_ids)) {
    return;
  }

  auto request_with_max_sequence_length =
      std::max_element(
          scheduled_requests.begin(), scheduled_requests.end(),
          [](const std::shared_ptr<Request>& a, const std::shared_ptr<Request>& b) {
            return a->UnprocessedTokens().size() < b->UnprocessedTokens().size();
          });

  const size_t max_sequence_length = (*request_with_max_sequence_length)->UnprocessedTokens().size();
  const size_t batch_size = scheduled_requests.size();
  const std::vector<int64_t> position_ids_shape = {static_cast<int64_t>(batch_size), static_cast<int64_t>(max_sequence_length)};
  auto position_ids_tensor = std::make_unique<Tensor>(model->p_device_inputs_, Ort::TypeToTensorType<int64_t>);
  position_ids_tensor->CreateTensor(position_ids_shape);
  auto device_span = position_ids_tensor->GetDeviceSpan<int64_t>();
  auto cpu_span = device_span.CpuSpan();

  for (size_t i = 0; i < batch_size; ++i) {
    auto request = scheduled_requests[i];
    auto input_ids = request->UnprocessedTokens().CopyDeviceToCpu();
    auto current_sequence_length = request->IsPrefill() ? 1 : request->CurrentSequenceLength();

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
  logits_ = std::make_unique<Tensor>(model->p_device_inputs_, model->session_info_.GetOutputDataType(model->config_->model.decoder.outputs.logits));
  logits_->CreateTensor(logits_shape);

  output_names_.push_back(model->config_->model.decoder.outputs.logits.c_str());
  outputs_.push_back(logits_->GetOrtTensor());
}

std::vector<DeviceSpan<float>> StaticBatchDecoderIO::ProcessLogits() {
  std::vector<int64_t> valid_token_indices;
  for (auto& request : scheduled_requests_) {
    if (request->IsPrefill()) {
      valid_token_indices.push_back(request->CurrentSequenceLength() - 1);
    } else {
      valid_token_indices.push_back(0);
    }
  }

  // [batch_size, max_sequence_length, vocab_size]
  const auto all_tokens_logits_shape = logits_->GetShape();
  const int64_t batch_size = all_tokens_logits_shape[0],
                max_sequence_length = all_tokens_logits_shape[1],
                vocab_size = all_tokens_logits_shape[2];
  const int64_t element_size = static_cast<int64_t>(Ort::SizeOf(logits_->GetType()));

  auto logits_bytes = logits_->GetByteSpan();
  std::vector<decltype(logits_bytes)> logits_bytes_vector;
  for (size_t i = 0; i < valid_token_indices.size(); ++i) {
    auto logits_of_last_token = logits_bytes.subspan((i * max_sequence_length * vocab_size + valid_token_indices[i] * vocab_size) * element_size,
                                                     vocab_size);
    logits_bytes_vector.push_back(logits_of_last_token);
  }

  std::vector<DeviceSpan<float>> logits_vector;
  const std::vector<int64_t> logits_shape{batch_size, vocab_size};

  const bool requires_cast = logits_->GetType() != Ort::TypeToTensorType<float>;
  if (requires_cast) {
    logits_fp32_ = std::make_unique<Tensor>(model_.p_device_inputs_, Ort::TypeToTensorType<float>);
    logits_fp32_->CreateTensor(logits_shape);
  }

  for (size_t i = 0; i < logits_bytes_vector.size(); ++i) {
    if (requires_cast) {
      auto logits_of_last_token_fp32 = logits_fp32_->GetDeviceSpan<float>().subspan(i * vocab_size, vocab_size);
      void* src_data = logits_bytes_vector[i].Span().data();
      void* dst_data = logits_of_last_token_fp32.Span().data();
      model_.p_device_inputs_->Cast(src_data, dst_data, logits_->GetType(), Ort::TypeToTensorType<float>, vocab_size);
      logits_vector.push_back(logits_of_last_token_fp32);
    } else {
      auto logits_of_last_token_fp32 = model_.p_device_inputs_->WrapMemory<float>(
          std::span(reinterpret_cast<float*>(logits_bytes_vector[i].Span().data()), vocab_size));
      logits_vector.push_back(logits_of_last_token_fp32);
    }
  }

  return logits_vector;
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
  cache_manager_->Step();
  std::unique_ptr<DecoderIO> decoder_state =
      cache_manager_->SupportsDynamicBatching()
          ? static_cast<std::unique_ptr<DecoderIO>>(std::make_unique<VarlenDecoderIO>(model_, scheduled_requests, cache_manager_))
          : static_cast<std::unique_ptr<DecoderIO>>(std::make_unique<StaticBatchDecoderIO>(model_, scheduled_requests, cache_manager_));

  auto run_options = scheduled_requests.RunOptions();
  decoder_state->DumpInputs();
  model_->session_decoder_->Run(run_options.get(),
                                decoder_state->input_names_.data(),
                                decoder_state->inputs_.data(),
                                decoder_state->input_names_.size(),
                                decoder_state->output_names_.data(),
                                decoder_state->outputs_.data(),
                                decoder_state->output_names_.size());
  decoder_state->DumpOutputs();

  scheduled_requests.AddDecoderState(std::move(decoder_state));
}

}  // namespace Generators
