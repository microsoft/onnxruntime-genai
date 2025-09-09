// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "varlen_decoder_io.h"
#include "../../models/decoder_only.h"

namespace Generators {

VarlenDecoderIO::VarlenDecoderIO(std::shared_ptr<DecoderOnly_Model> model,
                                 ScheduledRequests& scheduled_requests,
                                 std::shared_ptr<CacheManager> cache_manager)
    : DecoderIO(model, scheduled_requests, cache_manager) {
  PrepareInputIds(model, scheduled_requests);
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

void VarlenDecoderIO::PrepareInputIds(std::shared_ptr<DecoderOnly_Model> model, ScheduledRequests& scheduled_requests) {
  size_t num_tokens = std::accumulate(scheduled_requests.begin(), scheduled_requests.end(), static_cast<size_t>(0),
                                      [](size_t sum, const std::shared_ptr<Request>& request) -> size_t {
                                        return sum + request->UnprocessedTokens().size();
                                      });
  const std::vector<int64_t> input_ids_shape = {static_cast<int64_t>(num_tokens)};
  auto input_ids_tensor = std::make_unique<Tensor>(model->p_device_inputs_, Ort::TypeToTensorType<int64_t>);
  input_ids_tensor->CreateTensor(input_ids_shape);
  auto device_span = input_ids_tensor->GetDeviceSpan<int64_t>();
  auto cpu_span = device_span.CpuSpan();

  const std::vector<int64_t> cumulative_sequence_lengths_shape = {static_cast<int64_t>(scheduled_requests.size() + 1)};
  auto cumulative_sequence_lengths_tensor = std::make_unique<Tensor>(model->p_device_inputs_, Ort::TypeToTensorType<int32_t>);
  cumulative_sequence_lengths_tensor->CreateTensor(cumulative_sequence_lengths_shape);
  auto cumulative_sequence_lengths_span = cumulative_sequence_lengths_tensor->GetDeviceSpan<int32_t>();
  auto cumulative_sequence_lengths_cpu_span = cumulative_sequence_lengths_span.CpuSpan();
  cumulative_sequence_lengths_cpu_span[0] = 0;

  const std::vector<int64_t> sequence_lengths_shape = {static_cast<int64_t>(scheduled_requests.size())};
  auto sequence_lengths_tensor = std::make_unique<Tensor>(model->p_device_inputs_, Ort::TypeToTensorType<int32_t>);
  sequence_lengths_tensor->CreateTensor(sequence_lengths_shape);
  auto sequence_lengths_span = sequence_lengths_tensor->GetDeviceSpan<int32_t>();
  auto sequence_lengths_cpu_span = sequence_lengths_span.CpuSpan();

  for (size_t i = 0, running_length = 0; i < scheduled_requests.size(); ++i) {
    auto request = scheduled_requests[i];
    auto input_ids = request->UnprocessedTokens().CopyDeviceToCpu();
    std::copy(input_ids.begin(), input_ids.end(), cpu_span.begin() + running_length);

    if (request->IsPrefill()) {
      // When a request is created, the current sequence length becomes the prompt length.
      // But the kv cache is not updated until the first token is generated.
      // So we set the past sequence length to current sequence length minus the unprocessed tokens length.
      sequence_lengths_cpu_span[i] = static_cast<int32_t>(request->CurrentSequenceLength() - input_ids.size());
    } else {
      sequence_lengths_cpu_span[i] = static_cast<int32_t>(request->CurrentSequenceLength());
    }

    running_length += input_ids.size();
    cumulative_sequence_lengths_cpu_span[i + 1] = static_cast<int32_t>(running_length);
  }

  device_span.CopyCpuToDevice();
  cumulative_sequence_lengths_span.CopyCpuToDevice();
  sequence_lengths_span.CopyCpuToDevice();

  input_names_.push_back(model->config_->model.decoder.inputs.input_ids.c_str());
  inputs_.push_back(input_ids_tensor->GetOrtTensor());
  owned_inputs_.push_back(std::move(input_ids_tensor));

  input_names_.push_back(model->config_->model.decoder.inputs.cumulative_sequence_lengths.c_str());
  inputs_.push_back(cumulative_sequence_lengths_tensor->GetOrtTensor());
  owned_inputs_.push_back(std::move(cumulative_sequence_lengths_tensor));

  input_names_.push_back(model->config_->model.decoder.inputs.past_sequence_lengths.c_str());
  inputs_.push_back(sequence_lengths_tensor->GetOrtTensor());
  owned_inputs_.push_back(std::move(sequence_lengths_tensor));
}

void VarlenDecoderIO::PrepareLogits(std::shared_ptr<DecoderOnly_Model> model, ScheduledRequests& scheduled_requests) {
  size_t num_tokens = std::accumulate(scheduled_requests.begin(), scheduled_requests.end(), static_cast<size_t>(0),
                                      [](size_t sum, const std::shared_ptr<Request>& request) {
                                        return sum + request->UnprocessedTokens().size();
                                      });
  const std::vector<int64_t> logits_shape = {static_cast<int64_t>(num_tokens), static_cast<int64_t>(model->config_->model.vocab_size)};
  logits_ = std::make_unique<Tensor>(model->p_device_inputs_, model->session_info_.GetOutputDataType(model->config_->model.decoder.outputs.logits));
  logits_->CreateTensor(logits_shape);

  output_names_.push_back(model->config_->model.decoder.outputs.logits.c_str());
  outputs_.push_back(logits_->GetOrtTensor());
}

std::vector<DeviceSpan<float>> VarlenDecoderIO::ProcessLogits() {
  std::vector<size_t> valid_token_indices(scheduled_requests_.size());
  for (size_t i = 0, running_length = 0; i < scheduled_requests_.size(); ++i) {
    valid_token_indices[i] = running_length + scheduled_requests_[i]->UnprocessedTokens().size() - 1;
    running_length += scheduled_requests_[i]->UnprocessedTokens().size();
  }

  // [num_tokens, vocab_size]
  const auto all_tokens_logits_shape = logits_->GetShape();
  const int64_t vocab_size = all_tokens_logits_shape[1];
  const int64_t element_size = static_cast<int64_t>(Ort::SizeOf(logits_->GetType()));

  auto logits_bytes = logits_->GetByteSpan();
  std::vector<decltype(logits_bytes)> logits_bytes_vector;
  for (size_t i = 0; i < valid_token_indices.size(); ++i) {
    auto logits_of_last_token = logits_bytes.subspan(valid_token_indices[i] * vocab_size * element_size, vocab_size * element_size);
    logits_bytes_vector.push_back(logits_of_last_token);
  }

  std::vector<DeviceSpan<float>> logits_vector;
  const std::vector<int64_t> logits_shape{static_cast<int64_t>(scheduled_requests_.size()),
                                          vocab_size};

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

}  // namespace Generators
