// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "varlen_decoder_io.h"
#include "../../models/decoder_only.h"
#include "../sequence_positions.h"

namespace Generators {

VarlenGraphBuffers::VarlenGraphBuffers(DecoderOnly_Model& model) {
  max_batch_size = model.config_->engine.dynamic_batching->max_batch_size;

  auto make = [&](DeviceInterface* device, ONNXTensorElementDataType type, int64_t elements) {
    auto tensor = std::make_unique<Tensor>(device, type);
    tensor->CreateTensor(std::vector<int64_t>{elements}, /*make_static=*/true);
    return tensor;
  };

  // One token per sequence on a decode step, so the token count equals the batch size.
  input_ids = make(model.p_device_inputs_, Ort::TypeToTensorType<int64_t>, static_cast<int64_t>(max_batch_size));
  cumulative_sequence_lengths =
      make(model.p_device_inputs_, Ort::TypeToTensorType<int32_t>, static_cast<int64_t>(max_batch_size + 1));
  past_sequence_lengths =
      make(model.p_device_inputs_, Ort::TypeToTensorType<int32_t>, static_cast<int64_t>(max_batch_size));

  logits = std::make_unique<Tensor>(model.p_device_inputs_,
                                    model.session_info_.GetOutputDataType(model.config_->model.decoder.outputs.logits));
  logits->CreateTensor(std::vector<int64_t>{static_cast<int64_t>(max_batch_size),
                                            static_cast<int64_t>(model.config_->model.vocab_size)},
                       /*make_static=*/true);
}

int VarlenGraphBuffers::GraphId(size_t batch_size, size_t block_table_columns) {
  // Block table columns are bucketed to powers of two by the cache, so the exponent is enough to
  // separate them. Ids must be positive: ORT reserves -1 for "do not capture or replay".
  int columns_bucket = 0;
  while ((size_t{1} << columns_bucket) < block_table_columns) {
    ++columns_bucket;
  }
  return static_cast<int>(batch_size) * 64 + columns_bucket + 1;
}

VarlenDecoderIO::VarlenDecoderIO(std::shared_ptr<DecoderOnly_Model> model,
                                 ScheduledRequests& scheduled_requests,
                                 std::shared_ptr<CacheManager> cache_manager,
                                 VarlenGraphBuffers* graph_buffers)
    : DecoderIO(model, scheduled_requests, cache_manager), graph_buffers_{graph_buffers} {
  PrepareInputIds(model, scheduled_requests);
  PrepareAttentionMetadata(model, scheduled_requests);
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
  // On a capturable step the tensors are views onto buffers that were allocated once, so their
  // device addresses match the ones recorded in the graph. Otherwise each step allocates its own.
  auto reshape = [&](std::unique_ptr<Tensor>& owned, Tensor* borrowed, ONNXTensorElementDataType type,
                     std::vector<int64_t> shape) -> Tensor* {
    if (borrowed != nullptr) {
      borrowed->CreateTensor(shape, /*make_static=*/true);
      return borrowed;
    }
    owned = std::make_unique<Tensor>(model->p_device_inputs_, type);
    owned->CreateTensor(shape);
    return owned.get();
  };

  std::unique_ptr<Tensor> owned_input_ids, owned_cumulative, owned_sequence_lengths;
  Tensor* input_ids_tensor =
      reshape(owned_input_ids, graph_buffers_ ? graph_buffers_->input_ids.get() : nullptr,
              Ort::TypeToTensorType<int64_t>, {static_cast<int64_t>(num_tokens)});
  Tensor* cumulative_sequence_lengths_tensor =
      reshape(owned_cumulative, graph_buffers_ ? graph_buffers_->cumulative_sequence_lengths.get() : nullptr,
              Ort::TypeToTensorType<int32_t>, {static_cast<int64_t>(scheduled_requests.size() + 1)});
  Tensor* sequence_lengths_tensor =
      reshape(owned_sequence_lengths, graph_buffers_ ? graph_buffers_->past_sequence_lengths.get() : nullptr,
              Ort::TypeToTensorType<int32_t>, {static_cast<int64_t>(scheduled_requests.size())});

  auto device_span = input_ids_tensor->GetDeviceSpan<int64_t>();
  auto cpu_span = device_span.CpuSpan();

  auto cumulative_sequence_lengths_span = cumulative_sequence_lengths_tensor->GetDeviceSpan<int32_t>();
  auto cumulative_sequence_lengths_cpu_span = cumulative_sequence_lengths_span.CpuSpan();
  cumulative_sequence_lengths_cpu_span[0] = 0;

  auto sequence_lengths_span = sequence_lengths_tensor->GetDeviceSpan<int32_t>();
  auto sequence_lengths_cpu_span = sequence_lengths_span.CpuSpan();

  for (size_t i = 0, running_length = 0; i < scheduled_requests.size(); ++i) {
    auto request = scheduled_requests[i];
    auto input_ids = request->UnprocessedTokensCpu();
    std::copy(input_ids.begin(), input_ids.end(), cpu_span.begin() + running_length);

    // The operator writes this step's token j of sequence i at absolute position
    // past_sequence_lengths[i] + j, so this has to be the number of tokens already in the cache.
    // Deriving it from the sequence length instead would be off by one on a decode step, where the
    // search has already appended the token that is about to be processed, and would be wrong by a
    // whole chunk during a chunked prefill.
    sequence_lengths_cpu_span[i] = static_cast<int32_t>(request->ProcessedSequenceLength());

    running_length += input_ids.size();
    cumulative_sequence_lengths_cpu_span[i + 1] = static_cast<int32_t>(running_length);
  }

  device_span.CopyCpuToDevice();
  cumulative_sequence_lengths_span.CopyCpuToDevice();
  sequence_lengths_span.CopyCpuToDevice();

  input_names_.push_back(model->config_->model.decoder.inputs.input_ids.c_str());
  inputs_.push_back(input_ids_tensor->GetOrtTensor());

  input_names_.push_back(model->config_->model.decoder.inputs.cumulative_sequence_lengths.c_str());
  inputs_.push_back(cumulative_sequence_lengths_tensor->GetOrtTensor());

  input_names_.push_back(model->config_->model.decoder.inputs.past_sequence_lengths.c_str());
  inputs_.push_back(sequence_lengths_tensor->GetOrtTensor());

  if (owned_input_ids) owned_inputs_.push_back(std::move(owned_input_ids));
  if (owned_cumulative) owned_inputs_.push_back(std::move(owned_cumulative));
  if (owned_sequence_lengths) owned_inputs_.push_back(std::move(owned_sequence_lengths));
}

// PagedAttention accepts an optional `attention_metadata` CPU input holding
// [max_query_len_bound, max_kv_len_bound]. Both are upper bounds on the current step; the operator
// uses them only to select a backend and to size its launch dimensions and workspaces, never as a
// mask boundary. Supplying them lets the operator skip the device-to-host readback of the sequence
// lengths, which otherwise forces a full stream synchronization inside every attention node.
//
// The engine already has both quantities on the host while it builds the sequence length inputs, so
// this costs nothing. The bounds are exact rather than conservative, which is the tightest valid
// choice: a larger bound is still correct but sizes the launch for more work than the step needs.
void VarlenDecoderIO::PrepareAttentionMetadata(std::shared_ptr<DecoderOnly_Model> model, ScheduledRequests& scheduled_requests) {
  const std::string& metadata_name = model->config_->model.decoder.inputs.attention_metadata;
  if (!model->session_info_.HasInput(metadata_name)) {
    // Model was built before `attention_metadata` existed. The operator falls back to the readback.
    return;
  }

  int32_t max_query_len = 0;
  int32_t max_kv_len = 0;
  if (graph_buffers_ != nullptr) {
    // A CPU input is read once, while the graph is being captured, and the recorded launch is reused
    // for every later replay. Reporting the current step's exact lengths would freeze them, so a
    // capturable step reports bounds that hold for the whole life of the graph instead: one token per
    // sequence, and the KV length the block table can address at its current column count.
    max_query_len = 1;
    max_kv_len = static_cast<int32_t>(cache_manager_->BlockTableColumns() *
                                      model->config_->engine.dynamic_batching->block_size);
  } else {
    for (auto& request : scheduled_requests) {
      const int32_t query_len = static_cast<int32_t>(request->UnprocessedTokens().size());
      const int32_t kv_len = static_cast<int32_t>(SlotsAfterStep(request->ProcessedSequenceLength(), query_len));
      max_query_len = std::max(max_query_len, query_len);
      max_kv_len = std::max(max_kv_len, kv_len);
    }
  }

  auto metadata_tensor = std::make_unique<Tensor>(GetDeviceInterface(DeviceType::CPU), Ort::TypeToTensorType<int32_t>);
  metadata_tensor->CreateTensor(std::vector<int64_t>{2});
  auto metadata_span = metadata_tensor->GetDeviceSpan<int32_t>().CpuSpan();
  metadata_span[0] = max_query_len;
  metadata_span[1] = max_kv_len;

  input_names_.push_back(metadata_name.c_str());
  inputs_.push_back(metadata_tensor->GetOrtTensor());
  owned_inputs_.push_back(std::move(metadata_tensor));
}

void VarlenDecoderIO::PrepareLogits(std::shared_ptr<DecoderOnly_Model> model, ScheduledRequests& scheduled_requests) {
  size_t num_tokens = std::accumulate(scheduled_requests.begin(), scheduled_requests.end(), static_cast<size_t>(0),
                                      [](size_t sum, const std::shared_ptr<Request>& request) {
                                        return sum + request->UnprocessedTokens().size();
                                      });
  const std::vector<int64_t> logits_shape = {static_cast<int64_t>(num_tokens), static_cast<int64_t>(model->config_->model.vocab_size)};
  if (graph_buffers_ != nullptr) {
    graph_buffers_->logits->CreateTensor(logits_shape, /*make_static=*/true);
    active_logits_ = graph_buffers_->logits.get();
  } else {
    logits_ = std::make_unique<Tensor>(model->p_device_inputs_, model->session_info_.GetOutputDataType(model->config_->model.decoder.outputs.logits));
    logits_->CreateTensor(logits_shape);
    active_logits_ = logits_.get();
  }

  output_names_.push_back(model->config_->model.decoder.outputs.logits.c_str());
  outputs_.push_back(active_logits_->GetOrtTensor());
}

std::vector<DeviceSpan<float>> VarlenDecoderIO::ProcessLogits() {
  std::vector<size_t> valid_token_indices(scheduled_requests_.size());
  for (size_t i = 0, running_length = 0; i < scheduled_requests_.size(); ++i) {
    valid_token_indices[i] = running_length + scheduled_requests_[i]->UnprocessedTokens().size() - 1;
    running_length += scheduled_requests_[i]->UnprocessedTokens().size();
  }

  // [num_tokens, vocab_size]
  const auto all_tokens_logits_shape = active_logits_->GetShape();
  const int64_t vocab_size = all_tokens_logits_shape[1];
  const int64_t element_size = static_cast<int64_t>(Ort::SizeOf(active_logits_->GetType()));

  auto logits_bytes = active_logits_->GetByteSpan();
  std::vector<decltype(logits_bytes)> logits_bytes_vector;
  for (size_t i = 0; i < valid_token_indices.size(); ++i) {
    auto logits_of_last_token = logits_bytes.subspan(valid_token_indices[i] * vocab_size * element_size, vocab_size * element_size);
    logits_bytes_vector.push_back(logits_of_last_token);
  }

  std::vector<DeviceSpan<float>> logits_vector;
  const std::vector<int64_t> logits_shape{static_cast<int64_t>(scheduled_requests_.size()),
                                          vocab_size};

  const bool requires_cast = active_logits_->GetType() != Ort::TypeToTensorType<float>;
  DeviceSpan<float> logits_fp32_span;
  if (requires_cast) {
    logits_fp32_ = std::make_unique<Tensor>(model_.p_device_inputs_, Ort::TypeToTensorType<float>);
    logits_fp32_->CreateTensor(logits_shape);
    // Wrapped once so that every row below is a subspan of the same allocation: GetDeviceSpan()
    // wraps the tensor memory afresh on each call, which would leave the rows unrelated.
    logits_fp32_span = logits_fp32_->GetDeviceSpan<float>();
  }

  // On a pure decode step every request contributes exactly one token, so the rows the search needs
  // are already the first `batch` rows of the output and the whole batch converts in one launch
  // instead of one launch per request.
  bool rows_are_contiguous = !valid_token_indices.empty();
  for (size_t i = 0; i < valid_token_indices.size() && rows_are_contiguous; ++i) {
    rows_are_contiguous = valid_token_indices[i] == i;
  }

  if (requires_cast && rows_are_contiguous) {
    model_.p_device_inputs_->Cast(logits_bytes.Span().data(), logits_fp32_span.Span().data(),
                                  active_logits_->GetType(), Ort::TypeToTensorType<float>,
                                  valid_token_indices.size() * static_cast<size_t>(vocab_size));
    for (size_t i = 0; i < valid_token_indices.size(); ++i) {
      logits_vector.push_back(logits_fp32_span.subspan(i * vocab_size, vocab_size));
    }
    return logits_vector;
  }

  for (size_t i = 0; i < logits_bytes_vector.size(); ++i) {
    if (requires_cast) {
      auto logits_of_last_token_fp32 = logits_fp32_span.subspan(i * vocab_size, vocab_size);
      void* src_data = logits_bytes_vector[i].Span().data();
      void* dst_data = logits_of_last_token_fp32.Span().data();
      model_.p_device_inputs_->Cast(src_data, dst_data, active_logits_->GetType(), Ort::TypeToTensorType<float>, vocab_size);
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
