// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "varlen_decoder_io.h"

#include <algorithm>
#include <array>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <string>
#include <string_view>

#include "../../models/decoder_only.h"
#include "../paged_key_value_cache.h"
#include "../sequence_positions.h"

namespace Generators {

void ValidatePackedPositionIdsInput(
    ONNXTensorElementDataType data_type,
    std::span<const int64_t> shape,
    std::span<const char* const> symbolic_shape) {
  const bool packed_vector = shape.size() == 1 && shape[0] < 0;
  const bool packed_mrope =
      shape.size() == 2 && shape[0] == 3 && shape[1] < 0;
  const char* token_dimension =
      symbolic_shape.size() == shape.size() ? symbolic_shape.back() : nullptr;
  if (data_type != ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64 ||
      (!packed_vector && !packed_mrope) ||
      (token_dimension &&
       std::string_view{token_dimension} == "batch_size")) {
    throw std::runtime_error(
        "Packed hybrid execution requires position_ids with dynamic int64 "
        "[num_tokens] or [3, num_tokens] geometry");
  }
}

namespace {

int32_t CheckedMetadataLength(size_t value, const char* name) {
  if (value > static_cast<size_t>(std::numeric_limits<int32_t>::max())) {
    throw std::runtime_error(std::string{name} + " exceeds the int32 attention metadata range.");
  }
  return static_cast<int32_t>(value);
}

int32_t CheckedMetadataLength(int64_t value, const char* name) {
  if (value < 0 || value > std::numeric_limits<int32_t>::max()) {
    throw std::runtime_error(std::string{name} + " is outside the int32 attention metadata range.");
  }
  return static_cast<int32_t>(value);
}

int32_t CheckedMetadataLength(size_t count, size_t block_size, const char* name) {
  const size_t int32_max = static_cast<size_t>(std::numeric_limits<int32_t>::max());
  if (count > int32_max / block_size) {
    throw std::runtime_error(std::string{name} + " exceeds the int32 attention metadata range.");
  }
  return static_cast<int32_t>(count * block_size);
}

}  // namespace

AttentionMetadataValues GetAttentionMetadataForPlan(const StepPlan& plan) {
  AttentionMetadataValues metadata;
  for (const auto& entry : plan.requests) {
    metadata.max_query_len_bound =
        std::max(metadata.max_query_len_bound,
                 CheckedMetadataLength(entry.unprocessed_token_count, "Query length"));
    metadata.max_kv_len_bound =
        std::max(metadata.max_kv_len_bound,
                 CheckedMetadataLength(entry.target_cache_slots, "KV length"));
  }
  metadata.max_kv_len_lower_bound = metadata.max_kv_len_bound;
  return metadata;
}

AttentionMetadataValues GetAttentionMetadataForGraph(size_t block_table_columns, size_t block_size) {
  if (block_table_columns == 0 || block_size == 0) {
    throw std::runtime_error("Captured attention metadata requires non-zero block-table columns and block size.");
  }

  size_t max_kv_len_lower_bound = 1;
  if (block_table_columns > kMinGraphBlockTableColumns) {
    // Graph shapes use 8, 16, 32, ... columns, with the configured maximum possibly truncating the
    // final bucket. Reaching a bucket wider than eight proves that the longest live table exceeded
    // the preceding power-of-two boundary because capturable decode steps reserve exactly the blocks
    // needed by their current KV length. This remains true when the graph is reused for different
    // requests, unlike the exact length observed during capture.
    size_t preceding_bucket = 1;
    while (preceding_bucket <= (block_table_columns - 1) / 2) {
      preceding_bucket *= 2;
    }
    max_kv_len_lower_bound =
        static_cast<size_t>(CheckedMetadataLength(preceding_bucket, block_size, "KV lower bound")) + 1;
  }

  return {
      1,
      CheckedMetadataLength(block_table_columns, block_size, "KV length"),
      CheckedMetadataLength(max_kv_len_lower_bound, "KV lower bound"),
  };
}

AttentionMetadataValues GetAttentionMetadataForGraphStep(
    const StepPlan& plan, size_t block_table_columns, size_t block_size) {
  auto metadata = GetAttentionMetadataForGraph(block_table_columns, block_size);
  const auto exact_metadata = GetAttentionMetadataForPlan(plan);
  if (exact_metadata.max_query_len_bound > metadata.max_query_len_bound ||
      exact_metadata.max_kv_len_bound > metadata.max_kv_len_bound) {
    throw std::runtime_error("Captured attention metadata upper bounds do not cover the current step.");
  }
  if (exact_metadata.max_kv_len_lower_bound < metadata.max_kv_len_lower_bound) {
    // The lower bound is only a backend-selection hint. If reservation policy ever gets ahead of
    // the live KV length, keep the graph correct and conservatively disable lower-bound optimizations.
    metadata.max_kv_len_lower_bound = 1;
  }
  return metadata;
}

std::array<int32_t, kAttentionMetadataElementCount> PackAttentionMetadata(
    const AttentionMetadataValues& metadata) {
  return {
      metadata.max_query_len_bound,
      metadata.max_kv_len_bound,
      metadata.max_kv_len_lower_bound,
  };
}

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
                                 const ExecutionContext* execution_context,
                                 VarlenGraphBuffers* graph_buffers,
                                 size_t position_planes)
    : DecoderIO(model, scheduled_requests, cache_manager),
      graph_buffers_{graph_buffers},
      plan_{execution_context ? execution_context->plan : nullptr},
      block_table_columns_{
          execution_context ? execution_context->block_table_columns : 0},
      position_planes_{position_planes} {
  // Logits with a symbolic batch_size first dimension contain one row per request. Any other first
  // dimension is treated as one row per packed token.
  const auto logits_symbolic_shape =
      model->session_info_.GetOutputSymbolicShape(model->config_->model.decoder.outputs.logits);
  logits_are_per_token_ = logits_symbolic_shape.empty() ||
                          logits_symbolic_shape[0] == nullptr ||
                          std::string_view(logits_symbolic_shape[0]) != "batch_size";

  PrepareInputIds(model, scheduled_requests);
  PreparePositionIds(model, scheduled_requests);
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
  const StepPlan* plan = plan_;
  if (plan && plan->requests.size() != scheduled_requests.size()) {
    throw std::runtime_error("Step plan size does not match the scheduled batch.");
  }
  const size_t num_tokens =
      plan ? plan->token_count
           : std::accumulate(scheduled_requests.begin(), scheduled_requests.end(), size_t{0},
                             [](size_t sum, const std::shared_ptr<Request>& request) {
                               return sum + request->ScheduledTokenCount();
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
    const RequestStepPlan* entry = plan ? &plan->requests[i] : nullptr;
    if (entry && (entry->request != request ||
                  entry->unprocessed_token_count != input_ids.size() ||
                  entry->packed_token_offset != running_length)) {
      throw std::runtime_error("Step plan token layout does not match the scheduled request.");
    }
    std::copy(input_ids.begin(), input_ids.end(), cpu_span.begin() + running_length);

    // The batch is represented as three coordinated arrays:
    //   input_ids                  = all pending tokens concatenated
    //   cumulative_sequence_lengths = boundaries of each request in that flat token array
    //   past_sequence_lengths       = the absolute KV-cache write position for each request
    // The operator writes token j at past_sequence_lengths[i] + j. The processed cursor is the
    // number of tokens already in the cache and therefore the base position for this step.
    const int64_t processed_sequence_length = request->ProcessedSequenceLength();
    if (entry && (request->CurrentSequenceLength() != entry->sequence_length_before ||
                  SlotsAfterStep(processed_sequence_length, entry->unprocessed_token_count) !=
                      entry->target_cache_slots)) {
      throw std::runtime_error("Step plan processed sequence length does not match the request.");
    }
    sequence_lengths_cpu_span[i] = static_cast<int32_t>(processed_sequence_length);

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

void VarlenDecoderIO::PreparePositionIds(
    std::shared_ptr<DecoderOnly_Model> model,
    ScheduledRequests& scheduled_requests) {
  if (position_planes_ == 0) {
    return;
  }
  const std::string& position_ids_name =
      model->config_->model.decoder.inputs.position_ids;
  if (graph_buffers_ != nullptr) {
    throw std::logic_error(
        "Packed position_ids are not supported by CUDA graph capture.");
  }

  const StepPlan* plan = plan_;
  if (plan && plan->requests.size() != scheduled_requests.size()) {
    throw std::logic_error(
        "Step plan size does not match packed position_ids batch.");
  }
  const size_t num_tokens =
      plan ? plan->token_count
           : std::accumulate(
                 scheduled_requests.begin(), scheduled_requests.end(), size_t{0},
                 [](size_t sum, const std::shared_ptr<Request>& request) {
                   return sum + request->ScheduledTokenCount();
                 });
  auto position_ids = std::make_unique<Tensor>(
      model->p_device_inputs_, Ort::TypeToTensorType<int64_t>);
  const std::vector<int64_t> position_shape =
      position_planes_ == 1
          ? std::vector<int64_t>{static_cast<int64_t>(num_tokens)}
          : std::vector<int64_t>{3, static_cast<int64_t>(num_tokens)};
  position_ids->CreateTensor(position_shape);
  auto position_span = position_ids->GetDeviceSpan<int64_t>();
  auto position_cpu = position_span.CpuSpan();

  size_t packed_offset = 0;
  for (size_t row = 0; row < scheduled_requests.size(); ++row) {
    const auto& request = scheduled_requests[row];
    const size_t token_count = request->ScheduledTokenCount();
    if (plan) {
      const auto& entry = plan->requests[row];
      if (entry.request != request ||
          entry.packed_token_offset != packed_offset ||
          entry.unprocessed_token_count != token_count) {
        throw std::logic_error(
            "Step plan token layout does not match packed position_ids.");
      }
    }
    const int64_t first_position = request->ProcessedSequenceLength();
    for (size_t token = 0; token < token_count; ++token) {
      const int64_t position =
          first_position + static_cast<int64_t>(token);
      for (size_t plane = 0; plane < position_planes_; ++plane) {
        position_cpu[plane * num_tokens + packed_offset + token] = position;
      }
    }
    packed_offset += token_count;
  }
  position_span.CopyCpuToDevice();

  input_names_.push_back(position_ids_name.c_str());
  inputs_.push_back(position_ids->GetOrtTensor());
  owned_inputs_.push_back(std::move(position_ids));
}

// PagedAttention accepts an optional `attention_metadata` CPU input holding
// [max_query_len_bound, max_kv_len_bound, max_kv_len_lower_bound]. The first two values are upper
// bounds and the third value is a lower bound on the longest live KV sequence. The operator
// uses them only to select a backend and to size its launch dimensions and workspaces, never as a
// mask boundary. Supplying them lets the operator skip the device-to-host readback of the sequence
// lengths, which otherwise forces a full stream synchronization inside every attention node.
//
// The engine already has both quantities on the host while it builds the sequence length inputs, so
// this costs nothing. Eager bounds are exact. Captured bounds describe the entire graph bucket so
// they remain valid when the graph is replayed for different requests.
void VarlenDecoderIO::PrepareAttentionMetadata(std::shared_ptr<DecoderOnly_Model> model, ScheduledRequests& scheduled_requests) {
  const std::string& metadata_name = model->config_->model.decoder.inputs.attention_metadata;
  if (!model->session_info_.HasInput(metadata_name)) {
    // Model was built before `attention_metadata` existed. The operator falls back to the readback.
    return;
  }

  AttentionMetadataValues metadata;
  if (graph_buffers_ != nullptr) {
    if (plan_ == nullptr) {
      throw std::runtime_error("Captured attention metadata requires a step plan.");
    }
    metadata = GetAttentionMetadataForGraphStep(
        *plan_,
        block_table_columns_,
        model->config_->engine.dynamic_batching->block_size);
  } else if (plan_) {
    metadata = GetAttentionMetadataForPlan(*plan_);
  } else {
    for (auto& request : scheduled_requests) {
      const int32_t query_len =
          CheckedMetadataLength(request->ScheduledTokenCount(), "Query length");
      const int32_t kv_len =
          CheckedMetadataLength(request->ProcessedSequenceLength() + query_len, "KV length");
      metadata.max_query_len_bound = std::max(metadata.max_query_len_bound, query_len);
      metadata.max_kv_len_bound = std::max(metadata.max_kv_len_bound, kv_len);
    }
    metadata.max_kv_len_lower_bound = metadata.max_kv_len_bound;
  }

  const auto packed_metadata = PackAttentionMetadata(metadata);
  auto metadata_tensor = std::make_unique<Tensor>(GetDeviceInterface(DeviceType::CPU), Ort::TypeToTensorType<int32_t>);
  metadata_tensor->CreateTensor(std::vector<int64_t>{static_cast<int64_t>(packed_metadata.size())});
  auto metadata_span = metadata_tensor->GetDeviceSpan<int32_t>().CpuSpan();
  std::copy(packed_metadata.begin(), packed_metadata.end(), metadata_span.begin());

  input_names_.push_back(metadata_name.c_str());
  inputs_.push_back(metadata_tensor->GetOrtTensor());
  owned_inputs_.push_back(std::move(metadata_tensor));
}

void VarlenDecoderIO::PrepareLogits(std::shared_ptr<DecoderOnly_Model> model, ScheduledRequests& scheduled_requests) {
  size_t logits_rows = scheduled_requests.size();
  if (logits_are_per_token_) {
    const StepPlan* plan = plan_;
    logits_rows =
        plan ? plan->token_count
             : std::accumulate(scheduled_requests.begin(), scheduled_requests.end(), size_t{0},
                               [](size_t sum, const std::shared_ptr<Request>& request) {
                                 return sum + request->ScheduledTokenCount();
                               });
  }
  const std::vector<int64_t> logits_shape = {
      static_cast<int64_t>(logits_rows),
      static_cast<int64_t>(model->config_->model.vocab_size)};
  if (graph_buffers_ != nullptr) {
    graph_buffers_->logits->CreateTensor(logits_shape, /*make_static=*/true);
    active_logits_ = graph_buffers_->logits.get();
  } else {
    logits_ = std::make_unique<Tensor>(
        model->p_device_inputs_,
        model->session_info_.GetOutputDataType(model->config_->model.decoder.outputs.logits));
    logits_->CreateTensor(logits_shape);
    active_logits_ = logits_.get();
  }

  output_names_.push_back(model->config_->model.decoder.outputs.logits.c_str());
  outputs_.push_back(active_logits_->GetOrtTensor());
}

std::vector<DeviceSpan<float>> VarlenDecoderIO::ProcessLogits() {
  std::vector<size_t> valid_token_indices(scheduled_requests_.size());
  if (logits_are_per_token_) {
    if (plan_) {
      const auto& plan = *plan_;
      if (plan.requests.size() != scheduled_requests_.size()) {
        throw std::runtime_error("Step plan size does not match logits batch size.");
      }
      for (size_t i = 0; i < plan.requests.size(); ++i) {
        if (plan.requests[i].request != scheduled_requests_[i]) {
          throw std::runtime_error("Step plan order does not match logits batch order.");
        }
        valid_token_indices[i] = plan.requests[i].logits_row_index;
      }
    } else {
      for (size_t i = 0, running_length = 0; i < scheduled_requests_.size(); ++i) {
        valid_token_indices[i] = running_length + scheduled_requests_[i]->ScheduledTokenCount() - 1;
        running_length += scheduled_requests_[i]->ScheduledTokenCount();
      }
    }
  } else {
    for (size_t i = 0; i < scheduled_requests_.size(); ++i) {
      valid_token_indices[i] = i;
    }
  }

  // The output shape is either [batch_size, vocab_size] or [num_tokens, vocab_size].
  const auto active_logits_shape = active_logits_->GetShape();
  const int64_t vocab_size = active_logits_shape[1];
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

  // Per-request logits occupy contiguous rows. Per-token logits do too on pure decode steps because
  // every request contributes exactly one token. Convert the whole batch in one launch in either case.
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
