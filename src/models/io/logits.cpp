// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
//
// Modifications Copyright(C) 2026 Advanced Micro Devices, Inc. All rights reserved.
#include "generator/generators.h"
#include "models/model.h"
#include "models/io/logits.h"
#include "ep/openvino/interface.h"
#include <limits>

namespace Generators {

std::vector<int32_t> LastTokenIndices(std::span<const int32_t> tokens, size_t batch_size,
                                      size_t num_beams, size_t sequence_length, int32_t pad_token_id,
                                      bool allow_padding_only) {
  if (!batch_size || !num_beams || !sequence_length ||
      sequence_length > static_cast<size_t>(std::numeric_limits<int32_t>::max()) ||
      batch_size > tokens.size() / sequence_length ||
      batch_size > std::numeric_limits<size_t>::max() / num_beams)
    throw std::invalid_argument("Invalid last-token index dimensions or token count");
  std::vector<int32_t> indices(batch_size * num_beams);
  for (size_t batch = 0; batch < batch_size; ++batch) {
    size_t end = sequence_length;
    while (end && tokens[batch * sequence_length + end - 1] == pad_token_id)
      --end;
    if (!end && !allow_padding_only)
      throw std::invalid_argument("Last-token prefill requires a non-padding token in each current chunk");
    std::fill_n(indices.begin() + batch * num_beams, num_beams, static_cast<int32_t>(end ? end - 1 : 0));
  }
  return indices;
}

Logits::Logits(State& state)
    : state_{state},
      shape_{static_cast<int64_t>(state_.params_->search.batch_size) * state_.params_->search.num_beams, 0,
             model_.config_->model.vocab_size},
      type_{model_.session_info_.GetOutputDataType(model_.config_->model.decoder.outputs.logits)} {
  if (state_.params_->search.batch_size < 1 || state_.params_->search.num_beams < 1 ||
      shape_[0] > std::numeric_limits<int>::max())
    throw std::invalid_argument("Logits requires a positive, representable batch and beam count");

  output_raw_ = std::make_unique<Tensor>(model_.p_device_logits_, type_);

  input_sequence_lengths.resize(state_.params_->search.batch_size);

  if (IsOpenVINOStatefulModel(state.model_) || state.model_.IsPruned()) {
    // In the case of OpenVINO stateful models, or any model whose ONNX graph
    // has been patched to only output last-token logits (logits dim[1]==1), they only return the
    // sliced logits needed for sampling. For example, given 43 prompt tokens, instead of returning
    // logits of the shape:  [1,43,<vocab_size>]
    // they will have shape: [1, 1,<vocab_size>].
    if (g_log.enabled)
      Log("info", "Logits: Using Trimmed Prefill Logits");

    trimmed_prefill_logits_ = true;
  }
}

DeviceSpan<float> Logits::Get(bool all_tokens_valid) {
  size_t element_count = shape_[0] * shape_[1] * shape_[2];

  // The model's output logits are {batch_size*num_beams, input_seq_len, vocab_size}
  OrtValue* logits_of_last_token = output_raw_->GetOrtTensor();
  std::array<int64_t, 3> shape_last{shape_[0], 1, shape_[2]};
  if (shape_[1] != 1) {
    const size_t seq_length = shape_[1];
    const size_t vocab_size = shape_[2];
    const size_t num_beams = state_.params_->search.num_beams;

    // create new OrtValue for logits_of_last_token and use output_last_tokens_ to hold it
    output_last_tokens_ = OrtValue::CreateTensor(model_.p_device_logits_->GetAllocator(), shape_last, type_);

    if (type_ == Ort::TypeToTensorType<Ort::Float16_t> || type_ == Ort::TypeToTensorType<Ort::BFloat16_t>)
      logits_of_last_token_fp32_ = OrtValue::CreateTensor<float>(model_.p_device_logits_->GetAllocator(), shape_last);

    logits_of_last_token = output_last_tokens_.get();

    size_t element_size = Ort::SizeOf(type_);
    size_t vocab_index = 0;  // Simpler math to have this index go up by vocab_size for every logit chunk we process

    auto logits_raw = output_raw_->GetByteSpan();
    auto logits_last_tokens = ByteWrapTensor(*model_.p_device_logits_, *logits_of_last_token);

    for (int batch_index = 0; batch_index < state_.params_->search.batch_size; batch_index++) {
      // Replayed committed tokens are valid even when their ID equals PAD.
      size_t token_index = all_tokens_valid ? seq_length - 1 : input_sequence_lengths[batch_index] - 1;
      for (int beam_index = 0; beam_index < num_beams; beam_index++) {
        auto target = logits_last_tokens.subspan(vocab_index * element_size, vocab_size * element_size);
        auto source = logits_raw.subspan((vocab_index * seq_length + token_index * vocab_size) * element_size, vocab_size * element_size);
        target.CopyFrom(source);
        vocab_index += vocab_size;
      }
    }

    element_count = shape_[0] * shape_[2];  // shape_[1] is now 1, so the element count must be updated
  }

  // Convert from float16/bfloat16 to float32 if necessary
  if (type_ == Ort::TypeToTensorType<Ort::Float16_t> || type_ == Ort::TypeToTensorType<Ort::BFloat16_t>) {
    Cast(*logits_of_last_token, logits_of_last_token_fp32_, *model_.p_device_logits_, Ort::TypeToTensorType<float>);
    logits_of_last_token = logits_of_last_token_fp32_.get();
  }

  if (logits_.empty() || logits_of_last_token->GetTensorMutableRawData() != logits_.Span().data())
    logits_ = WrapTensor<float>(*model_.p_device_logits_, *logits_of_last_token);

  return logits_;
}

void Logits::Update(const DeviceSpan<int32_t>& next_tokens, size_t new_kv_length,
                    bool all_tokens_valid, bool is_final_chunk) {
  if (token_indices_ &&
      (!output_raw_->ort_tensor_ || new_kv_length > 1 ||
       std::any_of(input_sequence_lengths.begin(), input_sequence_lengths.end(), [](int n) { return n != 1; }))) {
    if (!new_kv_length || new_kv_length > static_cast<size_t>(std::numeric_limits<int32_t>::max()))
      throw std::invalid_argument("Invalid last-token sequence length");
    const auto batch_size = static_cast<size_t>(state_.params_->search.batch_size);
    const auto num_beams = static_cast<size_t>(state_.params_->search.num_beams);
    const auto batch_beams = static_cast<size_t>(shape_[0]);
    std::vector<int32_t> indices(batch_beams, all_tokens_valid ? static_cast<int32_t>(new_kv_length - 1) : 0);
    if (new_kv_length > 1 && !all_tokens_valid) {
      auto token_span = next_tokens;
      auto tokens = token_span.CopyDeviceToCpu();
      const size_t rows = tokens.size() / new_kv_length;
      if (tokens.size() % new_kv_length || (rows != batch_size && rows != batch_beams))
        throw std::invalid_argument("Last-token indices require batch or batch-beam token rows");
      indices = LastTokenIndices(tokens, rows, rows == batch_size ? num_beams : 1,
                                 new_kv_length, model_.config_->model.pad_token_id, !is_final_chunk);
    }
    for (size_t batch = 0; batch < batch_size; ++batch)
      input_sequence_lengths[batch] = indices[batch * num_beams] + 1;
    auto host = OrtValue::CreateTensor<int32_t>(model_.allocator_cpu_,
                                                std::array<int64_t, 3>{shape_[0], 1, 1});
    std::copy(indices.begin(), indices.end(), host->GetTensorMutableData<int32_t>());
    token_indices_->GetByteSpan().CopyFrom(
        ByteWrapTensor(model_.p_device_inputs_->GetCpuFallbackDevice(), *host));
  }
  if (trimmed_prefill_logits_) {
    new_kv_length = 1;
  }

  if (output_raw_->ort_tensor_ && static_cast<size_t>(output_raw_->GetShape()[1]) == new_kv_length && new_kv_length == 1) {
    return;
  }

  auto token_span = next_tokens;
  // Store length of input sequence for each batch for the get step
  for (int b = 0; !trimmed_prefill_logits_ && b < state_.params_->search.batch_size; b++) {
    // Find the first non pad token from the end
    size_t token_index = new_kv_length;
    while (token_index-- > 0) {
      auto next_token = token_span.CpuSpan()[b * new_kv_length + token_index];
      if (next_token != model_.config_->model.pad_token_id)
        break;
    }
    input_sequence_lengths[b] = static_cast<int>(token_index + 1);
  }

  if (output_raw_->ort_tensor_ && static_cast<size_t>(output_raw_->GetShape()[1]) == new_kv_length) {
    return;
  }

  shape_[1] = new_kv_length;
  const int max_cap = state_.params_->max_graph_capture_length;
  const bool use_static = state_.params_->use_graph_capture && shape_[1] >= 1 && shape_[1] <= max_cap;
  const size_t static_cap_bytes = use_static ? static_cast<size_t>(shape_[0]) * max_cap * shape_[2] * Ort::SizeOf(type_) : 0;
  output_raw_->CreateTensor(shape_, use_static, static_cap_bytes);
  state_.outputs_[output_index_] = output_raw_->GetOrtTensor();
}

void Logits::Add() {
  if (model_.session_info_.HasInput("last_token_indices")) {
    if (!trimmed_prefill_logits_ ||
        model_.session_info_.GetInputDataType("last_token_indices") != Ort::TypeToTensorType<int32_t>)
      throw std::runtime_error("last_token_indices requires pruned logits and int32 indices");
    token_indices_ = std::make_unique<Tensor>(model_.p_device_inputs_, Ort::TypeToTensorType<int32_t>);
    token_indices_->CreateTensor(std::array<int64_t, 3>{shape_[0], 1, 1}, state_.params_->use_graph_capture);
    state_.input_names_.push_back("last_token_indices");
    state_.inputs_.push_back(token_indices_->GetOrtTensor());
  }
  output_index_ = state_.outputs_.size();

  state_.output_names_.push_back(model_.config_->model.decoder.outputs.logits.c_str());
  state_.outputs_.push_back(output_raw_->GetOrtTensor());
}

}  // namespace Generators
