// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#include "../generators.h"
#include "model.h"
#include "logits.h"
#include "../openvino/interface.h"

namespace Generators {

Logits::Logits(State& state)
    : state_{state},
      shape_{static_cast<int64_t>(state_.params_->BatchBeamSize()), 0, model_.config_->model.vocab_size},
      type_{model_.session_info_.GetOutputDataType(model_.config_->model.decoder.outputs.logits)} {
  output_raw_ = std::make_unique<Tensor>(model_.p_device_inputs_, type_);

  input_sequence_lengths.resize(state_.params_->search.batch_size);

  if (IsOpenVINOStatefulModel(state.model_)) {
    // In the case of OpenVINO stateful models, they are patched in a way so that they only return the
    // sliced logits needed for sampling. For example, given 43 prompt tokens, instead of returning
    // logits of the shape:  [1,43,<vocab_size>]
    // they will have shape: [1, 1,<vocab_size>].
    if (g_log.enabled)
      Log("info", "Logits: Using Trimmed Prefill Logits");

    trimmed_prefill_logits_ = true;
  }
}

DeviceSpan<float> Logits::Get() {
  size_t element_count = shape_[0] * shape_[1] * shape_[2];

  // The model's output logits are {batch_size*num_beams, input_seq_len, vocab_size}
  OrtValue* logits_of_last_token = output_raw_->GetOrtTensor();
  std::array<int64_t, 3> shape_last{shape_[0], 1, shape_[2]};
  if (shape_[1] != 1) {
    const size_t seq_length = shape_[1];
    const size_t vocab_size = shape_[2];
    const size_t num_beams = state_.params_->search.num_beams;

    // create new OrtValue for logits_of_last_token and use output_last_tokens_ to hold it
    output_last_tokens_ = OrtValue::CreateTensor(model_.p_device_inputs_->GetAllocator(), shape_last, type_);

    if (type_ == Ort::TypeToTensorType<Ort::Float16_t>)
      logits_of_last_token_fp32_ = OrtValue::CreateTensor<float>(model_.p_device_inputs_->GetAllocator(), shape_);

    logits_of_last_token = output_last_tokens_.get();

    size_t element_size = Ort::SizeOf(type_);
    size_t vocab_index = 0;  // Simpler math to have this index go up by vocab_size for every logit chunk we process

    auto logits_raw = output_raw_->GetByteSpan();
    auto logits_last_tokens = ByteWrapTensor(*model_.p_device_inputs_, *logits_of_last_token);

    for (int batch_index = 0; batch_index < state_.params_->search.batch_size; batch_index++) {
      // Find the first non pad token from the end
      size_t token_index = input_sequence_lengths[batch_index] - 1;
      for (int beam_index = 0; beam_index < num_beams; beam_index++) {
        auto target = logits_last_tokens.subspan(vocab_index * element_size, vocab_size * element_size);
        auto source = logits_raw.subspan((vocab_index * seq_length + token_index * vocab_size) * element_size, vocab_size * element_size);
        target.CopyFrom(source);
        vocab_index += vocab_size;
      }
    }

    element_count = shape_[0] * shape_[2];  // shape_[1] is now 1, so the element count must be updated
  }

  // Convert from float16 to float32 if necessary
  if (type_ == Ort::TypeToTensorType<Ort::Float16_t>) {
    Cast(*logits_of_last_token, logits_of_last_token_fp32_, *model_.p_device_inputs_, Ort::TypeToTensorType<float>);
    logits_of_last_token = logits_of_last_token_fp32_.get();
  }

  if (logits_.empty() || logits_of_last_token->GetTensorMutableRawData() != logits_.Span().data())
    logits_ = WrapTensor<float>(*model_.p_device_inputs_, *logits_of_last_token);

  return logits_;
}

void Logits::Update(const DeviceSpan<int32_t>& next_tokens, size_t new_kv_length) {
  if (trimmed_prefill_logits_) {
    new_kv_length = 1;
  }

  if (output_raw_->ort_tensor_ && static_cast<size_t>(output_raw_->GetShape()[1]) == new_kv_length && new_kv_length == 1) {
    return;
  }

  // Store length of input sequence for each batch for the get step
  for (int b = 0; b < state_.params_->search.batch_size; b++) {
    // Find the first non pad token from the end
    size_t token_index = new_kv_length;
    while (token_index-- > 0) {
      auto next_token = const_cast<DeviceSpan<int32_t>&>(next_tokens).CpuSpan()[b * new_kv_length + token_index];
      if (next_token != model_.config_->model.pad_token_id)
        break;
    }
    input_sequence_lengths[b] = static_cast<int>(token_index + 1);
  }

  if (output_raw_->ort_tensor_ && static_cast<size_t>(output_raw_->GetShape()[1]) == new_kv_length) {
    return;
  }

  shape_[1] = new_kv_length;
  output_raw_->CreateTensor(shape_, state_.params_->use_graph_capture && shape_[1] == 1);
  state_.outputs_[output_index_] = output_raw_->GetOrtTensor();
}

void Logits::Add() {
  output_index_ = state_.outputs_.size();

  state_.output_names_.push_back(model_.config_->model.decoder.outputs.logits.c_str());
  state_.outputs_.push_back(output_raw_->GetOrtTensor());
}

}  // namespace Generators
