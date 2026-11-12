// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "generator/generators.h"
#include "lfm2.h"

namespace Generators {

namespace {

// Row `row` of a [rows, ...] tensor as a batch-1 view of the same memory, with dim 1 cut to `length` when given.
std::unique_ptr<OrtValue> RowView(OrtValue& batch, size_t row, std::optional<int64_t> length = {}) {
  const auto info = batch.GetTensorTypeAndShapeInfo();
  const auto type = info->GetElementType();
  auto shape = info->GetShape();
  const size_t row_bytes = info->GetElementCount() / shape[0] * Ort::SizeOf(type);
  shape[0] = 1;
  if (length) {
    shape[1] = std::min(shape[1], *length);
  }
  auto* data = static_cast<uint8_t*>(batch.GetTensorMutableRawData()) + row * row_bytes;
  const size_t bytes = ElementCountFromShape(shape) * Ort::SizeOf(type);
  return OrtValue::CreateTensor(batch.GetTensorMemoryInfo(), data, bytes, shape, type);
}

// Copies a row's [1, H, n, D] KV present into the first n steps of that row of a [rows, H, steps, D] present.
void CopyIntoRow(DeviceInterface& device, OrtValue& row_present, OrtValue& batch, size_t row) {
  const auto info = row_present.GetTensorTypeAndShapeInfo();
  const auto shape = info->GetShape();
  const size_t step_bytes = shape[3] * Ort::SizeOf(info->GetElementType());
  const size_t head_bytes = shape[2] * step_bytes;
  const size_t batch_head_bytes = batch.GetTensorTypeAndShapeInfo()->GetShape()[2] * step_bytes;
  auto source = ByteWrapTensor(device, row_present);
  auto target = ByteWrapTensor(device, batch);
  for (int64_t head = 0; head < shape[1]; ++head) {
    target.subspan((row * shape[1] + head) * batch_head_bytes, head_bytes)
        .CopyFrom(source.subspan(head * head_bytes, head_bytes));
  }
}

// Puts the batch bindings back, also when a run throws.
struct RestoreBindings {
  std::vector<OrtValue*>& bindings;
  const std::vector<OrtValue*> batch{bindings};
  ~RestoreBindings() { bindings = batch; }
};

}  // namespace

LFM2_Model::LFM2_Model(std::unique_ptr<Config> config, OrtEnv& ort_env)
    : Model{std::move(config)} {
  session_decoder_ = CreateSession(ort_env, config_->model.decoder.filename, session_options_.get());
  session_info_.Add(*session_decoder_);
}

std::unique_ptr<State> LFM2_Model::CreateState(DeviceSpan<int32_t> sequence_lengths_unk, const GeneratorParams& params) const {
  return std::make_unique<LFM2_State>(*this, sequence_lengths_unk, params);
}

LFM2_State::LFM2_State(const LFM2_Model& model, DeviceSpan<int32_t> sequence_lengths_unk, const GeneratorParams& params)
    : State{params, model},
      model_{model},
      cache_(model_.p_device_kvcache_->CreateKeyValueCache(*this)),
      position_inputs_{model_.p_device_inputs_->CreatePositionInputs(*this, sequence_lengths_unk, model_.config_->model.decoder.inputs.attention_mask)} {
  input_ids_.Add();
  position_inputs_->Add();
  logits_.Add();
  cache_->Add();
  extra_inputs_begin_ = inputs_.size();
}

void LFM2_State::SetExtraInputs(const std::vector<ExtraInput>& extra_inputs) {
  extra_inputs_.Add(extra_inputs, model_.session_decoder_->GetInputNames());
}

DeviceSpan<float> LFM2_State::Run(int total_length, DeviceSpan<int32_t>& next_tokens, DeviceSpan<int32_t> next_indices) {
  UpdateInputsOutputs(next_tokens, next_indices, total_length);
  if (model_.config_->model.decoder.run_options.has_value()) {
    State::SetRunOptions(model_.config_->model.decoder.run_options.value());
  }

  const auto row_lengths = PaddedRowLengths(next_tokens);
  if (row_lengths.empty()) {
    bool graph_capture_this_run = params_->use_graph_capture && input_ids_.GetShape()[1] == 1;
    State::Run(*model_.session_decoder_, graph_capture_this_run);
  } else {
    RunRowsAlone(row_lengths);
  }

  return logits_.Get();
}

// Each row's real length when the first prompt is right-padded and the export has window-sized KV caches, else nothing.
// As for the attention mask and Logits::Update, trailing pad tokens count as padding, even where the prompt has them.
std::vector<int64_t> LFM2_State::PaddedRowLengths(DeviceSpan<int32_t>& next_tokens) {
  const auto& search = params_->search;
  // A batch of one gets an all-ones attention mask, even with trailing pad tokens.
  if (!first_run_ || search.batch_size == 1 || !model_.config_->model.decoder.sliding_window) {
    return {};
  }
  const auto tokens = next_tokens.CpuSpan();
  const int64_t steps = input_ids_.GetShape()[1];
  std::vector<int64_t> lengths;
  for (int row = 0; row < search.batch_size; ++row) {
    int64_t length = steps;
    while (length > 0 && tokens[row * steps + length - 1] == model_.config_->model.pad_token_id) {
      --length;
    }
    if (length == 0) {
      // An empty row has nothing to run alone. TODO: batches of one and exports without window-sized caches still
      // reach Logits::Update with such a row, which reads before the logits buffer.
      throw std::runtime_error("LFM2: row " + std::to_string(row) + " of the batch has only pad tokens.");
    }
    lengths.insert(lengths.end(), search.num_beams, length);
  }
  if (std::all_of(lengths.begin(), lengths.end(), [steps](int64_t length) { return length == steps; })) {
    return {};
  }
  return lengths;
}

// ORT's windowed GroupQueryAttention takes every token of a step as real, so each row runs alone, into its row of the
// batch buffers.
void LFM2_State::RunRowsAlone(std::span<const int64_t> lengths) {
  if (!model_.p_device_kvcache_->SupportsOffsetTensorViews()) {
    throw std::runtime_error("LFM2: padded batches with window-sized KV caches need offset tensor views, which " +
                             to_string(model_.p_device_kvcache_->GetType()) + " does not support.");
  }
  if (inputs_.size() != extra_inputs_begin_ || outputs_.size() != model_.session_decoder_->GetOutputNames().size()) {
    throw std::runtime_error("LFM2: padded batches with window-sized KV caches take no extra model inputs or outputs.");
  }

  const auto& decoder = model_.config_->model.decoder;
  const auto has_steps = [&decoder](const char* name) {
    return decoder.inputs.input_ids == name || decoder.inputs.attention_mask == name || decoder.outputs.logits == name;
  };
  const RestoreBindings inputs{inputs_}, outputs{outputs_};
  for (size_t row = 0; row < lengths.size(); ++row) {
    // A shared or windowed KV buffer is bound as past and present, so both take the same view.
    std::unordered_map<OrtValue*, std::unique_ptr<OrtValue>> views;
    const auto view = [&](OrtValue* batch, const char* name) {
      auto& row_view = views[batch];
      if (!row_view) {
        row_view = has_steps(name) ? RowView(*batch, row, lengths[row]) : RowView(*batch, row);
      }
      return row_view.get();
    };
    for (size_t i = 0; i < inputs_.size(); ++i) {
      inputs_[i] = view(inputs.batch[i], input_names_[i]);
    }
    std::vector<std::pair<OrtValue*, std::unique_ptr<OrtValue>>> presents;
    for (size_t i = 0; i < outputs_.size(); ++i) {
      OrtValue* batch = outputs.batch[i];
      const auto info = batch->GetTensorTypeAndShapeInfo();
      auto shape = info->GetShape();
      // A KV present apart from its past is [rows, H, steps, D], where a row's steps are not contiguous.
      if (views.count(batch) == 0 && shape.size() == 4) {
        shape[0] = 1;
        shape[2] = lengths[row];
        auto present = OrtValue::CreateTensor(model_.p_device_kvcache_->GetAllocator(), shape, info->GetElementType());
        outputs_[i] = presents.emplace_back(batch, std::move(present)).second.get();
      } else {
        outputs_[i] = view(batch, output_names_[i]);
      }
    }

    State::Run(*model_.session_decoder_);

    for (auto& [batch, present] : presents) {
      CopyIntoRow(*model_.p_device_kvcache_, *present, *batch, row);
    }
  }
}

void LFM2_State::RewindTo(size_t index) {
  throw std::runtime_error("LFM2 does not support RewindTo. Conv state requires replaying all prior tokens.");
}

void LFM2_State::UpdateInputsOutputs(DeviceSpan<int32_t>& next_tokens, DeviceSpan<int32_t> beam_indices, int total_length) {
  input_ids_.Update(next_tokens);
  size_t new_length = static_cast<size_t>(input_ids_.GetShape()[1]);
  position_inputs_->Update(next_tokens, total_length, static_cast<int>(new_length));
  cache_->Update(beam_indices, total_length);
  logits_.Update(next_tokens, new_length);
}

}  // namespace Generators
