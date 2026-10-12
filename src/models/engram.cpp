#include "generator/generators.h"
#include "engram.h"
#include "utils.h"

namespace Generators {

EngramState::EngramState(const Model& model, OrtSession& session, const GeneratorParams& params)
    : State{params, model}, model_{model}, session_{session} {
  if (params.search.num_beams != 1) {
    throw std::runtime_error("Engram-backed models currently support num_beams=1 only");
  }
  input_ids_index_ = inputs_.size();
  inputs_.push_back(nullptr);
  input_names_.push_back(model_.config_->model.engram.inputs.input_ids.c_str());
  embeddings_type_ = model_.session_info_.GetOutputDataType(model_.config_->model.engram.outputs.embeddings);
  const auto embeddings_shape = model_.session_info_.GetOutputShape(model_.config_->model.engram.outputs.embeddings);
  if (embeddings_shape.empty() || embeddings_shape.back() <= 0) {
    throw std::runtime_error("Engram embeddings must have a static width");
  }
  embeddings_width_ = embeddings_shape.back();
  embeddings_output_index_ = outputs_.size();
  outputs_.push_back(nullptr);
  output_names_.push_back(model_.config_->model.engram.outputs.embeddings.c_str());

  auto token_shape = model_.session_info_.GetInputShape(model_.config_->model.engram.inputs.past_tokens);
  if (token_shape.empty()) {
    throw std::runtime_error("Engram token history must have a batch dimension");
  }
  if (token_shape[0] <= 0) token_shape[0] = params_->BatchBeamSize();
  for (int64_t dimension : token_shape) {
    if (dimension <= 0) {
      throw std::runtime_error("Engram token history must have a static shape");
    }
  }

  past_tokens_ = OrtValue::CreateTensor(model_.allocator_cpu_, token_shape, ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64);
  present_tokens_ = OrtValue::CreateTensor(model_.allocator_cpu_, token_shape, ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64);
  InitializeTokens(*past_tokens_);
  InitializeTokens(*present_tokens_);
  past_input_index_ = inputs_.size();
  inputs_.push_back(past_tokens_.get());
  input_names_.push_back(model_.config_->model.engram.inputs.past_tokens.c_str());
  present_output_index_ = outputs_.size();
  outputs_.push_back(present_tokens_.get());
  output_names_.push_back(model_.config_->model.engram.outputs.present_tokens.c_str());
}

void EngramState::UpdateInputsOutputs(DeviceSpan<int32_t>& next_tokens) {
  const int64_t batch_size = params_->BatchBeamSize();
  const int64_t sequence_length = static_cast<int64_t>(next_tokens.size()) / batch_size;
  input_ids_ = OrtValue::CreateTensor(model_.allocator_cpu_,
      std::array<int64_t, 2>{batch_size, sequence_length}, ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64);
  const auto tokens_cpu = next_tokens.CpuSpan();
  std::transform(tokens_cpu.begin(), tokens_cpu.end(), input_ids_->GetTensorMutableData<int64_t>(),
                 [](int32_t token) { return static_cast<int64_t>(token); });
  inputs_[input_ids_index_] = input_ids_.get();
  embeddings_output_ = OrtValue::CreateTensor(model_.allocator_cpu_,
      std::array<int64_t, 3>{batch_size, sequence_length, embeddings_width_}, embeddings_type_);
  outputs_[embeddings_output_index_] = embeddings_output_.get();
  if (!first_run_) {
    std::swap(past_tokens_, present_tokens_);
    inputs_[past_input_index_] = past_tokens_.get();
    outputs_[present_output_index_] = present_tokens_.get();
  }

  cache_hit_ = false;
  active_cache_key_.reset();
  if (sequence_length == 1 && model_.config_->model.engram.cache_capacity > 0) {
    std::vector<int64_t> key;
    const auto past_info = past_tokens_->GetTensorTypeAndShapeInfo();
    const size_t past_count = past_info->GetElementCount();
    const auto* past_data = past_tokens_->GetTensorData<int64_t>();
    key.assign(past_data, past_data + past_count);
    key.push_back(input_ids_->GetTensorData<int64_t>()[0]);
    auto cached = cache_.find(key);
    if (cached != cache_.end()) {
      const auto output_info = embeddings_output_->GetTensorTypeAndShapeInfo();
      const size_t output_bytes = output_info->GetElementCount() * Ort::SizeOf(output_info->GetElementType());
      if (cached->second.embeddings.size() == output_bytes &&
          cached->second.present_tokens.size() == present_tokens_->GetTensorTypeAndShapeInfo()->GetElementCount()) {
        std::memcpy(embeddings_output_->GetTensorMutableRawData(), cached->second.embeddings.data(), output_bytes);
        std::copy(cached->second.present_tokens.begin(), cached->second.present_tokens.end(),
                  present_tokens_->GetTensorMutableData<int64_t>());
        cache_hit_ = true;
      }
    }
    active_cache_key_ = std::move(key);
  }
}

DeviceSpan<float> EngramState::Run(int current_length, DeviceSpan<int32_t>& next_tokens,
                                    DeviceSpan<int32_t> next_indices) {
  (void)current_length;
  (void)next_tokens;
  (void)next_indices;
  if (model_.config_->model.engram.run_options.has_value()) {
    State::SetRunOptions(model_.config_->model.engram.run_options.value());
  }
  if (!cache_hit_) {
    State::Run(session_);
    if (active_cache_key_.has_value()) {
      if (cache_.size() >= model_.config_->model.engram.cache_capacity) cache_.clear();
      CacheEntry entry;
      const auto output_info = embeddings_output_->GetTensorTypeAndShapeInfo();
      const size_t output_bytes = output_info->GetElementCount() * Ort::SizeOf(output_info->GetElementType());
      const auto* output_data = static_cast<const uint8_t*>(embeddings_output_->GetTensorRawData());
      entry.embeddings.assign(output_data, output_data + output_bytes);
      const auto* present_data = present_tokens_->GetTensorData<int64_t>();
      const size_t present_count = present_tokens_->GetTensorTypeAndShapeInfo()->GetElementCount();
      entry.present_tokens.assign(present_data, present_data + present_count);
      cache_[*active_cache_key_] = std::move(entry);
    }
  }
  first_run_ = false;
  return {};
}

void EngramState::CopyEmbeddingsTo(Embeddings& destination) {
  if (!destination.Get()) throw std::runtime_error("Engram embeddings are not allocated");
  CopyEmbeddingsTo(*destination.Get());
}

void EngramState::CopyEmbeddingsTo(OrtValue& destination) {
  if (!embeddings_output_) throw std::runtime_error("Engram embeddings are not allocated");
  const auto source_info = embeddings_output_->GetTensorTypeAndShapeInfo();
  const auto destination_info = destination.GetTensorTypeAndShapeInfo();
  if (source_info->GetElementType() != destination_info->GetElementType() ||
      source_info->GetElementCount() != destination_info->GetElementCount()) {
    throw std::runtime_error("Engram output must match the decoder Engram input type and shape");
  }
  ByteWrapTensor(*model_.p_device_inputs_, destination).CopyFrom(
      ByteWrapTensor(*GetDeviceInterface(DeviceType::CPU), *embeddings_output_));
}

void EngramState::RewindTo(size_t index) {
  if (index == 0) {
    InitializeTokens(*past_tokens_);
    InitializeTokens(*present_tokens_);
    snapshot_valid_ = false;
    first_run_ = true;
    inputs_[past_input_index_] = past_tokens_.get();
    outputs_[present_output_index_] = present_tokens_.get();
    return;
  }
  if (!snapshot_valid_ || snapshot_position_ != index) {
    throw std::runtime_error("EngramState cannot rewind to position " + std::to_string(index) +
                             " without a matching snapshot");
  }
  CopyTokens(*snapshot_tokens_, *present_tokens_);
  if (std::getenv("ORTGENAI_MTP_DEBUG_STATE") != nullptr) {
    const auto* expected = snapshot_tokens_->GetTensorData<int64_t>();
    const auto* actual = present_tokens_->GetTensorData<int64_t>();
    const size_t count = snapshot_tokens_->GetTensorTypeAndShapeInfo()->GetElementCount();
    if (!std::equal(expected, expected + count, actual)) {
      throw std::runtime_error("EngramState snapshot restore mismatch");
    }
  }
}

void EngramState::SnapshotState(size_t position) {
  if (!snapshot_tokens_) {
    auto shape = present_tokens_->GetTensorTypeAndShapeInfo()->GetShape();
    snapshot_tokens_ = OrtValue::CreateTensor(model_.allocator_cpu_, shape, ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64);
  }
  CopyTokens(*present_tokens_, *snapshot_tokens_);
  snapshot_position_ = position;
  snapshot_valid_ = true;
}

void EngramState::CommitAcceptedPrefix(size_t token_count) {
  if (!snapshot_valid_ || !snapshot_tokens_ || !input_ids_) {
    throw std::runtime_error("EngramState accepted-prefix commit requires a snapshot and input tokens");
  }
  const auto input_info = input_ids_->GetTensorTypeAndShapeInfo();
  const auto input_shape = input_info->GetShape();
  if (input_shape.size() != 2 || input_shape[0] != 1 || token_count > static_cast<size_t>(input_shape[1])) {
    throw std::runtime_error("EngramState accepted-prefix commit received an invalid token count");
  }
  CopyTokens(*snapshot_tokens_, *present_tokens_);
  auto* history = present_tokens_->GetTensorMutableData<int64_t>();
  const size_t history_size = present_tokens_->GetTensorTypeAndShapeInfo()->GetElementCount();
  const auto* tokens = input_ids_->GetTensorData<int64_t>();
  for (size_t token_index = 0; token_index < token_count; ++token_index) {
    if (history_size > 1) std::move(history + 1, history + history_size, history);
    history[history_size - 1] = tokens[token_index];
  }
}

void EngramState::InitializeTokens(OrtValue& tokens) {
  auto* data = tokens.GetTensorMutableData<int64_t>();
  const size_t count = tokens.GetTensorTypeAndShapeInfo()->GetElementCount();
  std::fill_n(data, count, model_.config_->model.decoder.ple_token_pad_id);
}

void EngramState::CopyTokens(OrtValue& source, OrtValue& destination) {
  const size_t bytes = source.GetTensorTypeAndShapeInfo()->GetElementCount() * sizeof(int64_t);
  std::memcpy(destination.GetTensorMutableData<int64_t>(), source.GetTensorData<int64_t>(), bytes);
}

}  // namespace Generators