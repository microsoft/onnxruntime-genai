// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "../generators.h"
#include "model.h"
#include "kv_cache.h"
#include "threadpool.h"

namespace Generators {

namespace {

std::string ComposeKeyValueName(const std::string& template_string, int index) {
  constexpr int32_t KeyValueNameLength = 64;
  char key_value_name[KeyValueNameLength];
  if (auto length = snprintf(key_value_name, std::size(key_value_name), template_string.c_str(), index);
      length < 0 || length >= KeyValueNameLength) {
    throw std::runtime_error("Unable to compose key value name from the provided template " + template_string +
                             ". This could be either due to an encoding error or the name being too long.");
  }
  return std::string(key_value_name);
}

int64_t ElementCountFromShape(const std::array<int64_t, 4>& shape) {
  return std::accumulate(shape.begin(), shape.end(), int64_t{1}, std::multiplies<int64_t>());
}

bool IsCacheNeeded(const Model& model) {
  return model.session_info_->HasInput(ComposeKeyValueName(model.config_->model.decoder.inputs.past_key_names, 0));
}

}  // namespace

CombinedKeyValueCache::CombinedKeyValueCache(State& state)
    : state_{state},
      layer_count_{model_.config_->model.decoder.num_hidden_layers},
      shape_{2, state_.params_->BatchBeamSize(), model_.config_->model.decoder.num_key_value_heads, 0, model_.config_->model.decoder.head_size} {
  pasts_.resize(layer_count_);
  presents_.reserve(layer_count_);

  for (int i = 0; i < layer_count_; ++i) {
    input_name_strings_.emplace_back(ComposeKeyValueName(model_.config_->model.decoder.inputs.past_names, i));
    output_name_strings_.emplace_back(ComposeKeyValueName(model_.config_->model.decoder.outputs.present_names, i));
  }

  // Derive the KV data type from the KV input 0
  type_ = model_.session_info_->GetInputDataType(input_name_strings_[0]);

  empty_past_ = OrtValue::CreateTensor(*model_.allocator_kvcache_, shape_, type_);
  shape_[3] = 0;

  for (int i = 0; i < layer_count_; ++i) {
    presents_.push_back(OrtValue::CreateTensor(*model_.allocator_kvcache_, shape_, type_));
  }
}

void CombinedKeyValueCache::Add() {
  input_index_ = state_.inputs_.size();
  output_index_ = state_.outputs_.size();

  for (int i = 0; i < layer_count_; i++) {
    state_.inputs_.push_back(empty_past_.get());
    state_.input_names_.push_back(input_name_strings_[i].c_str());
    state_.outputs_.push_back(presents_[i].get());
    state_.output_names_.push_back(output_name_strings_[i].c_str());
  }
}

void CombinedKeyValueCache::Update(DeviceSpan<int32_t> beam_indices, int total_length) {
  assert(state_.params_->search.num_beams == 1 || !beam_indices.empty());  // We require beam_indices if we're a beam search

  if (!is_first_update_) {
    for (int i = 0; i < layer_count_; i++) {
      if (beam_indices.empty()) {
        pasts_[i] = std::move(presents_[i]);
      } else {
        PickPastState(beam_indices, i);
      }
      state_.inputs_[input_index_ + i] = pasts_[i].get();
    }
  }

  shape_[3] = total_length;
  for (int i = 0; i < layer_count_; i++) {
    presents_[i] = OrtValue::CreateTensor(*model_.allocator_kvcache_, shape_, type_);
    state_.outputs_[output_index_ + i] = presents_[i].get();
  }

  is_first_update_ = false;
}

void CombinedKeyValueCache::RewindTo(size_t index) {
  if (shape_[3] <= static_cast<int>(index)) {
    throw std::runtime_error("Requested length of rewind is greater than the current length.");
  }

  is_first_update_ = true;
  if (index == 0) {
    for (int i = 0; i < layer_count_; i++) {
      pasts_[i] = nullptr;
      state_.inputs_[input_index_ + i] = empty_past_.get();
    }
  } else if (type_ == Ort::TypeToTensorType<float>) {
    RewindPastTensorsTo<float>(index);
  } else {
    RewindPastTensorsTo<Ort::Float16_t>(index);
  }
}

template <typename T>
void CombinedKeyValueCache::RewindPastTensorsTo(size_t index) {
  assert(index > 0 && shape_[3] >= static_cast<int64_t>(index));
  std::array<int64_t, 5> new_shape = shape_;
  new_shape[3] = static_cast<int>(index);
  auto batch_x_num_heads = new_shape[1] * new_shape[2];
  auto new_length_x_head_size = new_shape[3] * new_shape[4];
  auto old_length_x_head_size = shape_[3] * new_shape[4];
  shape_[3] = new_shape[3];

  for (int i = 0; i < layer_count_; i++) {
    OrtValue& present = *presents_[i];
    std::unique_ptr<OrtValue> past = OrtValue::CreateTensor(*model_.allocator_kvcache_, shape_, type_);
    for (int j = 0; j < 2 * batch_x_num_heads; j++) {
      auto present_data = present.GetTensorData<T>() + j * old_length_x_head_size;
      auto past_data = past->GetTensorMutableData<T>() + j * new_length_x_head_size;
#if USE_CUDA
      if (model_.device_type_ == DeviceType::CUDA) {
        cudaMemcpyAsync(past_data, present_data, new_length_x_head_size * sizeof(T), cudaMemcpyDeviceToDevice, model_.cuda_stream_);
      } else
#elif USE_DML
      if (model_.device_type_ == DeviceType::DML) {
        // TODO: Implement DML version
      } else
#endif
      {
        copy(std::span<const T>(present_data, new_length_x_head_size), std::span<T>(past_data, new_length_x_head_size));
      }
    }
    pasts_[i] = std::move(past);
    state_.inputs_[input_index_ + i] = pasts_[i].get();
  }
}

// Copy present state to past state reordered by the beam_indices
template <typename ScoreType>
void CombinedKeyValueCache::PickPastState(DeviceSpan<int32_t> beam_indices_device, int index) {
  std::span<const int32_t> beam_indices = beam_indices_device.CopyDeviceToCpu();
  auto block_size_per_beam = shape_[2] * shape_[3] * shape_[4];
  auto past_key_size = shape_[1] * block_size_per_beam;
  auto element_count = shape_[0] * past_key_size;

  const OrtValue& present = *presents_[index];
  std::unique_ptr<OrtValue> past = OrtValue::CreateTensor<ScoreType>(*model_.allocator_kvcache_, shape_);
  auto past_span = std::span<ScoreType>(past->GetTensorMutableData<ScoreType>(), element_count);
  auto present_span = std::span<const ScoreType>(present.GetTensorData<ScoreType>(), element_count);

#if USE_CUDA
  if (model_.device_type_ == DeviceType::CUDA) {
    for (size_t j = 0; j < beam_indices.size(); j++) {
      int32_t beam_index = beam_indices[j];
      auto present_key = present_span.subspan(beam_index * block_size_per_beam, block_size_per_beam);
      auto present_value = present_span.subspan(past_key_size + beam_index * block_size_per_beam, block_size_per_beam);

      auto past_key = past_span.subspan(j * block_size_per_beam, block_size_per_beam);
      auto past_value = past_span.subspan(past_key_size + j * block_size_per_beam, block_size_per_beam);
      cudaMemcpyAsync(past_key.data(), present_key.data(), present_key.size_bytes(), cudaMemcpyDeviceToDevice, model_.cuda_stream_);
      cudaMemcpyAsync(past_value.data(), present_value.data(), present_value.size_bytes(), cudaMemcpyDeviceToDevice, model_.cuda_stream_);
    }
  } else
#endif
  {
    for (size_t j = 0; j < beam_indices.size(); j++) {
      int32_t const beam_index = beam_indices[j];
      auto present_key = present_span.subspan(beam_index * block_size_per_beam, block_size_per_beam);
      auto present_value = present_span.subspan(past_key_size + beam_index * block_size_per_beam, block_size_per_beam);

      auto past_key = past_span.subspan(j * block_size_per_beam, block_size_per_beam);
      auto past_value = past_span.subspan(past_key_size + j * block_size_per_beam, block_size_per_beam);
      copy(present_key, past_key);
      copy(present_value, past_value);
    }
  }

  pasts_[index] = std::move(past);
}

void CombinedKeyValueCache::PickPastState(DeviceSpan<int32_t> beam_indices, int index) {
  if (type_ == Ort::TypeToTensorType<float>) {
    PickPastState<float>(beam_indices, index);
  } else {
    PickPastState<Ort::Float16_t>(beam_indices, index);
  }
}

DefaultKeyValueCache::DefaultKeyValueCache(State& state)
    : state_{state},
      layer_count_{model_.config_->model.decoder.num_hidden_layers},
      past_present_share_buffer_{state_.params_->search.past_present_share_buffer && (state_.params_->search.num_beams == 1 || model_.config_->model.type == "whisper")},
      shape_{state_.params_->BatchBeamSize(), model_.config_->model.decoder.num_key_value_heads, 0, model_.config_->model.decoder.head_size} {
  if (g_log.enabled && g_log.warning && past_present_share_buffer_ != state_.params_->search.past_present_share_buffer)
    Log("warning", "past_present_share_buffer search option set to true, but has been disabled due to the current configuration. See https://aka.ms/generate_config for details");

  pasts_.resize(layer_count_ * 2);
  presents_.reserve(layer_count_ * 2);

  for (int i = 0; i < layer_count_; ++i) {
    input_name_strings_.emplace_back(ComposeKeyValueName(model_.config_->model.decoder.inputs.past_key_names, i));
    input_name_strings_.emplace_back(ComposeKeyValueName(model_.config_->model.decoder.inputs.past_value_names, i));

    output_name_strings_.emplace_back(ComposeKeyValueName(model_.config_->model.decoder.outputs.present_key_names, i));
    output_name_strings_.emplace_back(ComposeKeyValueName(model_.config_->model.decoder.outputs.present_value_names, i));
  }

  // Derive the KV data type from the KV input 0
  type_ = model_.session_info_->GetInputDataType(input_name_strings_[0]);

  empty_past_ = OrtValue::CreateTensor(*model_.allocator_kvcache_, shape_, type_);

  // Set the size after empty_past_ has been created with 0 for this field
  if (past_present_share_buffer_) {
    shape_[2] = state_.params_->search.max_length;

    if (state_.GetCapturedGraphInfo()) {
      sb_kv_caches_.reserve(layer_count_ * 2);
      for (int i = 0; i < layer_count_ * 2; ++i) {
        sb_kv_caches_.push_back(state_.GetCapturedGraphInfo()->sb_kv_caches_[i].get());
      }
    }
  }

  auto kv_cache_size_bytes = SizeOf(type_) * shape_[0] * shape_[1] * shape_[2] * shape_[3];
  try {
    for (int i = 0; i < layer_count_ * 2; ++i) {
      presents_.push_back(
          sb_kv_caches_.empty() ? OrtValue::CreateTensor(*model_.allocator_kvcache_, shape_, type_)
                                : sb_kv_caches_[i]->CreateTensorOnStaticBuffer(shape_, type_));
#if USE_CUDA
      if (model_.device_type_ == DeviceType::CUDA) {
        cudaMemsetAsync(presents_.back()->GetTensorMutableRawData(), 0, kv_cache_size_bytes, model_.cuda_stream_);
      } else
#endif
      {
        if (model_.device_type_ == DeviceType::CPU) {
          // FIXME: this is a device ternsor and we can only use memset for cpu. Revisit for other EPs.
          memset(presents_.back()->GetTensorMutableRawData(), 0, kv_cache_size_bytes);
        }
      }
    }
  } catch (const Ort::Exception&) {
    std::ostringstream oss;
    oss << "Could not allocate the key-value cache buffer of shape: ["
        << "batch_size (" << shape_[0] << "), num_key_value_heads ("
        << shape_[1] << "), max_length (" << shape_[2] << "), head_size ("
        << shape_[3] << ")] for " << layer_count_ << " layers. "
        << "Try reducing the max_length requested or reducing the batch size.";
    throw std::runtime_error(oss.str());
  }
}

void DefaultKeyValueCache::AddEncoder() {
  // We don't set the input_index_ & output_index_ because the encoder step only runs once, there's no update

  for (int i = 0; i < layer_count_ * 2; ++i) {
    state_.outputs_.push_back(presents_[i].get());
    state_.output_names_.push_back(output_name_strings_[i].c_str());
  }
}

void DefaultKeyValueCache::Add() {
  input_index_ = state_.inputs_.size();
  output_index_ = state_.outputs_.size();

  for (int i = 0; i < layer_count_ * 2; ++i) {
    state_.inputs_.push_back(empty_past_.get());  // Set empty past here, AddEncoder() & Update() take care of the rest
    state_.input_names_.push_back(input_name_strings_[i].c_str());
    state_.outputs_.push_back(presents_[i].get());
    state_.output_names_.push_back(output_name_strings_[i].c_str());
  }

  // For shared_past_present, the past & presents never change, so set the inputs to the present values (outputs are already set above)
  if (past_present_share_buffer_) {
    for (int i = 0; i < layer_count_ * 2; ++i) {
      state_.inputs_[input_index_ + i] = presents_[i].get();
    }
  }
}

void DefaultKeyValueCache::Update(DeviceSpan<int32_t> beam_indices, int total_length) {
  // If we're sharing past & present buffers there is nothing to do here, so early exit
  if (past_present_share_buffer_)
    return;

  if (!is_first_update_) {
    for (int i = 0; i < layer_count_ * 2; i++) {
      if (beam_indices.empty()) {
        pasts_[i] = std::move(presents_[i]);
      } else {
        PickPastState(beam_indices, i);
      }
      state_.inputs_[input_index_ + i] = pasts_[i].get();
    }
  }

  shape_[2] = total_length;
  for (int i = 0; i < layer_count_ * 2; i++) {
    presents_[i] = OrtValue::CreateTensor(*model_.allocator_kvcache_, shape_, type_);
    state_.outputs_[output_index_ + i] = presents_[i].get();
  }

  is_first_update_ = false;
}

void DefaultKeyValueCache::RewindTo(size_t index) {
  if (past_present_share_buffer_) {
    return;
  } else if (shape_[2] <= static_cast<int>(index)) {
    throw std::runtime_error("Requested length of rewind is greater than the current length.");
  }

  is_first_update_ = true;
  if (index == 0) {
    for (int i = 0; i < layer_count_ * 2; i++) {
      pasts_[i] = nullptr;
      state_.inputs_[input_index_ + i] = empty_past_.get();
    }
  } else if (type_ == Ort::TypeToTensorType<float>) {
    RewindPastTensorsTo<float>(index);
  } else {
    RewindPastTensorsTo<Ort::Float16_t>(index);
  }
}

template <typename T>
void DefaultKeyValueCache::RewindPastTensorsTo(size_t index) {
  assert(index > 0 && shape_[2] >= static_cast<int64_t>(index) && !past_present_share_buffer_);
  std::array<int64_t, 4> new_shape = shape_;
  new_shape[2] = static_cast<int>(index);
  auto batch_x_num_heads = new_shape[0] * new_shape[1];
  auto new_length_x_head_size = new_shape[2] * new_shape[3];
  auto old_length_x_head_size = shape_[2] * new_shape[3];
  shape_[2] = new_shape[2];

  for (int i = 0; i < layer_count_ * 2; i++) {
    OrtValue& present = *presents_[i];
    std::unique_ptr<OrtValue> past = OrtValue::CreateTensor(*model_.allocator_kvcache_, shape_, type_);
    for (int j = 0; j < batch_x_num_heads; j++) {
      auto present_data = present.GetTensorData<T>() + j * old_length_x_head_size;
      auto past_data = past->GetTensorMutableData<T>() + j * new_length_x_head_size;
#if USE_CUDA
      if (model_.device_type_ == DeviceType::CUDA) {
        cudaMemcpyAsync(past_data, present_data, new_length_x_head_size * sizeof(T), cudaMemcpyDeviceToDevice, model_.cuda_stream_);
      } else
#elif USE_DML
      if (model_.device_type_ == DeviceType::DML) {
        // TODO: Implement DML copy
      } else
#endif
      {
        copy(std::span<const T>(present_data, new_length_x_head_size), std::span<T>(past_data, new_length_x_head_size));
      }
    }
    pasts_[i] = std::move(past);
    state_.inputs_[input_index_ + i] = pasts_[i].get();
  }
}

// Copy present state to past state reordered by the beam_indices
template <typename ScoreType>
void DefaultKeyValueCache::PickPastState(DeviceSpan<int32_t> beam_indices_device, int index) {
  std::span<int32_t> beam_indices = beam_indices_device.CopyDeviceToCpu();
  auto block_size_per_beam = shape_[1] * shape_[2] * shape_[3];
  auto element_count = shape_[0] * block_size_per_beam;

  const OrtValue& present_value = *presents_[index];
  std::unique_ptr<OrtValue> past_value = OrtValue::CreateTensor<ScoreType>(*model_.allocator_kvcache_, shape_);
  auto past_span = std::span<ScoreType>(past_value->GetTensorMutableData<ScoreType>(), element_count);
  auto present_span = std::span<const ScoreType>(present_value.GetTensorData<ScoreType>(), element_count);

#if USE_CUDA
  if (model_.device_type_ == DeviceType::CUDA) {
    for (size_t j = 0; j < beam_indices.size(); j++) {
      int32_t beam_index = beam_indices[j];
      auto present = present_span.subspan(beam_index * block_size_per_beam, block_size_per_beam);
      auto past = past_span.subspan(j * block_size_per_beam, block_size_per_beam);
      cudaMemcpyAsync(past.data(), present.data(), present.size_bytes(), cudaMemcpyDeviceToDevice, model_.cuda_stream_);
    }
  } else
#endif
  {
    for (size_t j = 0; j < beam_indices.size(); j++) {
      int32_t const beam_index = beam_indices[j];
      auto present = present_span.subspan(beam_index * block_size_per_beam, block_size_per_beam);
      auto past = past_span.subspan(j * block_size_per_beam, block_size_per_beam);
      copy(present, past);
    }
  }

  pasts_[index] = std::move(past_value);
}

void DefaultKeyValueCache::PickPastState(DeviceSpan<int32_t> beam_indices, int index) {
  if (type_ == Ort::TypeToTensorType<float>) {
    PickPastState<float>(beam_indices, index);
  } else {
    PickPastState<Ort::Float16_t>(beam_indices, index);
  }
}

CrossCache::CrossCache(State& state)
    : state_{state},
      layer_count_{model_.config_->model.decoder.num_hidden_layers},
      shape_{state_.params_->BatchBeamSize(), model_.config_->model.decoder.num_key_value_heads, 1500, model_.config_->model.decoder.head_size} {
  values_.reserve(layer_count_ * 2);

  for (int i = 0; i < layer_count_; ++i) {
    input_name_strings_.emplace_back(ComposeKeyValueName(model_.config_->model.decoder.inputs.cross_past_key_names, i));
    input_name_strings_.emplace_back(ComposeKeyValueName(model_.config_->model.decoder.inputs.cross_past_value_names, i));

    output_name_strings_.emplace_back(ComposeKeyValueName(model_.config_->model.decoder.outputs.cross_present_key_names, i));
    output_name_strings_.emplace_back(ComposeKeyValueName(model_.config_->model.decoder.outputs.cross_present_value_names, i));
  }

  // Derive the KV data type from the KV input 0
  type_ = model_.session_info_->GetInputDataType(input_name_strings_[0]);

  for (int i = 0; i < layer_count_; ++i) {
    values_.push_back(OrtValue::CreateTensor(*model_.allocator_kvcache_, shape_, type_));
    values_.push_back(OrtValue::CreateTensor(*model_.allocator_kvcache_, shape_, type_));
  }
}

void CrossCache::AddOutputs() {
  for (int i = 0; i < layer_count_ * 2; ++i) {
    state_.outputs_.push_back(values_[i].get());
    state_.output_names_.push_back(output_name_strings_[i].c_str());
  }
}

void CrossCache::AddInputs() {
  for (int i = 0; i < layer_count_ * 2; ++i) {
    state_.inputs_.push_back(values_[i].get());
    state_.input_names_.push_back(input_name_strings_[i].c_str());
  }
}

WindowedKeyValueCache::WindowedKeyValueCache(State& state)
    : state_{state},
      layer_count_{model_.config_->model.decoder.num_hidden_layers},
      window_size_{model_.config_->model.decoder.sliding_window->window_size},
      key_cache_shape_in_{model_.config_->model.decoder.num_key_value_heads, 1,
                          model_.config_->model.decoder.head_size, model_.config_->model.context_length - window_size_},
      key_cache_shape_out_{model_.config_->model.decoder.num_key_value_heads, 1,
                           model_.config_->model.decoder.head_size, window_size_},
      value_cache_shape_in_{model_.config_->model.decoder.num_key_value_heads, 1,
                            model_.config_->model.context_length - window_size_, model_.config_->model.decoder.head_size},
      value_cache_shape_out_{model_.config_->model.decoder.num_key_value_heads, 1,
                             window_size_, model_.config_->model.decoder.head_size} {
  if (layer_count_ == 0) {
    throw std::runtime_error("Expected there to be at least 1 layer in the model. Actual: " +
                             std::to_string(layer_count_) + ". Please check the num_hidden_layers attribute in the model configuration.");
  }
  for (int i = 0; i < layer_count_; ++i) {
    input_name_strings_.emplace_back(ComposeKeyValueName(model_.config_->model.decoder.inputs.past_key_names, i));
    input_name_strings_.emplace_back(ComposeKeyValueName(model_.config_->model.decoder.inputs.past_value_names, i));

    output_name_strings_.emplace_back(ComposeKeyValueName(model_.config_->model.decoder.outputs.present_key_names, i));
    output_name_strings_.emplace_back(ComposeKeyValueName(model_.config_->model.decoder.outputs.present_value_names, i));
  }

  type_ = model_.session_info_->GetInputDataType(input_name_strings_[0]);
  if (type_ != Ort::TypeToTensorType<uint8_t>) {
    throw std::runtime_error("Expected input data type to be uint8_t for WindowedKeyValueCache. Actual: " +
                             std::to_string(type_));
  }

  for (int i = 0; i < layer_count_; ++i) {
    key_caches_in_.push_back(
        OrtValue::CreateTensor(*model_.allocator_device_, key_cache_shape_in_, type_));
    std::fill_n(key_caches_in_[i]->GetTensorMutableData<uint8_t>(),
                ElementCountFromShape(key_cache_shape_in_),
                static_cast<uint8_t>(model_.config_->model.decoder.sliding_window->pad_value));

    value_caches_in_.push_back(
        OrtValue::CreateTensor(*model_.allocator_device_, value_cache_shape_in_, type_));
    std::fill_n(value_caches_in_[i]->GetTensorMutableData<uint8_t>(),
                ElementCountFromShape(value_cache_shape_in_),
                static_cast<uint8_t>(model_.config_->model.decoder.sliding_window->pad_value));

    key_caches_out_.push_back(
        OrtValue::CreateTensor(*model_.allocator_device_, key_cache_shape_out_, type_));
    value_caches_out_.push_back(
        OrtValue::CreateTensor(*model_.allocator_device_, value_cache_shape_out_, type_));
  }
}

void WindowedKeyValueCache::Add() {
  input_index_ = state_.inputs_.size();
  output_index_ = state_.outputs_.size();

  for (size_t layer_idx = 0; layer_idx < layer_count_; ++layer_idx) {
    state_.inputs_.push_back(key_caches_in_[layer_idx].get());
    state_.input_names_.push_back(input_name_strings_[2 * layer_idx].c_str());

    state_.inputs_.push_back(value_caches_in_[layer_idx].get());
    state_.input_names_.push_back(input_name_strings_[2 * layer_idx + 1].c_str());

    state_.outputs_.push_back(key_caches_out_[layer_idx].get());
    state_.output_names_.push_back(output_name_strings_[2 * layer_idx].c_str());

    state_.outputs_.push_back(value_caches_out_[layer_idx].get());
    state_.output_names_.push_back(output_name_strings_[2 * layer_idx + 1].c_str());
  }
}

void WindowedKeyValueCache::Slide() {
  ThreadPool thread_pool{static_cast<size_t>(layer_count_)};
  thread_pool.Compute([&](size_t layer_idx) {
    uint8_t* key_cache_in_data = key_caches_in_[layer_idx]->GetTensorMutableData<uint8_t>();
    uint8_t* key_cache_out_data = key_caches_out_[layer_idx]->GetTensorMutableData<uint8_t>();

    int64_t num_key_cache_chunks = key_cache_shape_in_[0] * key_cache_shape_in_[2];
    for (int64_t j = 0; j < num_key_cache_chunks; ++j) {
      {
        cpu_span<uint8_t> key_cache_dst(key_cache_in_data + j * key_cache_shape_in_[3],
                                        key_cache_shape_in_[3] - window_size_);
        cpu_span<uint8_t> key_cache_src(key_cache_in_data + j * key_cache_shape_in_[3] + window_size_,
                                        key_cache_shape_in_[3] - window_size_);
        std::copy(key_cache_src.begin(), key_cache_src.end(), key_cache_dst.begin());
      }
      {
        cpu_span<uint8_t> key_cache_dst(key_cache_in_data + j * key_cache_shape_in_[3] + key_cache_shape_in_[3] - window_size_,
                                        window_size_);
        cpu_span<uint8_t> key_cache_src(key_cache_out_data + j * key_cache_shape_out_[3],
                                        window_size_);
        std::copy(key_cache_src.begin(), key_cache_src.end(), key_cache_dst.begin());
      }
    }

    uint8_t* value_cache_in_data = value_caches_in_[layer_idx]->GetTensorMutableData<uint8_t>();
    uint8_t* value_cache_out_data = value_caches_out_[layer_idx]->GetTensorMutableData<uint8_t>();

    for (int64_t j = 0; j < value_cache_shape_in_[0]; ++j) {
      {
        cpu_span<uint8_t> value_cache_dst(value_cache_in_data + (j * value_cache_shape_in_[2] * value_cache_shape_in_[3]),
                                          (value_cache_shape_in_[2] - window_size_) * value_cache_shape_in_[3]);
        cpu_span<uint8_t> value_cache_src(value_cache_in_data + (j * value_cache_shape_in_[2] * value_cache_shape_in_[3]) +
                                              (window_size_ * value_cache_shape_in_[3]),
                                          (value_cache_shape_in_[2] - window_size_) * value_cache_shape_in_[3]);
        std::copy(value_cache_src.begin(), value_cache_src.end(), value_cache_dst.begin());
      }
      {
        cpu_span<uint8_t> value_cache_dst(value_cache_in_data + (j * value_cache_shape_in_[2] * value_cache_shape_in_[3]) +
                                              ((value_cache_shape_in_[2] - window_size_) * value_cache_shape_in_[3]),
                                          window_size_ * value_cache_shape_in_[3]);
        cpu_span<uint8_t> value_cache_src(value_cache_out_data + (j * value_cache_shape_out_[2] * value_cache_shape_out_[3]),
                                          window_size_ * value_cache_shape_out_[3]);
        std::copy(value_cache_src.begin(), value_cache_src.end(), value_cache_dst.begin());
      }
    }
  });
}

void WindowedKeyValueCache::Update(DeviceSpan<int32_t> beam_indices, int current_length) {
  if (is_first_update_) {
    num_windows_ = (current_length + window_size_ - 1) / window_size_;
    is_first_update_ = false;
    window_index_++;
    return;
  } else if (window_size_ == 1 || window_index_ < num_windows_) {
    Slide();
    window_index_++;
    return;
  }

  // Transition from prompt processing to token generation.
  // Concatenate the last window_size_ elements to the end of the cache

  // key_caches_in_ = Concat(key_caches_in_[:, :, :, 1:], key_caches_out_)
  // [num_key_value_heads, 1, head_size, context_length-1] = [num_key_value_heads, 1, head_size, context_length - window_size_ - 1] +
  //                                                         [num_key_value_heads, 1, head_size, window_size_]
  // value_cache = Concat(value_caches_in_[:, :, 1:, :], value_caches_out_)
  // [num_key_value_heads, 1, context_length - 1, head_size] = [num_key_value_heads, 1, context_length - window_size_ - 1, head_size] +
  //                                                           [num_key_value_heads, 1, window_size_, head_size]

  int updated_window_size = 1;
  auto updated_key_cache_shape_in = std::array<int64_t, 4>{model_.config_->model.decoder.num_key_value_heads, 1,
                                                           model_.config_->model.decoder.head_size,
                                                           model_.config_->model.context_length - updated_window_size};

  auto updated_value_cache_shape_in = std::array<int64_t, 4>{model_.config_->model.decoder.num_key_value_heads, 1,
                                                             model_.config_->model.context_length - updated_window_size,
                                                             model_.config_->model.decoder.head_size};

  auto updated_key_cache_shape_out = std::array<int64_t, 4>{model_.config_->model.decoder.num_key_value_heads, 1,
                                                            model_.config_->model.decoder.head_size,
                                                            updated_window_size};

  auto updated_value_cache_shape_out = std::array<int64_t, 4>{model_.config_->model.decoder.num_key_value_heads, 1,
                                                              updated_window_size,
                                                              model_.config_->model.decoder.head_size};

  ThreadPool thread_pool{static_cast<size_t>(layer_count_)};
  thread_pool.Compute([&](size_t layer_idx) {
    std::unique_ptr<OrtValue> key_cache = OrtValue::CreateTensor(*model_.allocator_device_, updated_key_cache_shape_in, type_);

    uint8_t* key_cache_data = key_cache->GetTensorMutableData<uint8_t>();
    uint8_t* key_cache_in_data = key_caches_in_[layer_idx]->GetTensorMutableData<uint8_t>();
    uint8_t* key_cache_out_data = key_caches_out_[layer_idx]->GetTensorMutableData<uint8_t>();

    int64_t num_key_cache_chunks = updated_key_cache_shape_in[0] * updated_key_cache_shape_in[2];
    for (int64_t j = 0; j < num_key_cache_chunks; ++j) {
      {
        cpu_span<uint8_t> key_cache_dst(key_cache_data + j * updated_key_cache_shape_in[3],
                                        updated_key_cache_shape_in[3] - updated_window_size);
        cpu_span<uint8_t> key_cache_src(key_cache_in_data + j * key_cache_shape_in_[3] + updated_window_size,
                                        key_cache_shape_in_[3] - updated_window_size);
        std::copy(key_cache_src.begin(), key_cache_src.end(), key_cache_dst.begin());
      }
      {
        cpu_span<uint8_t> key_cache_dst(key_cache_data + j * updated_key_cache_shape_in[3] +
                                            key_cache_shape_in_[3] - updated_window_size,
                                        window_size_);
        cpu_span<uint8_t> key_cache_src(key_cache_out_data + j * key_cache_shape_out_[3],
                                        window_size_);
        std::copy(key_cache_src.begin(), key_cache_src.end(), key_cache_dst.begin());
      }
    }

    key_caches_in_[layer_idx] = std::move(key_cache);
    key_caches_out_[layer_idx] = OrtValue::CreateTensor(*model_.allocator_device_, updated_key_cache_shape_out, type_);

    std::unique_ptr<OrtValue> value_cache = OrtValue::CreateTensor(*model_.allocator_device_, updated_value_cache_shape_in, type_);

    uint8_t* value_cache_data = value_cache->GetTensorMutableData<uint8_t>();
    uint8_t* value_cache_in_data = value_caches_in_[layer_idx]->GetTensorMutableData<uint8_t>();
    uint8_t* value_cache_out_data = value_caches_out_[layer_idx]->GetTensorMutableData<uint8_t>();

    for (int64_t j = 0; j < updated_value_cache_shape_in[0]; ++j) {
      {
        cpu_span<uint8_t> value_cache_dst(value_cache_data + (j * updated_value_cache_shape_in[2] * updated_value_cache_shape_in[3]),
                                          (value_cache_shape_in_[2] - updated_window_size) * updated_value_cache_shape_in[3]);
        cpu_span<uint8_t> value_cache_src(value_cache_in_data + (j * value_cache_shape_in_[2] * value_cache_shape_in_[3]) +
                                              (updated_window_size * value_cache_shape_in_[3]),
                                          (value_cache_shape_in_[2] - updated_window_size) * value_cache_shape_in_[3]);
        std::copy(value_cache_src.begin(), value_cache_src.end(), value_cache_dst.begin());
      }
      {
        cpu_span<uint8_t> value_cache_dst(value_cache_data + (j * updated_value_cache_shape_in[2] * updated_value_cache_shape_in[3]) +
                                              ((value_cache_shape_in_[2] - updated_window_size) * updated_value_cache_shape_in[3]),
                                          value_cache_shape_out_[2] * value_cache_shape_out_[3]);
        cpu_span<uint8_t> value_cache_src(value_cache_out_data + (j * value_cache_shape_out_[2] * value_cache_shape_out_[3]),
                                          value_cache_shape_out_[2] * value_cache_shape_out_[3]);
        std::copy(value_cache_src.begin(), value_cache_src.end(), value_cache_dst.begin());
      }
    }

    value_caches_in_[layer_idx] = std::move(value_cache);
    value_caches_out_[layer_idx] = OrtValue::CreateTensor(*model_.allocator_device_, updated_value_cache_shape_out, type_);
  });

  window_size_ = 1;
  key_cache_shape_in_ = updated_key_cache_shape_in;
  value_cache_shape_in_ = updated_value_cache_shape_in;
  key_cache_shape_out_ = updated_key_cache_shape_out;
  value_cache_shape_out_ = updated_value_cache_shape_out;

  for (size_t layer_idx = 0; layer_idx < layer_count_; ++layer_idx) {
    state_.inputs_[input_index_ + 2 * layer_idx] = key_caches_in_[layer_idx].get();
    state_.inputs_[input_index_ + 2 * layer_idx + 1] = value_caches_in_[layer_idx].get();
    state_.outputs_[output_index_ + 2 * layer_idx] = key_caches_out_[layer_idx].get();
    state_.outputs_[output_index_ + 2 * layer_idx + 1] = value_caches_out_[layer_idx].get();
  }
}

std::unique_ptr<KeyValueCache> CreateKeyValueCache(State& state) {
  if (!IsCacheNeeded(state.model_)) {
    return nullptr;
  }

  if (state.model_.config_->model.decoder.sliding_window) {
    return std::make_unique<WindowedKeyValueCache>(state);
  } else {
    return std::make_unique<DefaultKeyValueCache>(state);
  }
}

}  // namespace Generators
