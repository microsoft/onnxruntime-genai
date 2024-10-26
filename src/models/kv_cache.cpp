#include "../generators.h"
#include "model.h"
#include "kv_cache.h"

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

}  // namespace

KV_Cache_Combined::KV_Cache_Combined(State& state)
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
  shape_[3] = state_.params_->sequence_length;

  for (int i = 0; i < layer_count_; ++i) {
    presents_.push_back(OrtValue::CreateTensor(*model_.allocator_kvcache_, shape_, type_));
  }
}

void KV_Cache_Combined::Add() {
  input_index_ = state_.inputs_.size();
  output_index_ = state_.outputs_.size();

  for (int i = 0; i < layer_count_; i++) {
    state_.inputs_.push_back(empty_past_.get());
    state_.input_names_.push_back(input_name_strings_[i].c_str());
    state_.outputs_.push_back(presents_[i].get());
    state_.output_names_.push_back(output_name_strings_[i].c_str());
  }
}

void KV_Cache_Combined::Update(DeviceSpan<int32_t> beam_indices, int current_length) {
  assert(state_.params_->search.num_beams == 1 || !beam_indices.empty());  // We require beam_indices if we're a beam search

  for (int i = 0; i < layer_count_; i++) {
    if (beam_indices.empty()) {
      pasts_[i] = std::move(presents_[i]);
    } else {
      PickPastState(beam_indices, i);
    }
  }

  shape_[3] = current_length;
  for (int i = 0; i < layer_count_; i++) {
    presents_[i] = OrtValue::CreateTensor(*model_.allocator_kvcache_, shape_, type_);
    state_.inputs_[input_index_ + i] = pasts_[i].get();
    state_.outputs_[output_index_ + i] = presents_[i].get();
  }
}

// Copy present state to past state reordered by the beam_indices
template <typename ScoreType>
void KV_Cache_Combined::PickPastState(DeviceSpan<int32_t> beam_indices_device, int index) {
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

void KV_Cache_Combined::PickPastState(DeviceSpan<int32_t> beam_indices, int index) {
  if (type_ == Ort::TypeToTensorType<float>) {
    PickPastState<float>(beam_indices, index);
  } else {
    PickPastState<Ort::Float16_t>(beam_indices, index);
  }
}

bool KV_Cache::IsCacheNeeded(const Model& model) {
  return model.session_info_->HasInput(ComposeKeyValueName(model.config_->model.decoder.inputs.past_key_names, 0));
}

KV_Cache::KV_Cache(State& state)
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
  if (past_present_share_buffer_)
    shape_[2] = state_.params_->search.max_length;
  else
    shape_[2] = state_.params_->sequence_length;

  if (state_.GetCapturedGraphInfo()) {
    assert(past_present_share_buffer_);
    sb_kv_caches_.reserve(layer_count_ * 2);
    for (int i = 0; i < layer_count_ * 2; ++i) {
      sb_kv_caches_.push_back(state_.GetCapturedGraphInfo()->sb_kv_caches_[i].get());
    }
  }

  for (int i = 0; i < layer_count_ * 2; ++i) {
    presents_.push_back(
        sb_kv_caches_.empty() ? OrtValue::CreateTensor(*model_.allocator_kvcache_, shape_, type_)
                              : sb_kv_caches_[i]->CreateTensorOnStaticBuffer(shape_, type_));
  }
}

void KV_Cache::AddEncoder() {
  // We don't set the input_index_ & output_index_ because the encoder step only runs once, there's no update

  for (int i = 0; i < layer_count_ * 2; ++i) {
    state_.outputs_.push_back(presents_[i].get());
    state_.output_names_.push_back(output_name_strings_[i].c_str());
  }
}

void KV_Cache::Add() {
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

void KV_Cache::Update(DeviceSpan<int32_t> beam_indices, int current_length) {
  // If we're sharing past & present buffers there is nothing to do here, so early exit
  if (past_present_share_buffer_)
    return;

  for (int i = 0; i < layer_count_ * 2; i++) {
    if (beam_indices.empty()) {
      pasts_[i] = std::move(presents_[i]);
    } else {
      PickPastState(beam_indices, i);
    }
    state_.inputs_[input_index_ + i] = pasts_[i].get();
  }

  shape_[2] = current_length;
  for (int i = 0; i < layer_count_ * 2; i++) {
    presents_[i] = OrtValue::CreateTensor(*model_.allocator_kvcache_, shape_, type_);
    state_.outputs_[output_index_ + i] = presents_[i].get();
  }
}

// Copy present state to past state reordered by the beam_indices
template <typename ScoreType>
void KV_Cache::PickPastState(DeviceSpan<int32_t> beam_indices_device, int index) {
  std::span<int32_t> beam_indices = beam_indices_device.Span();
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

void KV_Cache::PickPastState(DeviceSpan<int32_t> beam_indices, int index) {
  if (type_ == Ort::TypeToTensorType<float>) {
    PickPastState<float>(beam_indices, index);
  } else {
    PickPastState<Ort::Float16_t>(beam_indices, index);
  }
}

Cross_Cache::Cross_Cache(State& state)
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

void Cross_Cache::AddOutputs() {
  for (int i = 0; i < layer_count_ * 2; ++i) {
    state_.outputs_.push_back(values_[i].get());
    state_.output_names_.push_back(output_name_strings_[i].c_str());
  }
}

void Cross_Cache::AddInputs() {
  for (int i = 0; i < layer_count_ * 2; ++i) {
    state_.inputs_.push_back(values_[i].get());
    state_.input_names_.push_back(input_name_strings_[i].c_str());
  }
}

}  // namespace Generators
