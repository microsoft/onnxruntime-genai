#include "../generators.h"
#include "model.h"
#include "kv_cache.h"

namespace Generators {

KV_Cache_Combined::KV_Cache_Combined(const Model& model, State& state)
    : model_{model},
      state_{state},
      layer_count_{model.config_->model.num_hidden_layers},
      shape_{2, state_.search_params_.batch_size * state_.search_params_.num_beams, model.config_->model.num_attention_heads, 0, model.config_->model.hidden_size},
      empty_past_{OrtValue::CreateTensor(*model_.allocator_device_, shape_, model_.config_->model.kv_type)} {
  pasts_.resize(layer_count_);
  presents_.reserve(layer_count_);

  shape_[3] = state_.search_params_.sequence_length;
  for (int i = 0; i < layer_count_; ++i) {
    presents_.push_back(OrtValue::CreateTensor(*model.allocator_device_, shape_, model_.config_->model.kv_type));

    char string[64];
    snprintf(string, std::size(string), model.config_->model.past_names.c_str(), i);
    input_name_strings_.emplace_back(string);

    snprintf(string, std::size(string), model.config_->model.present_names.c_str(), i);
    output_name_strings_.emplace_back(string);
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

void KV_Cache_Combined::Update(std::span<const int32_t> beam_indices, int current_length) {
  assert(state_.search_params_.num_beams == 1 || !beam_indices.empty());  // We require beam_indices if we're a beam search

  for (int i = 0; i < layer_count_; i++) {
    if (beam_indices.empty()) {
      pasts_[i] = std::move(presents_[i]);
    } else {
      PickPastState(beam_indices, i);
    }
  }

  shape_[3] = current_length;
  for (int i = 0; i < layer_count_; i++) {
    presents_[i] = OrtValue::CreateTensor(*model_.allocator_device_, shape_, model_.config_->model.kv_type);
    state_.inputs_[input_index_ + i] = pasts_[i].get();
    state_.outputs_[output_index_ + i] = presents_[i].get();
  }
}

// Copy present state to past state reordered by the beam_indices
template <typename ScoreType>
void KV_Cache_Combined::PickPastState(std::span<const int32_t> beam_indices, int index) {
  auto block_size_per_beam = shape_[2] * shape_[3] * shape_[4];
  auto past_key_size = shape_[1] * block_size_per_beam;
  auto element_count = shape_[0] * past_key_size;

  const OrtValue& present = *presents_[index];
  std::unique_ptr<OrtValue> past = OrtValue::CreateTensor<ScoreType>(*model_.allocator_device_, shape_);
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

void KV_Cache_Combined::PickPastState(std::span<const int32_t> beam_indices, int index) {
  if (model_.config_->model.kv_type == Ort::TypeToTensorType<float>::type) {
    PickPastState<float>(beam_indices, index);
  } else {
    PickPastState<Ort::Float16_t>(beam_indices, index);
  }
}

KV_Cache::KV_Cache(const Model& model, State& state)
    : model_{model},
      state_{state},
      layer_count_{model_.config_->model.num_hidden_layers},
      shape_{state_.search_params_.batch_size * state_.search_params_.num_beams, model.config_->model.num_attention_heads, 0, model.config_->model.hidden_size},
      empty_past_{OrtValue::CreateTensor(*model_.allocator_device_, shape_, model_.config_->model.kv_type)} {
  pasts_.resize(layer_count_ * 2);
  presents_.reserve(layer_count_ * 2);

  shape_[2] = state_.search_params_.sequence_length;  // Set this after empty_past_ has been created with 0 for this field

  for (int i = 0; i < layer_count_; ++i) {
    presents_.push_back(OrtValue::CreateTensor(*model_.allocator_device_, shape_, model_.config_->model.kv_type));
    presents_.push_back(OrtValue::CreateTensor(*model_.allocator_device_, shape_, model_.config_->model.kv_type));

    char string[64];
    snprintf(string, std::size(string), model.config_->model.past_names_key.c_str(), i);
    input_name_strings_.emplace_back(string);
    snprintf(string, std::size(string), model.config_->model.past_names_value.c_str(), i);
    input_name_strings_.emplace_back(string);

    snprintf(string, std::size(string), model.config_->model.present_names_key.c_str(), i);
    output_name_strings_.emplace_back(string);
    snprintf(string, std::size(string), model.config_->model.present_names_value.c_str(), i);
    output_name_strings_.emplace_back(string);
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
}

void KV_Cache::Update(std::span<const int32_t> beam_indices, int current_length) {
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
    presents_[i] = OrtValue::CreateTensor(*model_.allocator_device_, shape_, model_.config_->model.kv_type);
    state_.outputs_[output_index_ + i] = presents_[i].get();
  }
}

// Copy present state to past state reordered by the beam_indices
template <typename ScoreType>
void KV_Cache::PickPastState(std::span<const int32_t> beam_indices, int index) {
  auto block_size_per_beam = shape_[1] * shape_[2] * shape_[3];
  auto element_count = shape_[0] * block_size_per_beam;

  const OrtValue& present_value = *presents_[index];
  std::unique_ptr<OrtValue> past_value = OrtValue::CreateTensor<ScoreType>(*model_.allocator_device_, shape_);
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

void KV_Cache::PickPastState(std::span<const int32_t> beam_indices, int index) {
  if (model_.config_->model.kv_type == Ort::TypeToTensorType<float>::type) {
    PickPastState<float>(beam_indices, index);
  } else {
    PickPastState<Ort::Float16_t>(beam_indices, index);
  }
}

Cross_Cache::Cross_Cache(const Model& model, State& state)
    : model_{model},
      state_{state},
      layer_count_{model_.config_->model.num_hidden_layers},
      shape_{state_.search_params_.batch_size * state_.search_params_.num_beams, model.config_->model.num_attention_heads, 1500, model.config_->model.hidden_size} {
  values_.reserve(layer_count_ * 2);

  for (int i = 0; i < layer_count_; ++i) {
    values_.push_back(OrtValue::CreateTensor(*model_.allocator_device_, shape_, model_.config_->model.kv_type));
    values_.push_back(OrtValue::CreateTensor(*model_.allocator_device_, shape_, model_.config_->model.kv_type));

    char string[64];
    snprintf(string, std::size(string), model.config_->model.cross_past_names_key.c_str(), i);
    input_name_strings_.emplace_back(string);
    snprintf(string, std::size(string), model.config_->model.cross_past_names_value.c_str(), i);
    input_name_strings_.emplace_back(string);

    snprintf(string, std::size(string), model.config_->model.cross_present_names_key.c_str(), i);
    output_name_strings_.emplace_back(string);
    snprintf(string, std::size(string), model.config_->model.cross_present_names_value.c_str(), i);
    output_name_strings_.emplace_back(string);
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
