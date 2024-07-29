#include "../generators.h"
#include "model.h"
#include "kv_cache.h"

namespace Generators {

namespace {

constexpr int32_t PagedCacheBlockTablePadValue = -1;
constexpr int32_t PagedCacheSlotMappingPadValue = -1;
constexpr char PagedKeyCacheNamePrefix[] = "key_cache.";
constexpr char PagedValueCacheNamePrefix[] = "value_cache.";
constexpr char PagedCacheBlockTablesName[] = "block_tables";
constexpr char PagedCacheSlotMappingName[] = "slot_mapping";
constexpr int32_t DefaultBlockSize = 16;
constexpr float DefaultCacheGPUUtilizationFactor = 0.3f;

CacheOptions MakeCacheOptions(const Model& model, const State& state) {
  return CacheOptions(model.config_->model.decoder.num_hidden_layers,
                      model.config_->model.kv_cache.block_size,
                      model.config_->model.decoder.num_key_value_heads,
                      model.config_->model.decoder.head_size,
                      ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16,
                      model.config_->model.kv_cache.num_blocks,
                      model.config_->model.kv_cache.gpu_utilization_factor);
}

}  // namespace

KV_Cache_Combined::KV_Cache_Combined(const Model& model, State& state)
    : model_{model},
      state_{state},
      layer_count_{model.config_->model.decoder.num_hidden_layers},
      shape_{2, state_.params_->BatchBeamSize(), model.config_->model.decoder.num_key_value_heads, 0, model.config_->model.decoder.head_size} {
  pasts_.resize(layer_count_);
  presents_.reserve(layer_count_);

  for (int i = 0; i < layer_count_; ++i) {
    char string[64];
    snprintf(string, std::size(string), model.config_->model.decoder.inputs.past_names.c_str(), i);
    input_name_strings_.emplace_back(string);

    snprintf(string, std::size(string), model.config_->model.decoder.outputs.present_names.c_str(), i);
    output_name_strings_.emplace_back(string);
  }

  // Derive the KV data type from the KV input 0
  type_ = model_.session_info_->GetInputDataType(input_name_strings_[0]);

  empty_past_ = OrtValue::CreateTensor(*model_.allocator_device_, shape_, type_);
  shape_[3] = state_.params_->sequence_length;

  for (int i = 0; i < layer_count_; ++i) {
    presents_.push_back(OrtValue::CreateTensor(*model.allocator_device_, shape_, type_));
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
    presents_[i] = OrtValue::CreateTensor(*model_.allocator_device_, shape_, type_);
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
  if (type_ == Ort::TypeToTensorType<float>::type) {
    PickPastState<float>(beam_indices, index);
  } else {
    PickPastState<Ort::Float16_t>(beam_indices, index);
  }
}

KV_Cache::KV_Cache(const Model& model, State& state)
    : model_{model},
      state_{state},
      layer_count_{model_.config_->model.decoder.num_hidden_layers},
      past_present_share_buffer_{state_.params_->search.past_present_share_buffer && state_.params_->search.num_beams == 1},
      shape_{state_.params_->BatchBeamSize(), model.config_->model.decoder.num_key_value_heads, 0, model.config_->model.decoder.head_size} {
  if (g_log.enabled && g_log.warning && past_present_share_buffer_ != state_.params_->search.past_present_share_buffer)
    Log("warning", "past_present_share_buffer search option set to true, but has been disabled due to the current configuration. See https://aka.ms/generate_config for details");

  pasts_.resize(layer_count_ * 2);
  presents_.reserve(layer_count_ * 2);

  for (int i = 0; i < layer_count_; ++i) {
    char string[64];
    snprintf(string, std::size(string), model.config_->model.decoder.inputs.past_key_names.c_str(), i);
    input_name_strings_.emplace_back(string);
    snprintf(string, std::size(string), model.config_->model.decoder.inputs.past_value_names.c_str(), i);
    input_name_strings_.emplace_back(string);

    snprintf(string, std::size(string), model.config_->model.decoder.outputs.present_key_names.c_str(), i);
    output_name_strings_.emplace_back(string);
    snprintf(string, std::size(string), model.config_->model.decoder.outputs.present_value_names.c_str(), i);
    output_name_strings_.emplace_back(string);
  }

  // Derive the KV data type from the KV input 0
  type_ = model_.session_info_->GetInputDataType(input_name_strings_[0]);

  empty_past_ = OrtValue::CreateTensor(*model_.allocator_device_, shape_, type_);

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
        sb_kv_caches_.empty() ? OrtValue::CreateTensor(*model_.allocator_device_, shape_, type_)
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

void KV_Cache::Update(std::span<const int32_t> beam_indices, int current_length) {
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
    presents_[i] = OrtValue::CreateTensor(*model_.allocator_device_, shape_, type_);
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
  if (type_ == Ort::TypeToTensorType<float>::type) {
    PickPastState<float>(beam_indices, index);
  } else {
    PickPastState<Ort::Float16_t>(beam_indices, index);
  }
}

Cross_Cache::Cross_Cache(const Model& model, State& state)
    : model_{model},
      state_{state},
      layer_count_{model_.config_->model.decoder.num_hidden_layers},
      shape_{state_.params_->BatchBeamSize(), model.config_->model.decoder.num_key_value_heads, 1500, model.config_->model.decoder.head_size} {
  values_.reserve(layer_count_ * 2);

  for (int i = 0; i < layer_count_; ++i) {
    char string[64];
    snprintf(string, std::size(string), model.config_->model.decoder.inputs.cross_past_key_names.c_str(), i);
    input_name_strings_.emplace_back(string);
    snprintf(string, std::size(string), model.config_->model.decoder.inputs.cross_past_value_names.c_str(), i);
    input_name_strings_.emplace_back(string);

    snprintf(string, std::size(string), model.config_->model.decoder.outputs.cross_present_key_names.c_str(), i);
    output_name_strings_.emplace_back(string);
    snprintf(string, std::size(string), model.config_->model.decoder.outputs.cross_present_value_names.c_str(), i);
    output_name_strings_.emplace_back(string);
  }

  // Derive the KV data type from the KV input 0
  type_ = model_.session_info_->GetInputDataType(input_name_strings_[0]);

  for (int i = 0; i < layer_count_; ++i) {
    values_.push_back(OrtValue::CreateTensor(*model_.allocator_device_, shape_, type_));
    values_.push_back(OrtValue::CreateTensor(*model_.allocator_device_, shape_, type_));
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

// PagedCacheOrchestrator::PagedCacheOrchestrator(const Model& model, State& state)
//     : model_{model},
//       state_{state},
//       paged_cache_(std::make_unique<PagedCacheManager>(MakeCacheOptions(model, state),
//                                                        &model.allocator_cpu_, model.allocator_device_)) {
// }

// void PagedCacheOrchestrator::Add() {
//   input_offset_ = state_.inputs_.size();

//   auto num_sequences = state_.params_->BatchBeamSize();
//   std::vector<size_t> sequence_ids(num_sequences);
//   std::iota(sequence_ids.begin(), sequence_ids.end(), 0);
//   for (int i = 0; i < num_sequences; ++i) {
//     paged_cache_->Add(i, state_.params_->sequence_length);
//   }

//   for (int i = 0; i < layer_count_; ++i) {
//     auto [key_cache, value_cache] = paged_cache_->Cache(i);
//     state_.input_names_.push_back((PagedKeyCacheNamePrefix + std::to_string(i)).c_str());
//     state_.inputs_.push_back(key_cache);
//     state_.input_names_.push_back((PagedValueCacheNamePrefix + std::to_string(i)).c_str());
//     state_.inputs_.push_back(value_cache);
//   }

//   state_.input_names_.push_back(PagedCacheBlockTablesName);
//   block_tables_ = paged_cache_->BlockTables(sequence_ids);
//   state_.inputs_.push_back(block_tables_.get());
//   state_.input_names_.push_back(PagedCacheSlotMappingName);
//   slot_mapping_ = paged_cache_->SlotMapping(sequence_ids);
//   state_.inputs_.push_back(slot_mapping_.get());
// }

// void PagedCacheOrchestrator::Update([[maybe_unused]] std::span<const int32_t> beam_indices,
//                                     [[maybe_unused]] int current_length) {
//   auto num_sequences = state_.params_->BatchBeamSize();
//   std::vector<size_t> sequence_ids(num_sequences);
//   std::iota(sequence_ids.begin(), sequence_ids.end(), 0);
//   for (int i = 0; i < num_sequences; ++i) {
//     paged_cache_->AddToken(i);
//   }

//   size_t input_offset = state_.inputs_.size();

//   block_tables_ = paged_cache_->BlockTables(sequence_ids);
//   state_.inputs_[input_offset_ + layer_count_ * 2] = block_tables_.get();
//   slot_mapping_ = paged_cache_->SlotMapping(sequence_ids);
//   state_.inputs_[input_offset_ + layer_count_ * 2 + 1] = slot_mapping_.get();
// }

std::unique_ptr<CacheManagerInterface> CreateCacheManager(const Model& model, State& state) {
  // if (model.config_->model.kv_cache.paged_cache) {
  //   return std::make_unique<PagedCacheOrchestrator>(model, state);
  // }

  return std::make_unique<KV_Cache>(model, state);
}

}  // namespace Generators
