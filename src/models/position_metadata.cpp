#include "../generators.h"
#include "model.h"
#include "position_metadata.h"
#include "kernels.h"

namespace Generators {

PositionMetadata::PositionMetadata(const Model& model, State& state, RoamingArray<int32_t>& sequence_lengths_unk)
    : model_{model},
      state_{state} {
  type_ = model_.session_info_->GetInputDataType(model_.config_->model.decoder.inputs.position_ids);
  if (type_ != Ort::TypeToTensorType<int32_t>::type && type_ != Ort::TypeToTensorType<int64_t>::type)
    throw std::runtime_error("position_ids & attention_mask only support int32 or int64 types");

  std::array<int64_t, 2> shape{state_.params_.batch_size, state_.params_.sequence_length};  // Only batch_size initially, as we haven't expanded over the beams yet
  position_ids_ = OrtValue::CreateTensor(model.allocator_cpu_, shape, type_);
  position_ids_next_ = OrtValue::CreateTensor(model.allocator_cpu_, std::array<int64_t, 2>{shape[0], 1}, type_);
  attention_mask_ = OrtValue::CreateTensor(model.allocator_cpu_, shape, type_);

  initial_sequence_lengths_.resize(state_.params_.BatchBeamSize());

  if (type_ == Ort::TypeToTensorType<int32_t>::type)
    InitializeTensors<int32_t>(shape, sequence_lengths_unk);
  else
    InitializeTensors<int64_t>(shape, sequence_lengths_unk);

  position_ids_ = model_.ExpandInputs(position_ids_, state_.params_.search.num_beams);
  position_ids_next_ = model_.ExpandInputs(position_ids_next_, state_.params_.search.num_beams);
  attention_mask_ = model_.ExpandInputs(attention_mask_, state_.params_.search.num_beams);
  shape[0] *= state_.params_.search.num_beams;
  position_ids_shape_ = shape;
  attention_mask_shape_ = shape;

  if (model_.device_type_ == DeviceType::CUDA && model_.config_->use_cuda_graphs) {
    size_t max_beam_batch_size = model_.config_->search.num_beams * model_.config_->max_batch_size;
    sb_position_ids_ = std::make_unique<StaticBuffer>(model_.allocator_device_, max_beam_batch_size);
    sb_seqlens_k_ = std::make_unique<StaticBuffer>(model_.allocator_device_, max_beam_batch_size);
  }
}

void PositionMetadata::AddAttentionMask() {
  mask_input_index_ = state_.inputs_.size();

  state_.inputs_.push_back(attention_mask_.get());
  state_.input_names_.push_back(model_.config_->model.decoder.inputs.attention_mask.c_str());
}

void PositionMetadata::AddPositionIDs() {
  posid_input_index_ = state_.inputs_.size();

  state_.inputs_.push_back(position_ids_.get());
  state_.input_names_.push_back(model_.config_->model.decoder.inputs.position_ids.c_str());
}

void PositionMetadata::AddSeqlensK() {
  seqlens_k_input_index_ = state_.inputs_.size();

  senlens_k_shape_ = {static_cast<int64_t>(state_.params_.batch_size) * state_.params_.search.num_beams};
  seqlens_k_ = OrtValue::CreateTensor(model_.allocator_cpu_, senlens_k_shape_, Ort::TypeToTensorType<int32_t>::type);

  std::copy(initial_sequence_lengths_.begin(),
            initial_sequence_lengths_.end(),
            seqlens_k_->GetTensorMutableData<int32_t>());

  state_.inputs_.push_back(seqlens_k_.get());
  state_.input_names_.push_back(model_.config_->model.decoder.inputs.seqlens_k.c_str());
}

void PositionMetadata::AddTotalSequenceLength() {
  total_sequence_length_input_index_ = state_.inputs_.size();
  total_sequence_length_ = OrtValue::CreateTensor(model_.allocator_cpu_,
                                                  total_sequence_length_shape_,
                                                  Ort::TypeToTensorType<int32_t>::type);

  total_sequence_length_->GetTensorMutableData<int32_t>()[0] = state_.params_.sequence_length;
  state_.inputs_.push_back(total_sequence_length_.get());
  state_.input_names_.push_back(model_.config_->model.decoder.inputs.total_sequence_length.c_str());
}

void PositionMetadata::UpdatePositionIDs(int current_length) {
  // Reallocate position_ids for the 2nd and onward shape
  if (is_first_posid_update_) {
    position_ids_shape_[1] = 1;
    if (!sb_position_ids_) {
      position_ids_ = std::move(position_ids_next_);
    } else {
#if USE_CUDA
      position_ids_ = sb_position_ids_->GetOrCreateTensor(position_ids_shape_, type_);
      assert(model_.device_type_ == DeviceType::CUDA);
      if (type_ == Ort::TypeToTensorType<int32_t>::type) {
        cudaMemcpyAsync(position_ids_->GetTensorMutableRawData(),
                        position_ids_next_->GetTensorData<int32_t>(),
                        sizeof(int32_t) * position_ids_shape_[0],
                        cudaMemcpyDeviceToDevice,
                        model_.cuda_stream_);
      } else {
        cudaMemcpyAsync(position_ids_->GetTensorMutableRawData(),
                        position_ids_next_->GetTensorData<int64_t>(),
                        sizeof(int64_t) * position_ids_shape_[0],
                        cudaMemcpyDeviceToDevice,
                        model_.cuda_stream_);
      }
#endif
    }
    is_first_posid_update_ = false;
    state_.inputs_[posid_input_index_] = position_ids_.get();
  } else {  // Just incrementing existing position IDs
    switch (model_.device_type_) {
      case DeviceType::CPU: {
        if (type_ == Ort::TypeToTensorType<int32_t>::type)
          UpdatePositionIDsImpl<int32_t>();
        else
          UpdatePositionIDsImpl<int64_t>();
        break;
      }
#if USE_CUDA
      case DeviceType::CUDA:
        if (type_ == Ort::TypeToTensorType<int32_t>::type)
          cuda::Launch_UpdatePositionIds(position_ids_->GetTensorMutableData<int32_t>(), static_cast<int>(position_ids_shape_[0]), model_.cuda_stream_);
        else
          cuda::Launch_UpdatePositionIds(position_ids_->GetTensorMutableData<int64_t>(), static_cast<int>(position_ids_shape_[0]), model_.cuda_stream_);
        break;
#endif
      default:
        throw std::runtime_error("PositionIDs::Update - Unsupported device type");
    }
  }
}

void PositionMetadata::UpdateSeqlensK(int current_length) {
#if USE_CUDA
  assert(type_ == Ort::TypeToTensorType<int32_t>::type);
  assert(model_.device_type_ == DeviceType::CUDA);

  if (is_first_seqlen_update_) {
    if (!sb_seqlens_k_) {
      seqlens_k_ = OrtValue::CreateTensor(*model_.allocator_device_, senlens_k_shape_, Ort::TypeToTensorType<int32_t>::type);
    } else {
      seqlens_k_ = sb_seqlens_k_->GetOrCreateTensor(senlens_k_shape_, Ort::TypeToTensorType<int32_t>::type);
    }
    state_.inputs_[seqlens_k_input_index_] = seqlens_k_.get();
    cudaMemcpyAsync(seqlens_k_->GetTensorMutableRawData(), initial_sequence_lengths_.data(), sizeof(int32_t) * initial_sequence_lengths_.size(), cudaMemcpyHostToDevice, model_.cuda_stream_);
    is_first_seqlen_update_ = false;
  } else {
    cuda::Launch_UpdatePositionIds(seqlens_k_->GetTensorMutableData<int32_t>(), static_cast<int>(senlens_k_shape_[0]), model_.cuda_stream_);
  }
#endif
}

void PositionMetadata::UpdateTotalSequenceLength(int current_length) {
  total_sequence_length_->GetTensorMutableData<int32_t>()[0] = current_length;
}

void PositionMetadata::UpdateAttentionMask(int current_length) {
  {
    // Update attention mask
    assert(attention_mask_shape_[1] == current_length - 1);  // We should always be growing by 1
    attention_mask_shape_[1] = current_length;

    std::unique_ptr<OrtValue> next_attention_mask = OrtValue::CreateTensor(*model_.allocator_device_, attention_mask_shape_, type_);

    switch (model_.device_type_) {
      case DeviceType::CPU: {
        if (type_ == Ort::TypeToTensorType<int32_t>::type)
          UpdateAttentionMaskImpl(next_attention_mask->GetTensorMutableData<int32_t>(), attention_mask_->GetTensorData<int32_t>(), current_length);
        else
          UpdateAttentionMaskImpl(next_attention_mask->GetTensorMutableData<int64_t>(), attention_mask_->GetTensorData<int64_t>(), current_length);
        break;
      }
#if USE_CUDA
      case DeviceType::CUDA:
        if (type_ == Ort::TypeToTensorType<int32_t>::type)
          cuda::Launch_UpdateAttentionMask(next_attention_mask->GetTensorMutableData<int32_t>(), attention_mask_->GetTensorData<int32_t>(), static_cast<int>(attention_mask_shape_[0]), current_length, model_.cuda_stream_);
        else
          cuda::Launch_UpdateAttentionMask(next_attention_mask->GetTensorMutableData<int64_t>(), attention_mask_->GetTensorData<int64_t>(), static_cast<int>(attention_mask_shape_[0]), current_length, model_.cuda_stream_);
        break;
#endif
      default:
        throw std::runtime_error("PositionIDs::Update - Unsupported device type");
    }
    attention_mask_ = std::move(next_attention_mask);
    state_.inputs_[mask_input_index_] = attention_mask_.get();
  }
}

template <typename T>
void PositionMetadata::InitializeTensors(std::array<int64_t, 2> shape, cpu_span<int32_t> sequence_lengths) {
  // Set attention mask to be 0 for pad tokens, and 1 for all other tokens.
  // Set position id to be 0 for pad tokens, and accumulated sum of mask in a batch for other tokens
  auto* mask_data = attention_mask_->GetTensorMutableData<T>();
  auto* position_data = position_ids_->GetTensorMutableData<T>();
  auto* position_data_next = position_ids_next_->GetTensorMutableData<T>();
  const auto* word_id = state_.params_.input_ids.data();
  auto* mask = mask_data;
  auto* position = position_data;
  for (int i = 0; i < shape[0]; i++) {
    T abs_position = 0;
    for (int j = 0; j < shape[1]; j++, word_id++, mask++, position++) {
      if (*word_id == state_.params_.pad_token_id) {
        *mask = 0;
        *position = 0;
      } else {
        *mask = 1;
        *position = abs_position++;
      }
    }

    position_data_next[i] = abs_position;
    for (int k = 0; k < state_.params_.search.num_beams; k++) {
      sequence_lengths[i * state_.params_.search.num_beams + k] = static_cast<int32_t>(abs_position);
      initial_sequence_lengths_[i * state_.params_.search.num_beams + k] = static_cast<int32_t>(abs_position);
    }
  }
}

template <typename T>
void PositionMetadata::UpdatePositionIDsImpl() {
  // Increment position IDs
  auto* data = position_ids_->GetTensorMutableData<T>();
  for (int i = 0; i < position_ids_shape_[0]; i++) {
    data[i]++;
  }
};

template <typename T>
void PositionMetadata::UpdateAttentionMaskImpl(T* data, const T* old_data, int current_length) {
  for (int i = 0; i < attention_mask_shape_[0]; i++) {
    for (int j = 0; j < current_length - 1; j++) {
      data[i * current_length + j] = old_data[i * (current_length - 1) + j];
    }
    data[i * current_length + current_length - 1] = 1;
  }
};

}  // namespace Generators
