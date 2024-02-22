#include "../generators.h"
#include "model.h"
#include "position_ids.h"
#include "kernels.h"

namespace Generators {

template <typename T>
PositionIDs<T>::PositionIDs(const Model& model, State& state, RoamingArray<int32_t>& sequence_lengths_unk)
    : model_{model},
      state_{state} {
  cpu_span<int32_t> const sequence_lengths = sequence_lengths_unk;
  std::array<int64_t, 2> shape{state_.search_params_.batch_size, state_.search_params_.sequence_length};  // Only batch_size initially, as we haven't expanded over the beams yet
  position_ids_ = OrtValue::CreateTensor<T>(model.allocator_cpu_, shape);
  attention_mask_ = OrtValue::CreateTensor<T>(model.allocator_cpu_, shape);

  initial_sequence_lengths_.resize(state_.search_params_.batch_size * state_.search_params_.num_beams);

  // Set attention mask to be 0 for pad tokens, and 1 for all other tokens.
  // Set position id to be 0 for pad tokens, and accumulated sum of mask in a batch for other tokens
  auto* mask_data = attention_mask_->GetTensorMutableData<T>();
  auto* position_data = position_ids_->GetTensorMutableData<T>();
  const auto* word_id = state_.search_params_.input_ids.data();
  auto* mask = mask_data;
  auto* position = position_data;
  for (int i = 0; i < shape[0]; i++) {
    T abs_position = 0;
    for (int j = 0; j < shape[1]; j++, word_id++, mask++, position++) {
      if (*word_id == state_.search_params_.pad_token_id) {
        *mask = 0;
        *position = 0;
      } else {
        *mask = 1;
        *position = abs_position++;
      }
    }

    for (int k = 0; k < state_.search_params_.num_beams; k++) {
      sequence_lengths[i * state_.search_params_.num_beams + k] = static_cast<int32_t>(abs_position);
      initial_sequence_lengths_[i * state_.search_params_.num_beams + k] = abs_position;
    }
  }

  position_ids_ = model_.ExpandInputs(position_ids_, state_.search_params_.num_beams);
  attention_mask_ = model_.ExpandInputs(attention_mask_, state_.search_params_.num_beams);
  shape[0] *= state_.search_params_.num_beams;
  position_ids_shape_ = shape;
  attention_mask_shape_ = shape;
}

template <typename T>
void PositionIDs<T>::Add() {
  input_index_ = state_.inputs_.size();

  state_.inputs_.push_back(position_ids_.get());
  state_.input_names_.push_back("position_ids");

  state_.inputs_.push_back(attention_mask_.get());
  state_.input_names_.push_back("attention_mask");
}

template <typename T>
void PositionIDs<T>::Update(int current_length) {
  // Reallocate position_ids for the 2nd and onward shape
  if (initial_sequence_lengths_.size()) {
    position_ids_shape_[1] = 1;
    position_ids_ = OrtValue::CreateTensor<T>(*model_.allocator_device_, position_ids_shape_);
    state_.inputs_[input_index_] = position_ids_.get();

    // Copy the initial values over to the device specific tensor
    switch (model_.device_type_) {
      case DeviceType::CPU:
        std::copy(initial_sequence_lengths_.begin(), initial_sequence_lengths_.end(), position_ids_->GetTensorMutableData<T>());
        break;
#if USE_CUDA
      case DeviceType::CUDA:
        cudaMemcpyAsync(position_ids_->GetTensorMutableRawData(), initial_sequence_lengths_.data(), sizeof(T) * initial_sequence_lengths_.size(), cudaMemcpyHostToDevice, model_.cuda_stream_);
        break;
#endif
      default:
        throw std::runtime_error("PositionIDs::Update - Unsupported device type");
    }
    initial_sequence_lengths_.clear();
  } else { // Just incrementing existing position IDs
    switch (model_.device_type_) {
      case DeviceType::CPU: {
        // Increment position IDs
        auto* data = position_ids_->GetTensorMutableData<T>();
        for (int i = 0; i < position_ids_shape_[0]; i++) {
          data[i]++;
        }
        break;
      }
#if USE_CUDA
      case DeviceType::CUDA:
        cuda::Launch_UpdatePositionIds(position_ids_->GetTensorMutableData<T>(), static_cast<int>(position_ids_shape_[0]), model_.cuda_stream_);
        break;
#endif
      default:
        throw std::runtime_error("PositionIDs::Update - Unsupported device type");
    }
  }

  {
    // Update attention mask
    assert(attention_mask_shape_[1] == current_length - 1);  // We should always be growing by 1
    attention_mask_shape_[1] = current_length;

    const auto* old_data = attention_mask_->GetTensorData<T>();
    std::unique_ptr<OrtValue> attention_mask = OrtValue::CreateTensor<T>(*model_.allocator_device_, attention_mask_shape_);
    auto* data = attention_mask->GetTensorMutableData<T>();

    switch (model_.device_type_) {
      case DeviceType::CPU:
        for (int i = 0; i < attention_mask_shape_[0]; i++) {
          for (int j = 0; j < current_length - 1; j++) {
            data[i * current_length + j] = old_data[i * (current_length - 1) + j];
          }
          data[i * current_length + current_length - 1] = 1;
        }
        break;
#if USE_CUDA
      case DeviceType::CUDA:
        cuda::Launch_UpdateAttentionMask(data, old_data, static_cast<int>(attention_mask_shape_[0]), current_length, model_.cuda_stream_);
        break;
#endif
      default:
        throw std::runtime_error("PositionIDs::Update - Unsupported device type");
    }
    attention_mask_ = std::move(attention_mask);
    state_.inputs_[input_index_ + 1] = attention_mask_.get();
  }
}

template struct PositionIDs<int32_t>;
template struct PositionIDs<int64_t>;

}  // namespace Generators
