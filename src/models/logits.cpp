#include "../generators.h"
#include "model.h"
#include "logits.h"

namespace Generators {

Logits::Logits(const Model& model, State& state)
    : model_{model},
      state_{state},
      shape_{static_cast<int64_t>(state_.params_->batch_size) * state_.params_->search.num_beams, state_.params_->sequence_length, state_.params_->vocab_size},
      type_{model_.session_info_->GetOutputDataType(model_.config_->model.decoder.outputs.logits)} {
  auto logits_tensor = OrtValue::CreateTensor(*model_.allocator_device_, shape_, type_);
  if (type_ == Ort::TypeToTensorType<float>::type)
    value32_ = std::move(logits_tensor);
  else
    value16_ = std::move(logits_tensor);

  if (state_.GetCapturedGraphInfo()) {
    if (type_ == Ort::TypeToTensorType<float>::type) {
      sb_logits32_ = state_.GetCapturedGraphInfo()->sb_logits32_.get();
    }
    if (type_ == Ort::TypeToTensorType<Ort::Float16_t>::type) {
      sb_logits16_ = state_.GetCapturedGraphInfo()->sb_logits16_.get();
    }
  }
}

RoamingArray<float> Logits::Get() {
  size_t element_count = shape_[0] * shape_[1] * shape_[2];

  // Convert from float16 to float32 if necessary
  if (type_ == Ort::TypeToTensorType<Ort::Float16_t>::type) {
#if USE_DML
    DmlHelpers::DmlCastInputToOutput(
        model_.GetDmlExecutionContext(),
        *model_.allocator_device_,
        *value16_,
        value32_,
        model_.GetDmlDevice(),
        model_.GetOrtDmlApi(),
        logits_cast_command_list_state_);
#else
    ConvertFp16ToFp32(*model_.allocator_device_, *value16_, value32_, model_.device_type_, model_.cuda_stream_);
#endif
  }

  // First iteration? Then copy the logits over to a {batch_beams, 1, vocab_size} tensor
  // We'll reuse this tensor for all future iterations
  // The model's output logits are {batch_size*num_beams, input_seq_len, vocab_size}
  if (shape_[1] != 1) {
    const size_t seq_length = shape_[1];
    const size_t vocab_size = shape_[2];
    const size_t num_beams = state_.params_->search.num_beams;

    shape_[1] = 1;

    // bugbug: not done yet
    auto value_next = !sb_logits32_ ? OrtValue::CreateTensor<float>(*model_.allocator_device_, shape_)
                                    : sb_logits32_->CreateTensorOnStaticBuffer(shape_, type_);

    size_t vocab_index = 0;  // Simpler math to have this index go up by vocab_size for every logit chunk we process

    const auto* input_ids = state_.params_->input_ids.data();
    for (int batch_index = 0; batch_index < state_.params_->batch_size; batch_index++) {
      // Find the first non pad token from the end
      size_t token_index = seq_length;
      while (token_index-- > 0) {
        if (input_ids[token_index] != state_.params_->pad_token_id)
          break;
      }

      for (int beam_index = 0; beam_index < num_beams; beam_index++) {
        switch (model_.device_type_) {
#ifdef USE_DML
          case DeviceType::DML: {
            ComPtr<ID3D12Resource> source_resource;
            Ort::ThrowOnError(model_.GetOrtDmlApi()->GetD3D12ResourceFromAllocation(model_.allocator_device_, value32_->GetTensorMutableRawData(), &source_resource));

            ComPtr<ID3D12Resource> target_resource;
            Ort::ThrowOnError(model_.GetOrtDmlApi()->GetD3D12ResourceFromAllocation(model_.allocator_device_, value_next->GetTensorMutableRawData(), &target_resource));

            uint64_t source_offset = (vocab_index * seq_length + token_index * vocab_size) * sizeof(float);
            uint64_t target_offset = vocab_index * sizeof(float);
            uint64_t size_in_bytes = vocab_size * sizeof(float);

            model_.GetDmlExecutionContext()->CopyBufferRegion(
                target_resource.Get(),
                target_offset,
                D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                source_resource.Get(),
                source_offset,
                D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                size_in_bytes);
          } break;
#endif

#if USE_CUDA
          case DeviceType::CUDA: {
            auto logits = std::span<float>{value32_->GetTensorMutableData<float>(), element_count};
            auto logits_next = gpu_span<float>{value_next->GetTensorMutableData<float>(), element_count};
            auto target = logits_next.subspan(vocab_index, vocab_size);
            std::span<const float> source = logits.subspan(vocab_index * seq_length + token_index * vocab_size, vocab_size);
            CudaCheck() == cudaMemcpyAsync(target.data(), source.data(), source.size_bytes(), cudaMemcpyDeviceToDevice, state_.params_->cuda_stream);

          } break;
#endif
          case DeviceType::CPU: {
            auto logits = std::span<float>{value32_->GetTensorMutableData<float>(), element_count};
            auto logits_next = cpu_span<float>{value_next->GetTensorMutableData<float>(), element_count};
            auto target = logits_next.subspan(vocab_index, vocab_size);
            std::span<const float> source = logits.subspan(vocab_index * seq_length + token_index * vocab_size, vocab_size);
            copy(source, target);
          } break;
        }

        vocab_index += vocab_size;
      }

      input_ids += seq_length;
    }

    value32_ = std::move(value_next);
    if (type_ == Ort::TypeToTensorType<Ort::Float16_t>::type)
      value16_ = !sb_logits16_ ? OrtValue::CreateTensor(*model_.allocator_device_, shape_, type_)
                               : sb_logits16_->CreateTensorOnStaticBuffer(shape_, type_);

    state_.outputs_[output_index_] = type_ == Ort::TypeToTensorType<float>::type ? value32_.get() : value16_.get();
  }

#if USE_CUDA
  if (model_.device_type_ == DeviceType::CUDA)
    return gpu_span<float>{value32_->GetTensorMutableData<float>(), element_count};
#elif USE_DML
  if (model_.device_type_ == DeviceType::DML) {
    ComPtr<ID3D12Resource> gpu_resource;
    Ort::ThrowOnError(model_.GetOrtDmlApi()->GetD3D12ResourceFromAllocation(model_.allocator_device_, value32_->GetTensorMutableRawData(), &gpu_resource));
    size_t new_element_count = shape_[0] * shape_[1] * shape_[2];
    return RoamingArray<float>(model_.GetDmlReadbackHeap(), gpu_resource.Get(), new_element_count);
  }
#endif

  return cpu_span<float>{value32_->GetTensorMutableData<float>(), element_count};
}

void Logits::Add() {
  output_index_ = state_.outputs_.size();

  state_.output_names_.push_back(model_.config_->model.decoder.outputs.logits.c_str());
  state_.outputs_.push_back(type_ == Ort::TypeToTensorType<float>::type ? value32_.get() : value16_.get());
}

}  // namespace Generators
