#include "../generators.h"
#include "model.h"
#include "logits.h"

namespace Generators {

Logits::Logits(const Model& model, State& state)
    : model_{model},
      state_{state},
      shape_{static_cast<int64_t>(state_.params_->batch_size) * state_.params_->search.num_beams, state_.params_->sequence_length, state_.params_->vocab_size},
      type_{model_.session_info_->GetOutputDataType(model_.config_->model.decoder.outputs.logits)} {
  auto logits_tensor = OrtValue::CreateTensor(*model.allocator_device_, shape_, type_);
  if (type_ == Ort::TypeToTensorType<float>::type)
    value32_ = std::move(logits_tensor);
  else
    value16_ = std::move(logits_tensor);
}

RoamingArray<float> Logits::Get() {
  size_t element_count = shape_[0] * shape_[1] * shape_[2];

  // Convert from float16 to float32 if necessary
  if (type_ == Ort::TypeToTensorType<Ort::Float16_t>::type)
    ConvertFp16ToFp32(*model_.allocator_device_, *value16_, value32_, model_.device_type_, model_.cuda_stream_);

  // First iteration? Then copy the logits over to a {batch_beams, 1, vocab_size} tensor
  // We'll reuse this tensor for all future iterations
  // The model's output logits are {batch_size*num_beams, input_seq_len, vocab_size}
  if (shape_[1] != 1) {
    const size_t seq_length = shape_[1];
    const size_t vocab_size = shape_[2];
    const size_t num_beams = state_.params_->search.num_beams;

    shape_[1] = 1;
    auto value_next = OrtValue::CreateTensor<float>(*model_.allocator_device_, shape_);
    auto logits_next = cpu_span<float>{value_next->GetTensorMutableData<float>(), element_count};

    size_t vocab_index = 0;  // Simpler math to have this index go up by vocab_size for every logit chunk we process

    const auto* input_ids = state_.params_->input_ids.data();
    for (int batch_index = 0; batch_index < state_.params_->batch_size; batch_index++) {
      // Find the first non pad token from the end
      size_t token_index = seq_length;
      while (token_index-- > 0) {
        if (input_ids[token_index] != state_.params_->pad_token_id)
          break;
      }

      auto logits = std::span<float>{value32_->GetTensorMutableData<float>(), element_count};
      for (int beam_index = 0; beam_index < num_beams; beam_index++) {
        std::span<const float> source = logits.subspan(vocab_index * seq_length + token_index * vocab_size, vocab_size);
        auto target = logits_next.subspan(vocab_index, vocab_size);
#if USE_CUDA
        if (model_.device_type_ == DeviceType::CUDA)
          CudaCheck() == cudaMemcpyAsync(target.data(), source.data(), source.size_bytes(), cudaMemcpyDeviceToDevice, state_.params_->cuda_stream);
        else
#endif
          copy(source, target);

        vocab_index += vocab_size;
      }

      input_ids += seq_length;
    }

    value32_ = std::move(value_next);
    if (type_ == Ort::TypeToTensorType<Ort::Float16_t>::type)
      value16_ = OrtValue::CreateTensor(*model_.allocator_device_, shape_, type_);

    state_.outputs_[output_index_] = type_ == Ort::TypeToTensorType<float>::type ? value32_.get() : value16_.get();
  }

#if USE_CUDA
  if (model_.device_type_ == DeviceType::CUDA)
    return gpu_span<float>{value32_->GetTensorMutableData<float>(), element_count};
#endif
  return cpu_span<float>{value32_->GetTensorMutableData<float>(), element_count};
}

void Logits::Add() {
  output_index_ = state_.outputs_.size();

  state_.output_names_.push_back(model_.config_->model.decoder.outputs.logits.c_str());
  state_.outputs_.push_back(type_ == Ort::TypeToTensorType<float>::type ? value32_.get() : value16_.get());
}

}  // namespace Generators
