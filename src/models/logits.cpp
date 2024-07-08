// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#include "../generators.h"
#include "model.h"
#include "logits.h"
#if USE_CUDA
#include "kernels.h"
#endif

namespace Generators {

Logits::Logits(const Model& model, State& state)
    : model_{model},
      state_{state},
      shape_{static_cast<int64_t>(state_.params_->batch_size) * state_.params_->search.num_beams, state_.params_->sequence_length, state_.params_->vocab_size},
      type_{model_.session_info_->GetOutputDataType(model_.config_->model.decoder.outputs.logits)} {
  output_raw_ = OrtValue::CreateTensor(*model_.allocator_device_, shape_, type_);

  if (state_.GetCapturedGraphInfo()) {
    if (type_ == Ort::TypeToTensorType<float>::type) {
      sb_logits32_ = state_.GetCapturedGraphInfo()->sb_logits32_.get();
    }
    if (type_ == Ort::TypeToTensorType<Ort::Float16_t>::type) {
      sb_logits16_ = state_.GetCapturedGraphInfo()->sb_logits16_.get();
    }
  }

#if USE_CUDA
  if (model_.device_type_ == DeviceType::CUDA && !model_.config_->model.eos_token_ids.empty()) {
    auto& cpu_ids = model_.config_->model.eos_token_ids;
    cuda_eos_token_ids_ptr_ = CudaMallocArray<int32_t>(cpu_ids.size(), &cuda_eos_token_ids_);
    cudaMemcpyAsync(cuda_eos_token_ids_.data(), cpu_ids.data(), cpu_ids.size() * sizeof(int32_t), ::cudaMemcpyHostToDevice, model_.cuda_stream_);
  }
#endif
}

#pragma warning(push)
#pragma warning(disable : 4189)  // local variable is initialized but not referenced

RoamingArray<float> Logits::Get() {
  size_t element_count = shape_[0] * shape_[1] * shape_[2];

  // First iteration? Then copy the logits over to a {batch_beams, 1, vocab_size} tensor
  // The model's output logits are {batch_size*num_beams, input_seq_len, vocab_size}
  OrtValue* logits_of_last_token = output_raw_.get();
  if (shape_[1] != 1) {
    const size_t seq_length = shape_[1];
    const size_t vocab_size = shape_[2];
    const size_t num_beams = state_.params_->search.num_beams;
    const size_t element_count_last_token = shape_[0] * shape_[2];

    shape_[1] = 1;

    // create new OrtValue for logits_of_last_token and use output_last_tokens_ to hold it
    output_last_tokens_ = OrtValue::CreateTensor(*model_.allocator_device_, shape_, type_);
    logits_of_last_token = output_last_tokens_.get();

    size_t element_size = type_ == Ort::TypeToTensorType<float>::type ? 4 : 2;
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
#if USE_DML
          case DeviceType::DML: {
            ComPtr<ID3D12Resource> source_resource;
            Ort::ThrowOnError(model_.GetOrtDmlApi()->GetD3D12ResourceFromAllocation(model_.allocator_device_, output_raw_->GetTensorMutableRawData(), &source_resource));

            ComPtr<ID3D12Resource> target_resource;
            Ort::ThrowOnError(model_.GetOrtDmlApi()->GetD3D12ResourceFromAllocation(model_.allocator_device_, logits_of_last_token->GetTensorMutableRawData(), &target_resource));

            uint64_t source_offset = (vocab_index * seq_length + token_index * vocab_size) * element_size;
            uint64_t target_offset = vocab_index * element_size;
            uint64_t size_in_bytes = vocab_size * element_size;

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

          case DeviceType::CPU:
          case DeviceType::CUDA: {
            auto logits_raw = std::span<const uint8_t>{output_raw_->GetTensorMutableData<uint8_t>(), element_count * element_size};
            auto logits_last_tokens = std::span<uint8_t>{logits_of_last_token->GetTensorMutableData<uint8_t>(), element_count_last_token * element_size};
            auto target = logits_last_tokens.subspan(vocab_index * element_size, vocab_size * element_size);
            auto source = logits_raw.subspan((vocab_index * seq_length + token_index * vocab_size) * element_size, vocab_size * element_size);
            if (model_.device_type_ == DeviceType::CUDA)
#if USE_CUDA
              CudaCheck() == cudaMemcpyAsync(target.data(), source.data(), source.size_bytes(), cudaMemcpyDeviceToDevice, state_.params_->cuda_stream);
#else
              throw std::runtime_error("Unexpected CUDA device usage");
#endif
            else
              copy(source, target);
          } break;
        }

        vocab_index += vocab_size;
      }

      input_ids += seq_length;
    }

    element_count = shape_[0] * shape_[2];  // shape_[1] is now 1, so the element count must be updated
  }

  // Convert from float16 to float32 if necessary
  if (type_ == Ort::TypeToTensorType<Ort::Float16_t>::type) {
    std::unique_ptr<OrtValue> logits_of_last_token_fp32;
#if USE_DML
    if (model_.device_type_ == DeviceType::DML) {
      DmlHelpers::DmlCastInputToOutput(
          model_.GetDmlExecutionContext(),
          *model_.allocator_device_,
          *logits_of_last_token,
          logits_of_last_token_fp32,
          model_.GetDmlDevice(),
          model_.GetOrtDmlApi(),
          logits_cast_command_list_state_);
    } else
#endif
      ConvertFp16ToFp32(*model_.allocator_device_, *logits_of_last_token, logits_of_last_token_fp32, model_.device_type_, model_.cuda_stream_);

    output_last_tokens_ = std::move(logits_of_last_token_fp32);  // use output_last_tokens_ to hold the fp32 logits
    logits_of_last_token = output_last_tokens_.get();
  }

#if USE_DML
  // DML doesn't support on-device scoring yet, so we need to download some data to the CPU
  if (model_.device_type_ == DeviceType::DML) {
    value32_cpu_ = OrtValue::CreateTensor<float>(model_.allocator_cpu_, shape_);
  }
#endif

  assert(shape_[1] == 1);

#if USE_CUDA
  if (model_.device_type_ == DeviceType::CUDA) {
    auto batched_logits_gpu = gpu_span<float>{logits_of_last_token->GetTensorMutableData<float>(), element_count};
    if (cuda_eos_token_ids_ptr_)
      cuda::LaunchHandleEOSArray(
          batched_logits_gpu.data(),
          static_cast<int>(shape_[0]) /* batch_beam_size*/,
          static_cast<int>(shape_[2]) /* vocab_size */,
          cuda_eos_token_ids_.data(),
          static_cast<int>(cuda_eos_token_ids_.size()),
          model_.cuda_stream_);
    return batched_logits_gpu;
  }
#elif USE_DML
  if (model_.device_type_ == DeviceType::DML) {
    // DML doesn't support on-device scoring yet, so we transfer the data to the CPU
    ComPtr<ID3D12Resource> gpu_resource;
    Ort::ThrowOnError(model_.GetOrtDmlApi()->GetD3D12ResourceFromAllocation(
        model_.allocator_device_,
        logits_of_last_token->GetTensorMutableData<float>(),
        &gpu_resource));
    auto cpu_tensor = value32_cpu_->GetTensorMutableData<float>();

    model_.GetDmlReadbackHeap()->ReadbackFromGpu(
        std::span(reinterpret_cast<uint8_t*>(cpu_tensor), element_count * sizeof(float)),
        gpu_resource.Get(),
        0,
        D3D12_RESOURCE_STATE_UNORDERED_ACCESS);

    auto batched_logits_cpu = cpu_span<float>{cpu_tensor, element_count};
    HandleEOSArray(batched_logits_cpu);
    return batched_logits_cpu;
  }
#endif

  auto batched_logits_cpu = cpu_span<float>{logits_of_last_token->GetTensorMutableData<float>(), element_count};
  HandleEOSArray(batched_logits_cpu);
  return batched_logits_cpu;
}

#pragma warning(pop)

void Logits::Update() {
  if (output_raw_.get()->GetTensorTypeAndShapeInfo()->GetShape()[1] == 1) {
    return;
  }

  StaticBuffer* sb_logits = type_ == Ort::TypeToTensorType<Ort::Float16_t>::type ? sb_logits16_ : sb_logits32_;
  output_raw_ = !sb_logits ? OrtValue::CreateTensor(*model_.allocator_device_, shape_, type_)
                           : sb_logits->CreateTensorOnStaticBuffer(shape_, type_);
  state_.outputs_[output_index_] = output_raw_.get();
}

void Logits::HandleEOSArray(cpu_span<float> batched_logits) {
  if (model_.config_->model.eos_token_ids.empty())
    return;

  const size_t vocab_size = shape_[2];
  size_t vocab_index = 0;  // Simpler math to have this index go up by vocab_size for every logit chunk we process

  for (int index = 0; index < shape_[0]; index++) {
    auto logits = batched_logits.subspan(vocab_index, vocab_size);
    float max = std::numeric_limits<float>::lowest();
    for (auto id : model_.config_->model.eos_token_ids) {
      max = std::max(max, logits[id]);
      logits[id] = std::numeric_limits<float>::lowest();  // Set all EOS token options to never happen (the first will get the max of all)
    }

    logits[model_.config_->model.eos_token_id] = max;  // Set the score of the primary EOS token to the highest of any of the EOS tokens
    vocab_index += vocab_size;
  }
}

void Logits::Add() {
  output_index_ = state_.outputs_.size();

  state_.output_names_.push_back(model_.config_->model.decoder.outputs.logits.c_str());
  state_.outputs_.push_back(output_raw_.get());
}

}  // namespace Generators
