// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include "ocos.h"
#include "fast_gelu_impl2.cuh"
#include "cuda_type.h"


template <typename T>
struct FastGelu2 {
  template<typename TDict>
  OrtStatusPtr OnModelAttach(const TDict& /*dict*/) {
    return nullptr;
  }
  OrtStatusPtr Compute(Ort::Custom::CUDAKernelContext* ctx,
                       const ortc::Tensor<T>& input,
                       std::optional<const ortc::Tensor<T>*> bias,
                       ortc::Tensor<T>& output) const {
    const T* input_data = input.Data();
    T* output_data = output.Allocate(input.Shape());
    auto input_length = input.NumberOfElement();
    if (0 == input_length) {
      return nullptr;
    }
    const T* bias_data = bias.has_value() ? (*bias)->Data() : nullptr;
    auto bias_length = bias.has_value() ? (*bias)->NumberOfElement() : 0;
    using TT = typename CudaT<T>::MappedType;
    LaunchFastGeluKernel2<TT>(reinterpret_cast<cudaStream_t>(ctx->GetCudaStream()),
                             input_length,
                             bias_length,
                             reinterpret_cast<const TT*>(input_data),
                             reinterpret_cast<const TT*>(bias_data),
                             reinterpret_cast<TT*>(output_data),
                             use_half2_);
    return nullptr;
  }

 private:
  bool use_half2_ = false;  // to-do, read this from env var
};
