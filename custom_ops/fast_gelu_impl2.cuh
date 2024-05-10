// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include <cuda.h>
#include <cuda_runtime.h>

template <typename T>
cudaError_t LaunchFastGeluKernel2(cudaStream_t stream, int input_length, int bias_length,
                                 const T* input, const T* bias, T* output, bool use_half2);