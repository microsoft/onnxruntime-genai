// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "generators.h"

namespace Generators {

template <typename T>
void DumpCudaSpan(std::ostream& stream, std::span<const T> data) {
  auto cpu_copy = std::make_unique<T[]>(data.size());
  CudaCheck() == cudaMemcpy(cpu_copy.get(), data.data(), data.size_bytes(), cudaMemcpyDeviceToHost);

  DumpSpan(stream, std::span<const T>{cpu_copy.get(), data.size()});
}

template void DumpCudaSpan(std::ostream&, std::span<const float>);
template void DumpCudaSpan(std::ostream&, std::span<const int32_t>);

}  // namespace Generators