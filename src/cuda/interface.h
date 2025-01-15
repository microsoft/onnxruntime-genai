// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

struct GenaiInterface {
#if _WIN32
  virtual void* HeapAllocate(size_t size) = 0;
  virtual void HeapFree(void*) = 0;
#endif

  virtual void CopyThroughCpu(Generators::DeviceBuffer& dest, size_t begin_dest, Generators::DeviceBuffer& source, size_t begin_source, size_t size_in_bytes) = 0;

  virtual Generators::LogItems& GetLogItems() = 0;
  virtual std::ostream& operator_leftshift(std::ostream& stream, Generators::SGR sgr_code) = 0;
  virtual std::ostream& Log(std::string_view label, std::string_view text = {}) = 0;

  virtual void DumpSpan(std::ostream& stream, std::span<const float> values) = 0;
  virtual void DumpSpan(std::ostream& stream, std::span<const int> values) = 0;

  virtual void Sequences_AfterAppendNextTokens(Generators::Sequences* p_this, Generators::DeviceSpan<int32_t> next_tokens, size_t batch_beam_size) = 0;
  virtual void Sequences_RewindTo(Generators::Sequences* p_this, size_t new_length) = 0;
};

namespace Generators {
LogItems& GetLogItems();
DeviceInterface& GetCudaDeviceInterface();
}  // namespace Generators
