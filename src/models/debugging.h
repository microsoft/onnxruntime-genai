namespace Generators {

size_t SizeOfType(ONNXTensorElementDataType type);

void DumpTensor(std::ostream& stream, OrtValue* value, bool dump_value);
void DumpTensors(std::ostream& stream, OrtValue** values, const char** names, size_t count, bool dump_values);

template <typename T>
void DumpSpan(std::ostream& stream, std::span<const T> values);
template <typename T>
void DumpSpan(std::ostream& stream, std::span<T> values) { DumpSpan(stream, std::span<const T>{values}); }

#if USE_CUDA
template<typename T>
void DumpCudaSpan(std::ostream& stream, std::span<const T> data);
#endif
}  // namespace Generators