namespace Generators {

void DumpTensor(OrtValue* value, bool dump_value);
void DumpTensors(OrtValue** values, const char** names, size_t count, bool dump_values);

void DumpMemory(const char* name, gsl::span<const int32_t> data);
void DumpMemory(const char* name, gsl::span<const float> data);

#if USE_CUDA
void DumpCudaMemory(const char* name, gsl::span<const int32_t> data);
void DumpCudaMemory(const char* name, gsl::span<const float> data);
#endif
}