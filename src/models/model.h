#pragma once

namespace Generators {

void ConvertFp16ToFp32(OrtAllocator& allocator, cudaStream_t stream, OrtValue& in, std::unique_ptr<OrtValue>& p_out);

struct State {
  State(const SearchParams& search_params);
  virtual ~State() = default;

  virtual RoamingArray<float> Run(int current_length, RoamingArray<int32_t> next_tokens, RoamingArray<int32_t> next_indices = {}) = 0;

  const SearchParams& search_params_;

  std::vector<const char*> input_names_, output_names_;
  std::vector<OrtValue*> inputs_, outputs_;

 protected:
  void Run(OrtSession& session);  // Uses the inputs below to run
  void ClearIO();                 // Clear all inputs/outputs
};

struct Model {
  Model(std::unique_ptr<Config> config, OrtEnv& ort_env, const ProviderOptions* provider_options);
  virtual ~Model();

  std::vector<int32_t> Generate(const SearchParams& params);

  virtual std::unique_ptr<State> CreateState(RoamingArray<int32_t> sequence_lengths, const SearchParams& params) = 0;

  std::unique_ptr<OrtValue> ExpandInputs(std::unique_ptr<OrtValue>& input, int num_beams);

  std::unique_ptr<Config> config_;
  std::unique_ptr<OrtSessionOptions> session_options_;
  cudaStream_t cuda_stream_;
  DeviceType device_type_{DeviceType::CPU};
  Ort::Allocator& allocator_cpu_{Ort::Allocator::GetWithDefaultOptions()};

  std::unique_ptr<OrtMemoryInfo> memory_info_cuda_;
  std::unique_ptr<Ort::Allocator> allocator_cuda_;
  Ort::Allocator* allocator_device_{};  // Can be CUDA or CPU based on the DeviceType in the model

  bool logits_uses_seq_len_{};  // Logits shape is [... seq_len, vocab_size ] vs [... 1, vocab_size ]
  ONNXTensorElementDataType score_type_;

 protected:
  void InitDeviceAllocator(OrtSession& session);
  void ValidateLogits(OrtTypeInfo& info);
};

std::unique_ptr<Model> CreateModel(OrtEnv& ort_env, const char* config_path, const ProviderOptions* provider_options = nullptr);

#if USE_CUDA
namespace cuda {

void LaunchFp16ToFp32(const uint16_t* fp16, float* fp32, int count, cudaStream_t stream);
void LaunchInt32ToInt64(const int32_t* src, int64_t* dst, int count, cudaStream_t stream);

}  // namespace cuda
#endif

}  // namespace Generators
