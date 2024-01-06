#pragma once

namespace Generators {

std::unique_ptr<OrtValue> ExpandInputs(std::unique_ptr<OrtValue>& input, int num_beams, OrtAllocator& allocator, DeviceType device_type, cudaStream_t cuda_stream);
void ConvertFp16ToFp32(OrtAllocator& allocator, cudaStream_t stream, OrtValue& in, std::unique_ptr<OrtValue>& p_out);

struct State {
  virtual RoamingArray<float> Run(int current_length, RoamingArray<int32_t> next_tokens, RoamingArray<int32_t> next_indices = {}) = 0;
};

struct Model {

  Model(std::unique_ptr<Config> config, OrtEnv& ort_env, const ProviderOptions* provider_options);
  ~Model();

  std::vector<int32_t> Generate(const SearchParams &params);

  virtual std::unique_ptr<State> CreateState(RoamingArray<int32_t> sequence_lengths, const SearchParams& params)=0;

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
};

std::unique_ptr<Model> CreateModel(OrtEnv& ort_env, const char* config_path, const ProviderOptions* provider_options = nullptr);

#if USE_CUDA
namespace cuda {

void LaunchFp16ToFp32(const uint16_t* fp16, float* fp32, int count, cudaStream_t stream);
void LaunchInt32ToInt64(const int32_t* src, int64_t* dst, int count, cudaStream_t stream);

}
#endif

}
