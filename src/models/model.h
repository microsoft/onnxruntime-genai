#pragma once

namespace Generators {

struct Gpt_Model;
struct Llama_Model;

struct State {
  virtual std::span<float> Run(int current_length, std::span<const int32_t> next_tokens, std::span<const int32_t> next_indices = {}) = 0;
};

struct Model {

  Model(OrtEnv& ort_env, const char *config_path, const ProviderOptions* provider_options=nullptr);
  ~Model();

  std::vector<int32_t> Generate(const SearchParams &params);

  std::unique_ptr<State> CreateState(std::span<int32_t> sequence_lengths, const SearchParams& params);

  Config config_;
#if USE_CUDA
  cudaStream_t cuda_stream_;
#endif
  DeviceType device_type_{DeviceType::CPU};

  std::unique_ptr<Gpt_Model> impl_;
  std::unique_ptr<Llama_Model> impl_llama_;
};

}
