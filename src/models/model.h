#pragma once

namespace Generators {

struct Gpt_Model;
struct Llama_Model;
struct Whisper_Model;

std::unique_ptr<OrtValue> ExpandInputs(std::unique_ptr<OrtValue>& input, int num_beams, OrtAllocator& allocator, DeviceType device_type, cudaStream_t cuda_stream);

struct State {
  virtual RoamingArray<float> Run(int current_length, RoamingArray<int32_t> next_tokens, RoamingArray<int32_t> next_indices = {}) = 0;
};

struct Model {

  Model(OrtEnv& ort_env, const char *config_path, const ProviderOptions* provider_options=nullptr);
  ~Model();

  std::vector<int32_t> Generate(const SearchParams &params);

  std::unique_ptr<State> CreateState(RoamingArray<int32_t> sequence_lengths, const SearchParams& params);

  Config config_;
  cudaStream_t cuda_stream_;
  DeviceType device_type_{DeviceType::CPU};

  std::unique_ptr<Gpt_Model> impl_;
  std::unique_ptr<Llama_Model> impl_llama_;
  std::unique_ptr<Whisper_Model> impl_whisper_;
};

}
