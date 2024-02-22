#pragma once
#ifndef NO_TOKENIZER
#include "tfmtok_c.h"
#endif

namespace Generators {

struct Tokenizer;

void ConvertFp16ToFp32(OrtAllocator& allocator, cudaStream_t stream, OrtValue& in, std::unique_ptr<OrtValue>& p_out);

struct State {
  State(const GeneratorParams& search_params);
  virtual ~State() = default;

  virtual RoamingArray<float> Run(int current_length, RoamingArray<int32_t> next_tokens, RoamingArray<int32_t> next_indices = {}) = 0;

  const GeneratorParams& search_params_;

  std::vector<const char*> input_names_, output_names_;
  std::vector<OrtValue*> inputs_, outputs_;

 protected:
  void Run(OrtSession& session);  // Uses the inputs below to run
  void ClearIO();                 // Clear all inputs/outputs
};

#ifdef NO_TOKENIZER
struct TokenizerStream {
  const std::string& Decode(int32_t token);
};

struct Tokenizer {
  Tokenizer(Config& config);

  std::vector<int32_t> Encode(const char* text) const;
  std::string Decode(std::span<int32_t> tokens) const;
};
#else

template <typename T>
struct TfmPtr {
  ~TfmPtr() { TfmDispose(&p_); }
  T** Address() {
    assert(!p_);
    return &p_;
  }
  operator T*() { return p_; }
  operator const T*() const { return p_; }

  T* p_{};
};

struct TokenizerStream {
  TokenizerStream(const Tokenizer& tokenizer) : tokenizer_{tokenizer} {}

  const std::string& Decode(int32_t token);

 private:
  const Tokenizer& tokenizer_;
  std::string chunk_;
};

struct Tokenizer {
  Tokenizer(Config& config);

  std::unique_ptr<TokenizerStream> CreateStream() const;

  std::vector<int32_t> Encode(const char* text) const;
  std::string Decode(std::span<const int32_t> tokens) const;

  TfmPtr<TfmTokenizer> tokenizer_;
};
#endif

struct Model {
  Model(std::unique_ptr<Config> config, const ProviderOptions* provider_options);
  virtual ~Model();

  std::unique_ptr<Tokenizer> CreateTokenizer() const;

  virtual std::unique_ptr<State> CreateState(RoamingArray<int32_t> sequence_lengths, const GeneratorParams& params) const = 0;

  std::unique_ptr<OrtValue> ExpandInputs(std::unique_ptr<OrtValue>& input, int num_beams) const;

  std::unique_ptr<Config> config_;
  std::unique_ptr<OrtSessionOptions> session_options_;
  cudaStream_t cuda_stream_{};
  DeviceType device_type_{DeviceType::CPU};
  Ort::Allocator& allocator_cpu_{Ort::Allocator::GetWithDefaultOptions()};
  Ort::Allocator* allocator_device_{};  // Can be CUDA or CPU based on the DeviceType in the model

 protected:
  void InitDeviceAllocator(OrtSession& session);
};

}  // namespace Generators
