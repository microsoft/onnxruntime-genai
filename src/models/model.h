#pragma once
#ifndef NO_TOKENIZER
#include "tfmtok_c.h"
#endif

namespace Generators {

struct Tokenizer;

void ConvertFp16ToFp32(OrtAllocator& allocator, OrtValue& in, std::unique_ptr<OrtValue>& p_out, DeviceType device_type, cudaStream_t stream);

struct State {
  State(const GeneratorParams& params);
  virtual ~State() = default;

  virtual RoamingArray<float> Run(int current_length, RoamingArray<int32_t> next_tokens, RoamingArray<int32_t> next_indices = {}) = 0;

  std::shared_ptr<const GeneratorParams> params_;

  std::vector<const char*> input_names_, output_names_;
  std::vector<OrtValue*> inputs_, outputs_;

 protected:
  void Run(OrtSession& session, OrtRunOptions& run_options);  // Uses the inputs below to run
  void ClearIO();                                             // Clear all inputs/outputs
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
  TokenizerStream(const Tokenizer& tokenizer);

  const std::string& Decode(int32_t token);

 private:
  std::shared_ptr<const Tokenizer> tokenizer_;
  TfmPtr<TfmObject> cache_;
  std::string chunk_;
};

// Turn an array of ragged token sequences into a 2D input suitable for batching. Handles padding for the model
// Sequence length is vector.size()/count
std::vector<int32_t> PadInputs(std::span<std::span<const int32_t> > sequences, int32_t pad_token_id);

struct Tokenizer : std::enable_shared_from_this<Tokenizer> {
  Tokenizer(Config& config);

  std::unique_ptr<TokenizerStream> CreateStream() const;

  std::vector<int32_t> Encode(const char* text) const;
  std::string Decode(std::span<const int32_t> tokens) const;

  std::vector<int32_t> EncodeBatch(std::span<const std::string> strings) const;
  std::vector<std::string> DecodeBatch(std::span<const int32_t> sequences, size_t count) const;

  TfmPtr<TfmTokenizer> tokenizer_;
  std::shared_ptr<Tokenizer> external_owner_;  // Set to 'this' when created by the C API to preserve lifetime

 private:
  int32_t pad_token_id_;
};
#endif

struct SessionInfo {
  SessionInfo(OrtSession& session);

  bool HasInput(const std::string& name) const;
  bool HasOutput(const std::string& name) const;

  ONNXTensorElementDataType GetInputDataType(const std::string& name) const;
  ONNXTensorElementDataType GetOutputDataType(const std::string& name) const;

 private:
  std::unordered_map<std::string, ONNXTensorElementDataType> inputs_, outputs_;
};

struct Model : std::enable_shared_from_this<Model> {
  Model(std::unique_ptr<Config> config);
  virtual ~Model();

  std::shared_ptr<Tokenizer> CreateTokenizer() const;

  virtual std::unique_ptr<State> CreateState(RoamingArray<int32_t> sequence_lengths, const GeneratorParams& params) const = 0;

  std::unique_ptr<OrtValue> ExpandInputs(std::unique_ptr<OrtValue>& input, int num_beams) const;

  void GetMaxBatchSizeFromGeneratorParams(const GeneratorParams& params);

  std::unique_ptr<Config> config_;
  std::unique_ptr<OrtSessionOptions> session_options_;
  std::unique_ptr<OrtRunOptions> run_options_;

  cuda_stream_holder cuda_stream_;
  DeviceType device_type_{DeviceType::CPU};
  Ort::Allocator& allocator_cpu_{Ort::Allocator::GetWithDefaultOptions()};
  Ort::Allocator* allocator_device_{};  // Can be CUDA or CPU based on the DeviceType in the model

  std::unique_ptr<SessionInfo> session_info_;

  std::shared_ptr<Model> external_owner_;  // Set to 'this' when created by the C API to preserve lifetime

  bool use_cuda_graph_{};
  int max_batch_size_{};

 protected:
  void InitDeviceAllocator(OrtSession& session);
  void CreateSessionOptions();
};

}  // namespace Generators
