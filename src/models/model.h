// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#pragma once
#include "ortx_tokenizer.h"
#include "captured_graph_pool.h"
#include "utils.h"
#include "phi_image_processor.h"
#include "whisper_processor.h"
#include "phi_multimodal_processor.h"
#include "adapters.h"
#include "extra_outputs.h"

namespace Generators {

struct Tokenizer;

void Cast(OrtValue& input, std::unique_ptr<OrtValue>& output, DeviceInterface& device, ONNXTensorElementDataType type);
void CheckResult(extError_t error);

struct State {
  State(const GeneratorParams& params, const Model& model_);
  virtual ~State();

  virtual DeviceSpan<float> Run(int total_length, DeviceSpan<int32_t>& next_tokens, DeviceSpan<int32_t> next_indices = {}) = 0;
  virtual const CapturedGraphInfo* GetCapturedGraphInfo() const { return nullptr; }
  virtual void Finalize() {}

  void SetTerminate();
  void UnsetTerminate();
  mutable bool session_terminated_{};
  OrtValue* GetInput(const char* name);

  virtual void RewindTo(size_t index) { (void)index; };

  virtual OrtValue* GetOutput(const char* name);

  void ClearIO();  // Clear all inputs/outputs

  void SetActiveAdapter(Adapters* adapters, const std::string& adapter_name);

  const Model& model_;

  std::shared_ptr<const GeneratorParams> params_;

  std::vector<const char*> input_names_, output_names_;
  std::vector<std::string> adapter_names_;
  std::vector<OrtValue*> inputs_, outputs_;

 protected:
  void Run(OrtSession& session, int new_batch_size);  // Uses the inputs below to run
  bool first_run_{true};

  std::unique_ptr<OrtRunOptions> run_options_;

 private:
  int current_batch_size_{0};
  std::shared_ptr<Adapters> adapters_;
  ExtraOutputs extra_outputs_;
};

struct TokenizerStream : LeakChecked<TokenizerStream> {
  TokenizerStream(const Tokenizer& tokenizer);

  const std::string& Decode(int32_t token);

 private:
  std::shared_ptr<const Tokenizer> tokenizer_;
  OrtxPtr<OrtxObject> cache_;
  std::string chunk_;
};

// Turn an array of ragged token sequences into a 2D input suitable for batching. Handles padding for the model
// Sequence length is vector.size()/count
std::vector<int32_t> PadInputs(std::span<std::span<const int32_t>> sequences, int32_t pad_token_id);

struct Tokenizer : std::enable_shared_from_this<Tokenizer>, LeakChecked<Tokenizer>, ExternalRefCounted<Tokenizer> {
  Tokenizer(Config& config);

  std::unique_ptr<TokenizerStream> CreateStream() const;

  std::vector<int32_t> Encode(const char* text) const;
  std::string Decode(std::span<const int32_t> tokens) const;

  std::vector<int32_t> EncodeBatch(std::span<const std::string> strings) const;
  std::shared_ptr<Tensor> EncodeBatch(std::span<const char*> strings) const;
  std::vector<std::string> DecodeBatch(std::span<const int32_t> sequences, size_t count) const;

  int32_t TokenToTokenId(const char* token) const;

  OrtxPtr<OrtxTokenizer> tokenizer_;

 private:
  int32_t pad_token_id_;
};

struct MultiModalProcessor : std::enable_shared_from_this<MultiModalProcessor>, ExternalRefCounted<MultiModalProcessor> {
  MultiModalProcessor(Config& config, const SessionInfo& session_info);

  std::unique_ptr<NamedTensors> Process(const std::string& prompt, const Images* images, const Audios* audios) const;

  std::shared_ptr<Tokenizer> tokenizer_;
  std::shared_ptr<Processor> processor_;
};

struct SessionInfo {
  SessionInfo(OrtSession& session);

  void Add(OrtSession& session);

  bool HasInput(const std::string& name) const;
  bool HasOutput(const std::string& name) const;

  ONNXTensorElementDataType GetInputDataType(const std::string& name) const;
  ONNXTensorElementDataType GetOutputDataType(const std::string& name) const;

  std::vector<std::string> GetInputNames() const;

 private:
  std::unordered_map<std::string, ONNXTensorElementDataType> inputs_, outputs_;
};

struct Model : std::enable_shared_from_this<Model>, LeakChecked<Model>, ExternalRefCounted<Model> {
  Model(std::unique_ptr<Config> config);
  virtual ~Model();

  std::shared_ptr<Tokenizer> CreateTokenizer() const;

  std::shared_ptr<MultiModalProcessor> CreateMultiModalProcessor() const;

  virtual std::unique_ptr<State> CreateState(DeviceSpan<int32_t> sequence_lengths, const GeneratorParams& params) const = 0;

  std::unique_ptr<OrtValue> ExpandInputs(std::unique_ptr<OrtValue>& input, int num_beams) const;

  CapturedGraphPool* GetCapturedGraphPool() const { return captured_graph_pool_.get(); }

  OrtSessionOptions* GetSessionOptions(const std::string& model_id) const;

  std::unique_ptr<Config> config_;
  std::unique_ptr<OrtSessionOptions> session_options_;

  mutable DeviceInterface* p_device_{};          // The device we're running on (matches device_type_) used for things that work the same on all devices
  mutable DeviceInterface* p_device_inputs_{};   // For some model inputs, the device might be the CPU device (all but KV cache currently for WebGPU and DML)
  mutable DeviceInterface* p_device_kvcache_{};  // The kvcache is always allocated in device memory  (TODO: Remove in favor of just p_device_?)

  Ort::Allocator& allocator_cpu_{GetDeviceInterface(DeviceType::CPU)->GetAllocator()};

  std::unique_ptr<SessionInfo> session_info_;

 protected:
  void InitDeviceAllocator(OrtSession& session);
  void CreateSessionOptions();

  void CreateSessionOptionsFromConfig(const Config::SessionOptions& config_session_options,
                                      OrtSessionOptions& session_options,
                                      bool is_primary_session_options,
                                      bool disable_graph_capture);

  std::shared_ptr<CapturedGraphPool> captured_graph_pool_;
  std::map<std::string, std::unique_ptr<OrtSessionOptions>> pipeline_session_options_;
};

}  // namespace Generators
