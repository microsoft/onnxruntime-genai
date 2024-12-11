// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#pragma once
#include "ortx_tokenizer.h"
#include "captured_graph_pool.h"
#include "utils.h"
#include "prompt_image_processor.h"
#include "audio_processor.h"
#include "adapters.h"

#if USE_DML
#include "dml_provider_factory.h"
#include "../dml/dml_helpers.h"
#include "../dml/dml_execution_context.h"
#include "../dml/dml_pooled_upload_heap.h"
#include "../dml/dml_readback_heap.h"
#endif

namespace Generators {

struct Tokenizer;

void ConvertFp16ToFp32(OrtAllocator& allocator, OrtValue& in, std::unique_ptr<OrtValue>& p_out, DeviceType device_type, cudaStream_t stream);

void ConvertFp32ToFp16(OrtAllocator& allocator, OrtValue& in, std::unique_ptr<OrtValue>& p_out, DeviceType device_type, cudaStream_t stream);

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

struct Tokenizer : std::enable_shared_from_this<Tokenizer>, LeakChecked<Tokenizer> {
  Tokenizer(Config& config);

  std::unique_ptr<TokenizerStream> CreateStream() const;

  std::vector<int32_t> Encode(const char* text) const;
  std::string Decode(std::span<const int32_t> tokens) const;

  std::vector<int32_t> EncodeBatch(std::span<const std::string> strings) const;
  std::vector<std::string> DecodeBatch(std::span<const int32_t> sequences, size_t count) const;

  int32_t TokenToTokenId(const char* token) const;

  OrtxPtr<OrtxTokenizer> tokenizer_;
  std::shared_ptr<Tokenizer> external_owner_;  // Set to 'this' when created by the C API to preserve lifetime

 private:
  int32_t pad_token_id_;
};

struct MultiModalProcessor : std::enable_shared_from_this<MultiModalProcessor> {
  MultiModalProcessor(Config& config, const SessionInfo& session_info);

  std::shared_ptr<Tokenizer> tokenizer_;
  std::shared_ptr<ImageProcessor> image_processor_;
  std::shared_ptr<AudioProcessor> audio_processor_;

  std::shared_ptr<MultiModalProcessor> external_owner_;  // Set to 'this' when created by the C API to preserve lifetime
};

struct SessionInfo {
  SessionInfo(OrtSession& session);

  void Add(OrtSession& session);

  bool HasInput(const std::string& name) const;
  bool HasOutput(const std::string& name) const;

  ONNXTensorElementDataType GetInputDataType(const std::string& name) const;
  ONNXTensorElementDataType GetOutputDataType(const std::string& name) const;

 private:
  std::unordered_map<std::string, ONNXTensorElementDataType> inputs_, outputs_;
};

struct Model : std::enable_shared_from_this<Model>, LeakChecked<Model> {
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

  cudaStream_t cuda_stream_{};
  DeviceInterface* p_device_{};
  DeviceType device_type_{DeviceType::CPU};
  Ort::Allocator& allocator_cpu_{Ort::Allocator::GetWithDefaultOptions()};
  Ort::Allocator* allocator_device_{};   // Can be CUDA or CPU based on the DeviceType in the model
  Ort::Allocator* allocator_kvcache_{};  // keep allocator for kv_cache seperate to allow that only kv_cache is on device

  std::unique_ptr<SessionInfo> session_info_;

  std::shared_ptr<Model> external_owner_;  // Set to 'this' when created by the C API to preserve lifetime

#if USE_DML
  DmlExecutionContext* GetDmlExecutionContext() const { return dml_execution_context_.get(); }
  DmlReadbackHeap* GetDmlReadbackHeap() const { return dml_readback_heap_.get(); }
  DmlPooledUploadHeap* GetDmlUploadHeap() const { return dml_pooled_upload_heap_.get(); }
  const OrtDmlApi* GetOrtDmlApi() const { return p_dml_api_; }
  IDMLDevice* GetDmlDevice() const { return dml_device_.Get(); }
  ID3D12Device* GetD3D12Device() const { return dml_objects_.d3d12_device.Get(); }
#endif

 protected:
  void InitDeviceAllocator(OrtSession& session);
  void CreateSessionOptions();

  void CreateSessionOptionsFromConfig(const Config::SessionOptions& config_session_options,
                                      OrtSessionOptions& session_options,
                                      bool is_primary_session_options,
                                      bool disable_graph_capture);

#if USE_DML
  mutable DmlObjects dml_objects_;
  const OrtDmlApi* p_dml_api_{};
  std::unique_ptr<DmlPooledUploadHeap> dml_pooled_upload_heap_;
  std::unique_ptr<DmlExecutionContext> dml_execution_context_;
  std::unique_ptr<DmlReadbackHeap> dml_readback_heap_;
  ComPtr<IDMLDevice> dml_device_;
#endif

  std::unique_ptr<Ort::Allocator> owned_allocator_device_{};  // nullptr if n/a
  std::unique_ptr<OrtMemoryInfo> memory_info_device_{};       // nullptr if n/a

  std::shared_ptr<CapturedGraphPool> captured_graph_pool_;
  std::map<std::string, std::unique_ptr<OrtSessionOptions>> pipeline_session_options_;
};

}  // namespace Generators
