// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
//
// Modifications Copyright(C) 2026 Advanced Micro Devices, Inc. All rights reserved.
#pragma once
#include "model_type.h"
#include "ortx_tokenizer.h"
#include "../generators.h"
#include "utils.h"
#include "phi_image_processor.h"
#include "whisper_processor.h"
#include "phi_multimodal_processor.h"
#include "gemma_image_processor.h"
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
  virtual void Finalize(int current_length) {}

  virtual void RewindTo(size_t index) { (void)index; };
  virtual OrtValue* GetInput(const char* name);
  virtual OrtValue* GetOutput(const char* name);

  void ClearIO();  // Clear all inputs/outputs

  void SetActiveAdapter(Adapters* adapters, const std::string& adapter_name);
  void SetRunOption(const char* key, const char* value);
  void SetRunOptions(const Config::RunOptions& config_run_options);
  virtual void SetExtraInputs(const std::vector<ExtraInput>& extra_inputs) {}

  void DumpInputs();
  void DumpOutputs();

  const Model& model_;
  bool session_terminated_{};
  std::shared_ptr<const GeneratorParams> params_;

  std::vector<const char*> input_names_, output_names_;
  std::vector<std::string> adapter_names_;
  std::vector<OrtValue*> inputs_, outputs_;

  std::vector<std::pair<std::string, std::string>> ep_dynamic_options_next_run_;

 protected:
  void Run(OrtSession& session, bool graph_capture_this_run = false);
  bool first_run_{true};

  std::unique_ptr<OrtRunOptions> run_options_;

 private:
  std::string graph_id_{};
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

  void UpdateOptions(const char* const* keys, const char* const* values, size_t num_options);
  std::vector<int32_t> Encode(const char* text) const;
  std::string Decode(std::span<const int32_t> tokens) const;
  std::string ApplyChatTemplate(const char* template_str, const char* messages, const char* tools, bool add_generation_prompt) const;

  std::vector<int32_t> EncodeBatch(std::span<const std::string> strings) const;
  std::shared_ptr<Tensor> EncodeBatch(std::span<const char*> strings) const;
  std::vector<std::string> DecodeBatch(std::span<const int32_t> sequences, size_t count) const;

  int32_t TokenToTokenId(const char* token) const;
  int32_t GetBosTokenId() const { return bos_token_id_; }
  const std::vector<int32_t>& GetEosTokenIds() const { return eos_token_id_; }
  int32_t GetPadTokenId() const { return pad_token_id_; }

  OrtxPtr<OrtxTokenizer> tokenizer_;

 private:
  int32_t bos_token_id_;
  std::vector<int32_t> eos_token_id_;
  int32_t pad_token_id_;
};

struct MultiModalProcessor : std::enable_shared_from_this<MultiModalProcessor>, ExternalRefCounted<MultiModalProcessor> {
  MultiModalProcessor(Config& config, const SessionInfo& session_info);

  std::unique_ptr<NamedTensors> Process(const std::string& prompt, const Images* images, const Audios* audios) const;
  std::unique_ptr<NamedTensors> Process(std::span<const char*> prompts, const Images* images, const Audios* audios) const;

  std::shared_ptr<Tokenizer> tokenizer_;
  std::shared_ptr<Processor> processor_;

 private:
  std::unordered_map<std::string, std::function<std::shared_ptr<Processor>(Config&, const SessionInfo&)>> processor_factory_;
};

struct SessionInfo {
  SessionInfo() = default;

  void Add(OrtSession& session);

  bool HasInput(const std::string& name) const;
  bool HasOutput(const std::string& name) const;

  ONNXTensorElementDataType GetInputDataType(const std::string& name) const;
  ONNXTensorElementDataType GetOutputDataType(const std::string& name) const;

  std::vector<std::string> GetInputNames() const;

  std::vector<int64_t> GetInputShape(const std::string& name) const;
  std::vector<int64_t> GetOutputShape(const std::string& name) const;

  std::vector<const char*> GetInputSymbolicShape(const std::string& name) const;
  std::vector<const char*> GetOutputSymbolicShape(const std::string& name) const;

 private:
  std::unordered_map<std::string, std::unique_ptr<OrtTypeInfo>> inputs_, outputs_;
};

struct Model : std::enable_shared_from_this<Model>, LeakChecked<Model>, ExternalRefCounted<Model> {
  Model(std::unique_ptr<Config> config);
  virtual ~Model();

  std::shared_ptr<Tokenizer> CreateTokenizer() const;

  std::shared_ptr<MultiModalProcessor> CreateMultiModalProcessor() const;

  virtual std::unique_ptr<State> CreateState(DeviceSpan<int32_t> sequence_lengths, const GeneratorParams& params) const = 0;

  std::unique_ptr<OrtValue> ExpandInputs(std::unique_ptr<OrtValue>& input, int num_beams) const;

  OrtSessionOptions* GetSessionOptions(const std::string& model_id) const;

  /** \brief Gets the compiled model path for a pipeline model
   *
   * \param model_id The pipeline model ID
   * \return The compiled model path if available, empty string otherwise
   */
  std::string GetPipelineCompiledModelPath(const std::string& model_id) const;

  std::unique_ptr<OrtSession> CreateSession(OrtEnv& ort_env, const std::string& model_filename, OrtSessionOptions* session_options);

  bool IsPruned() const;

  /** \brief Returns the ORT execution provider name for the given device type if it supports EP context; empty string otherwise.
   * If EP Context is enabled for any provider, please add the provider name here.
   */
  static std::string EPContextSupportedProviders(DeviceType device_type) {
    switch (device_type) {
      case DeviceType::NvTensorRtRtx:
        return "NvTensorRTRTXExecutionProvider";
      default:
        return "";
    }
  }

  /** \brief Compiles the specified model and optionally all pipeline models
   *
   * Creates compilation options from session options and compiles the models.
   * Automatically configures compilation based on config settings:
   * - Input: Uses model data from buffer (if available via AddModelData), otherwise from file path
   * - Output: Creates "contexts" folder and saves as "{model_name}_{ep_name}_ctx.onnx", or as configured
   * - Reads compilation options from config.model.*.compile_options:
   *   * enable_ep_context - Controls whether model compilation is performed (default: not set, no compilation)
   *   * graph_optimization_level
   *   * ep_context_file_path - Full path (relative to config path) for compiled EP context model, e.g. "contexts/model_ctx.onnx"
   *   * ep_context_embed_mode - How EP context is stored (embedded vs external files)
   *   * flags
   *   * external_initializers_file_path and external_initializers_size_threshold
   *
   * Function pointers (write_func, get_initializer_location_func) must be set programmatically.
   *
   * Throws an exception on error.
   *
   * \param ort_env The OrtEnv object
   * \param model_filename The model filename to compile
   * \param session_options The session options to create compilation options from
   * \param is_primary_session_option If true, also compiles all pipeline models
   * \param compile_options The compile options from config for the specified model
   * \return The model path to use for creating session (original if not compiled, compiled path if compiled)
   */
  std::string CompileModel(OrtEnv& ort_env, const std::string& model_filename, OrtSessionOptions* session_options,
                           bool is_primary_session_option, const std::optional<Config::CompileOptions>& compile_options = std::nullopt);

 private:
  /** \brief Checks if a compiled model exists and is valid
   *
   * \param ort_env OrtEnv (used for EP device / compatibility validation)
   * \param model_filename The original model filename
   * \param compile_options_config The compile options from config (output path, force_compile_if_needed, etc.)
   * \param out_compiled_model_path Output parameter for the compiled model path (default or from config)
   * \return true if compiled model exists and is valid, false otherwise
   */
  bool CheckCompiledModelExists(OrtEnv& ort_env,
                                const std::string& model_filename,
                                const Config::CompileOptions& compile_options_config,
                                fs::path& out_compiled_model_path);

  /** \brief Validates a compiled model using EP compatibility APIs.
   * Context is valid only if: (1) compatibility info is present for this EP, and
   * (2) GetModelCompatibilityForEpDevices returns OPTIMAL or (PREFER_RECOMPILATION when force_compile_if_needed is false).
   * All other cases return false.
   *
   * \param ort_env OrtEnv (for GetEpDevices)
   * \param compiled_model_path Path to the compiled model file
   * \param force_compile_if_needed If true, PREFER_RECOMPILATION is treated as invalid (recompile); if false, it is accepted as valid with a warning
   * \return true if the compiled model is valid for the current EP (or validation not applicable)
   */
  bool ValidateCompiledModel(OrtEnv& ort_env, const fs::path& compiled_model_path, bool force_compile_if_needed);

 public:
  std::unique_ptr<Config> config_;
  std::unique_ptr<OrtSessionOptions> session_options_;
  std::unique_ptr<OrtArenaCfg> arena_cfg_;

  DeviceInterface* p_device_{};          // The device we're running on (matches device_type_) used for things that work the same on all devices
  DeviceInterface* p_device_inputs_{};   // For some model inputs, the device might be the CPU device (all but KV cache currently for WebGPU and DML)
  DeviceInterface* p_device_kvcache_{};  // The kvcache is always allocated in device memory  (TODO: Remove in favor of just p_device_?)

  Ort::Allocator& allocator_cpu_{GetDeviceInterface(DeviceType::CPU)->GetAllocator()};

  SessionInfo session_info_;

 protected:
  void CreateSessionOptions();
  std::unique_ptr<OrtModelCompilationOptions> CreateModelCompilationOptions(OrtEnv& ort_env, OrtSessionOptions* session_options);

  void CreateSessionOptionsFromConfig(const Config::SessionOptions& config_session_options,
                                      OrtSessionOptions& session_options,
                                      bool is_primary_session_options,
                                      bool disable_graph_capture);

  std::map<std::string, std::unique_ptr<OrtSessionOptions>> pipeline_session_options_;
  std::map<std::string, std::string> pipeline_compiled_model_paths_;  // Maps pipeline model_id to compiled model path
};

}  // namespace Generators
