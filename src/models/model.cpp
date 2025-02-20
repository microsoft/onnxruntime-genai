// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
//
// Modifications Copyright(C) 2024 Advanced Micro Devices, Inc. All rights reserved
#include <algorithm>
#include <set>
#include <string>
#include <thread>

#include "../generators.h"
#include "../search.h"
#include "model.h"
#include "gpt.h"
#include "decoder_only.h"
#include "whisper.h"
#include "multi_modal_vision_model.h"
#include "decoder_only_pipeline.h"
#include "../dml/interface.h"

namespace Generators {

State::State(const GeneratorParams& params, const Model& model)
    : model_{model},
      params_{params.shared_from_this()},
      run_options_{OrtRunOptions::Create()},
      extra_outputs_{*this} {}

void State::Run(OrtSession& session, int new_batch_size) {
  auto captured_graph_info = GetCapturedGraphInfo();

  if (first_run_) {
    if (captured_graph_info) {
      run_options_->AddConfigEntry("gpu_graph_id", "-1");
    }
    extra_outputs_.Add(session.GetOutputNames());
    first_run_ = false;
  } else {
    if (captured_graph_info && new_batch_size != current_batch_size_) {
      current_batch_size_ = new_batch_size;
      auto annotation_id = std::to_string(captured_graph_info->GenerateUniqueAnnotationID(new_batch_size));
      run_options_->AddConfigEntry("gpu_graph_id", annotation_id.c_str());
    }
    extra_outputs_.Update();
  }

  if (g_log.enabled && g_log.model_input_values) {
    auto& stream = Log("model_input_values");
    stream << std::endl;
    DumpTensors(model_, stream, inputs_.data(), input_names_.data(), input_names_.size(), true);
  }

  if (g_log.enabled && g_log.model_output_shapes) {
    auto& stream = Log("model_output_shapes");
    stream << std::endl;
    DumpTensors(model_, stream, outputs_.data(), output_names_.data(), output_names_.size(), false);
  }

  session.Run(run_options_.get(), input_names_.data(), inputs_.data(), input_names_.size(),
              output_names_.data(), outputs_.data(), output_names_.size());

  extra_outputs_.RegisterOutputs();

  if (g_log.enabled && g_log.model_output_values) {
    auto& stream = Log("model_output_values");
    stream << std::endl;
    DumpTensors(model_, stream, outputs_.data(), output_names_.data(), output_names_.size(), true);
  }
}

void State::SetTerminate() {
  session_terminated_ = true;
  run_options_->SetTerminate();
}

void State::UnsetTerminate() {
  session_terminated_ = false;
  run_options_->UnsetTerminate();
}

OrtValue* State::GetInput(const char* name) {
  ThrowErrorIfSessionTerminated(session_terminated_);
  for (size_t i = 0; i < input_names_.size(); i++) {
    if (std::strcmp(input_names_[i], name) == 0) {
      return inputs_[i];
    }
  }
  return nullptr;
}

OrtValue* State::GetOutput(const char* name) {
  ThrowErrorIfSessionTerminated(session_terminated_);
  for (size_t i = 0; i < output_names_.size(); i++) {
    if (std::strcmp(output_names_[i], name) == 0) {
      return outputs_[i];
    }
  }
  return nullptr;
}

void State::ClearIO() {
  input_names_.clear();
  output_names_.clear();
  inputs_.clear();
  outputs_.clear();
}

void State::SetActiveAdapter(Adapters* adapters, const std::string& adapter_name) {
  if (!adapters_) {
    adapters_ = adapters->shared_from_this();
  } else if (adapters_.get() != adapters) {
    // Two different instances of Adapters are being used. The Generator state can only manage
    // active adapters from a single Adapters container.
    throw std::runtime_error("Generator state can only register a single Adapters container.");
  }

  run_options_->AddActiveLoraAdapter(*adapters_->AcquireAdapter(adapter_name));
  adapter_names_.push_back(adapter_name);
}

State::~State() {
  if (adapters_) {
    for (const auto& adapter_name : adapter_names_) {
      adapters_->ReleaseAdapter(adapter_name);
    }
  }
}

std::vector<int32_t> PadInputs(std::span<std::span<const int32_t>> sequences, int32_t pad_token_id) {
  bool pad_right_{true};

  size_t max_length = 0;
  for (auto& sequence : sequences)
    max_length = std::max(max_length, sequence.size());

  std::vector<int32_t> result(max_length * sequences.size());
  std::span<int32_t> result_span(result);

  // Copy and pad the sequences with pad_token_id
  for (size_t i = 0; i < sequences.size(); i++) {
    auto output_span = result_span.subspan(i * max_length, max_length);
    auto input_span = sequences[i];

    auto pad_count = max_length - input_span.size();
    if (pad_right_) {
      std::copy(input_span.begin(), input_span.end(), output_span.begin());
      std::fill(output_span.end() - pad_count, output_span.end(), pad_token_id);
    } else {
      std::fill(output_span.begin(), output_span.begin() + pad_count, pad_token_id);
      std::copy(input_span.begin(), input_span.end(), output_span.begin() + pad_count);
    }
  }

  return result;
}

void CheckResult(extError_t error) {
  if (error != kOrtxOK)
    throw std::runtime_error(OrtxGetLastErrorMessage());
}

TokenizerStream::TokenizerStream(const Tokenizer& tokenizer)
    : tokenizer_{tokenizer.shared_from_this()} {
  CheckResult(OrtxCreate(kOrtxKindDetokenizerCache, cache_.Address()));
}

const std::string& TokenizerStream::Decode(int32_t token) {
  const char* string;
  CheckResult(OrtxDetokenizeCached(tokenizer_->tokenizer_, cache_, token, &string));
  chunk_ = string;
  return chunk_;
}

Tokenizer::Tokenizer(Config& config) : pad_token_id_{config.model.pad_token_id} {
  CheckResult(OrtxCreateTokenizer(tokenizer_.Address(), config.config_path.string().c_str()));
}

std::unique_ptr<TokenizerStream> Tokenizer::CreateStream() const {
  return std::make_unique<TokenizerStream>(*this);
}

std::vector<int32_t> Tokenizer::Encode(const char* text) const {
  OrtxPtr<OrtxTokenId2DArray> ids;
  CheckResult(OrtxTokenize(tokenizer_, &text, 1, ids.Address()));

  const extTokenId_t* tokens;
  size_t count;
  CheckResult(OrtxTokenId2DArrayGetItem(ids, 0, &tokens, &count));
  return {tokens, tokens + count};
}

std::string Tokenizer::Decode(std::span<const int32_t> tokens) const {
  OrtxPtr<OrtxStringArray> ortx_string_array;
  CheckResult(OrtxDetokenize1D(tokenizer_, reinterpret_cast<const uint32_t*>(tokens.data()), tokens.size(), ortx_string_array.Address()));

  const char* string;
  CheckResult(OrtxStringArrayGetItem(ortx_string_array, 0, &string));
  return string;
}

std::vector<int32_t> Tokenizer::EncodeBatch(std::span<const std::string> strings) const {
  std::vector<std::vector<int32_t>> sequences;
  std::vector<std::span<const int32_t>> span_sequences;
  for (size_t i = 0; i < strings.size(); i++) {
    sequences.emplace_back(Encode(strings[i].c_str()));
    span_sequences.emplace_back(sequences.back());
  }

  return PadInputs(span_sequences, pad_token_id_);
}

std::vector<std::string> Tokenizer::DecodeBatch(std::span<const int32_t> sequences, size_t count) const {
  if (sequences.size() % count != 0)
    throw std::runtime_error("DecodeBatch: sequences must be evenly divisible by the count");
  size_t sequence_length = sequences.size() / count;
  std::vector<std::string> strings;
  for (size_t i = 0; i < count; i++)
    strings.emplace_back(Decode(sequences.subspan(sequence_length * i, sequence_length)));
  return strings;
}

int32_t Tokenizer::TokenToTokenId(const char* token) const {
  extTokenId_t token_id;
  CheckResult(OrtxConvertTokenToId(tokenizer_, token, &token_id));
  return token_id;
}

// Since Python/Others can and will hold onto a generator object past the model object's lifetime we need to ensure
// the allocator used is not destroyed until last. This keeps the allocator around until exit, after all other memory
// has been destroyed. Without this, we will crash in the Onnxruntime BFCArena code when deleting tensors due to the
// arena already being destroyed.
void EnsureDeviceOrtInit(OrtSession& session, DeviceType type) {
  // CPU Allocator is a special case, it's not in the owned 'allocator_device_' table below so we handle it separately
  if (type == DeviceType::CPU)
    return;

  auto& device = GetOrtGlobals()->allocator_device_[static_cast<int>(type)];
  if (device)
    return;

  static const char* device_type_names[] = {"CPU (Not used, see above)", "Cuda", "DML", "WebGPU_Buffer", "QnnHtpShared"};
  static_assert(std::size(device_type_names) == static_cast<size_t>(DeviceType::MAX));

  auto name = device_type_names[static_cast<int>(type)];
  auto memory_info = OrtMemoryInfo::Create(name, OrtAllocatorType::OrtDeviceAllocator, 0, OrtMemType::OrtMemTypeDefault);
  device = Ort::Allocator::Create(session, *memory_info);
  if (!device)
    throw std::runtime_error("Unexpected failure to create device memory allocator for " + std::string(name));
  GetDeviceInterface(type)->InitOrt(*Ort::api, *device);  // Necessary for any shared library providers so they can access Ort::api
}

SessionInfo::SessionInfo(OrtSession& session) {
  Add(session);
}

void SessionInfo::Add(OrtSession& session) {
  auto input_names = session.GetInputNames();
  std::vector<ONNXTensorElementDataType> input_types(input_names.size());
  for (size_t i = 0; i < input_types.size(); i++) {
    auto input_type = session.GetInputTypeInfo(i)->GetTensorTypeAndShapeInfo().GetElementType();
    auto found_input = inputs_.find(input_names[i]);
    if (found_input != inputs_.end() && found_input->second != input_type)
      throw std::runtime_error("Model input type mismatch: " + input_names[i] + " expected " + std::to_string(found_input->second) + " got " + std::to_string(input_type));
    inputs_.emplace(std::make_pair(std::move(input_names[i]), input_type));
  }

  auto output_names = session.GetOutputNames();
  std::vector<ONNXTensorElementDataType> output_types(output_names.size());
  for (size_t i = 0; i < output_types.size(); i++) {
    auto output_type = session.GetOutputTypeInfo(i)->GetTensorTypeAndShapeInfo().GetElementType();
    outputs_.emplace(std::make_pair(std::move(output_names[i]), output_type));
  }
}

bool SessionInfo::HasInput(const std::string& name) const {
  return inputs_.find(name) != inputs_.end();
}

bool SessionInfo::HasOutput(const std::string& name) const {
  return outputs_.find(name) != outputs_.end();
}

ONNXTensorElementDataType SessionInfo::GetInputDataType(const std::string& name) const {
  auto result = inputs_.find(name);
  if (result == inputs_.end())
    throw std::runtime_error("Model input was not found: " + name);
  return result->second;
}

ONNXTensorElementDataType SessionInfo::GetOutputDataType(const std::string& name) const {
  auto result = outputs_.find(name);
  if (result == outputs_.end())
    throw std::runtime_error("Model output was not found: " + name);
  return result->second;
}

std::vector<std::string> SessionInfo::GetInputNames() const {
  std::vector<std::string> names;
  names.reserve(inputs_.size());
  for (const auto& input : inputs_)
    names.push_back(input.first);
  return names;
}

Model::Model(std::unique_ptr<Config> config) : config_{std::move(config)} {
  CreateSessionOptions();
}

Model::~Model() = default;

void Model::InitDeviceAllocator(OrtSession& session) {
  EnsureDeviceOrtInit(session, p_device_->GetType());

  // Only CUDA does every input on the device
  if (p_device_->GetType() == DeviceType::CUDA)
    p_device_inputs_ = p_device_;
  else
    p_device_inputs_ = GetDeviceInterface(DeviceType::CPU);

  // The kvcache is always allocated in device memory
  p_device_kvcache_ = p_device_;

  session_info_ = std::make_unique<SessionInfo>(session);
  captured_graph_pool_ = std::make_shared<CapturedGraphPool>(config_.get(), session_info_.get(), &p_device_->GetAllocator());
}

void Model::CreateSessionOptionsFromConfig(const Config::SessionOptions& config_session_options,
                                           OrtSessionOptions& session_options,
                                           bool is_primary_session_options,
                                           bool disable_graph_capture) {
  // Default to a limit of 16 threads to optimize performance
  constexpr int min_thread_nums = 1;
  constexpr int max_thread_nums = 16;
  int num_of_cores = std::max(min_thread_nums, static_cast<int>(std::thread::hardware_concurrency() / 2));
  session_options.SetIntraOpNumThreads(std::min(num_of_cores, max_thread_nums));

  if (config_session_options.intra_op_num_threads.has_value()) {
    session_options.SetIntraOpNumThreads(config_session_options.intra_op_num_threads.value());
  }

  if (config_session_options.inter_op_num_threads.has_value()) {
    session_options.SetInterOpNumThreads(config_session_options.inter_op_num_threads.value());
  }

  if (config_session_options.enable_cpu_mem_arena.has_value()) {
    if (config_session_options.enable_cpu_mem_arena.value())
      session_options.EnableCpuMemArena();
    else
      session_options.DisableCpuMemArena();
  }

  if (config_session_options.enable_mem_pattern.has_value()) {
    if (config_session_options.enable_mem_pattern.value())
      session_options.EnableMemPattern();
    else
      session_options.DisableMemPattern();
  }

  if (config_session_options.log_id.has_value()) {
    session_options.SetLogId(config_session_options.log_id.value().c_str());
  }

  if (config_session_options.log_severity_level.has_value()) {
    session_options.SetLogSeverityLevel(config_session_options.log_severity_level.value());
  }

  if (config_session_options.enable_profiling.has_value()) {
    fs::path profile_file_prefix{config_session_options.enable_profiling.value()};
    session_options.EnableProfiling(profile_file_prefix.c_str());
  }

  if (config_session_options.disable_cpu_ep_fallback.has_value()) {
    if (config_session_options.disable_cpu_ep_fallback.value())
      session_options.DisableCpuEpFallback();
    else
      session_options.EnableCpuEpFallback();
  }

  if (config_session_options.disable_quant_qdq.has_value()) {
    if (config_session_options.disable_quant_qdq.value())
      session_options.DisableQuantQdq();
    else
      session_options.EnableQuantQdq();
  }

  if (config_session_options.enable_quant_qdq_cleanup.has_value()) {
    if (config_session_options.enable_quant_qdq_cleanup.value())
      session_options.EnableQuantQdqCleanup();
    else
      session_options.DisableQuantQdqCleanup();
  }

  if (config_session_options.ep_context_enable.has_value()) {
    if (config_session_options.ep_context_enable.value())
      session_options.SetEpContextEnable();
  }

  if (config_session_options.ep_context_embed_mode.has_value()) {
    session_options.SetEpContextEmbedMode(config_session_options.ep_context_embed_mode.value().c_str());
  }

  if (config_session_options.ep_context_file_path.has_value()) {
    session_options.SetEpContextFilePath(config_session_options.ep_context_file_path.value().c_str());
  }

  if (config_session_options.provider_options.empty() && config_session_options.use_env_allocators) {
    // Share env allocators across sessions that only use the CPU provider
    session_options.AddConfigEntry("session.use_env_allocators", "1");
  }

  if (config_session_options.graph_optimization_level.has_value()) {
    session_options.SetGraphOptimizationLevel(config_session_options.graph_optimization_level.value());
  }

  for (auto& provider_options : config_session_options.provider_options) {
    if (provider_options.name == "cuda") {
      auto ort_provider_options = OrtCUDAProviderOptionsV2::Create();
      std::vector<const char*> keys, values;
      for (auto& option : provider_options.options) {
        keys.emplace_back(option.first.c_str());
        values.emplace_back(option.second.c_str());
      }
      ort_provider_options->Update(keys.data(), values.data(), keys.size());

      // Device type determines the scoring device.
      // Only use the primary session options to determine the device type
      if (is_primary_session_options) {
        p_device_ = GetDeviceInterface(DeviceType::CUDA);

        // Create and set our cudaStream_t
        ort_provider_options->UpdateValue("user_compute_stream", p_device_->GetCudaStream());
      }

      session_options.AppendExecutionProvider_CUDA_V2(*ort_provider_options);

    } else if (provider_options.name == "rocm") {
      OrtROCMProviderOptions ort_provider_options;

      std::vector<const char*> keys, values;
      for (auto& option : provider_options.options) {
        keys.emplace_back(option.first.c_str());
        values.emplace_back(option.second.c_str());
      }

      Ort::ThrowOnError(Ort::api->UpdateROCMProviderOptions(&ort_provider_options, keys.data(), values.data(), keys.size()));
      session_options.AppendExecutionProvider_ROCM(ort_provider_options);
#if USE_DML
    } else if (provider_options.name == "dml") {
      if (!GetDmlInterface()) {
        LUID device_luid{};
        LUID* p_device_luid{};
        for (const auto& [name, value] : provider_options.options) {
          if (name == "luid") {
            if (auto separator_position = value.find(":"); separator_position != std::string::npos) {
              device_luid.HighPart = std::stol(value.substr(0, separator_position));
              device_luid.LowPart = std::stol(value.substr(separator_position + 1));
              p_device_luid = &device_luid;
            }
          }
        }

        InitDmlInterface(p_device_luid);
      }

      SetDmlProvider(session_options);

      if (is_primary_session_options)
        p_device_ = GetDeviceInterface(DeviceType::DML);  // We use a DML allocator for input/output caches, but other tensors will use CPU tensors
#endif
    } else if (provider_options.name == "qnn") {
      session_options.AddConfigEntry("ep.share_ep_contexts", "1");
      std::unordered_map<std::string, std::string> opts;
      for (auto& option : provider_options.options) {
        opts.emplace(option.first, option.second);
      }

      // TODO set device_type_ in a less hacky way.
      // now, all QNN EP enable_htp_shared_memory_allocator option values had better be consistent...
      // on the other hand, not sure if is_primary_session_options is the right thing to check here.
      if (const auto opt_it = opts.find("enable_htp_shared_memory_allocator");
          opt_it != opts.end() && opt_it->second == "1") {
        p_device_ = GetDeviceInterface(DeviceType::QNN);
      }

      session_options.AppendExecutionProvider("QNN", opts);
    } else if (provider_options.name == "webgpu") {
      p_device_ = GetDeviceInterface(DeviceType::WEBGPU);
      std::unordered_map<std::string, std::string> opts;
      for (auto& option : provider_options.options) {
        opts.emplace(option.first, option.second);
      }
      session_options.AppendExecutionProvider("WebGPU", opts);
    } else
      throw std::runtime_error("Unknown provider type: " + provider_options.name);
  }

  // Fallback to CPU if no provider specific interface was set
  if (!p_device_)
    p_device_ = GetDeviceInterface(DeviceType::CPU);
}

void Model::CreateSessionOptions() {
  session_options_ = OrtSessionOptions::Create();

  CreateSessionOptionsFromConfig(config_->model.decoder.session_options, *session_options_, true, false);

  for (auto& pipeline_model : config_->model.decoder.pipeline) {
    if (pipeline_model.session_options.has_value()) {
      auto emplaced = pipeline_session_options_.emplace(pipeline_model.model_id, OrtSessionOptions::Create());
      CreateSessionOptionsFromConfig(*pipeline_model.session_options, *emplaced.first->second, false, false);
    }
  }
}

OrtSessionOptions* Model::GetSessionOptions(const std::string& model_id) const {
  auto session_options = pipeline_session_options_.find(model_id);
  // Use the pipeline model session options id config defined it.
  if (session_options != pipeline_session_options_.end())
    return session_options->second.get();

  // Else fallback to the main session options.
  return session_options_.get();
}

std::shared_ptr<Tokenizer> Model::CreateTokenizer() const {
  return std::make_shared<Tokenizer>(*config_);
}

std::shared_ptr<MultiModalProcessor> Model::CreateMultiModalProcessor() const {
  return std::make_shared<MultiModalProcessor>(*config_, *session_info_);
}

std::shared_ptr<Model> CreateModel(OrtEnv& ort_env, const char* config_path, const RuntimeSettings* settings /*= nullptr*/) {
  std::string config_overlay;
  if (settings) {
    config_overlay = settings->GenerateConfigOverlay();
  }
  auto config = std::make_unique<Config>(fs::path(config_path), config_overlay);
  return CreateModel(ort_env, std::move(config));
}

std::shared_ptr<Model> CreateModel(OrtEnv& ort_env, std::unique_ptr<Config> config) {
  std::set<std::string> llm_types = {"chatglm", "decoder", "gemma", "gemma2", "granite", "llama", "mistral", "nemotron", "olmo", "phi", "phimoe", "phi3", "phi3small", "qwen2"};
  if (config->model.type == "gpt2")
    return std::make_shared<Gpt_Model>(std::move(config), ort_env);
  if (llm_types.find(config->model.type) != llm_types.end())
    return std::make_shared<DecoderOnly_Model>(std::move(config), ort_env);
  if (config->model.type == "whisper")
    return std::make_shared<Whisper_Model>(std::move(config), ort_env);
  if (config->model.type == "phi3v")
    return std::make_shared<MultiModalVisionModel>(std::move(config), ort_env);
  if (config->model.type == "decoder-pipeline")
    return std::make_shared<DecoderOnlyPipelineModel>(std::move(config), ort_env);

  throw std::runtime_error("Unsupported model_type in config.json: " + config->model.type);
}

std::shared_ptr<GeneratorParams> CreateGeneratorParams(const Model& model) {
  return std::make_shared<GeneratorParams>(model);
}

// Used by benchmarking tests only, should not be used normally
std::shared_ptr<GeneratorParams> CreateGeneratorParams(const Config& config) {
  return std::make_shared<GeneratorParams>(config);
}

void Cast(OrtValue& input, std::unique_ptr<OrtValue>& output, DeviceInterface& device, ONNXTensorElementDataType output_type) {
  auto input_info = input.GetTensorTypeAndShapeInfo();
  auto shape = input_info->GetShape();

  if (output && shape != output->GetTensorTypeAndShapeInfo()->GetShape())
    output = nullptr;
  if (!output)
    output = OrtValue::CreateTensor(device.GetAllocator(), shape, output_type);

  if (!device.Cast(input, *output))
    GetDeviceInterface(DeviceType::CPU)->Cast(input, *output);
}

std::unique_ptr<OrtValue> Model::ExpandInputs(std::unique_ptr<OrtValue>& input, int num_beams) const {
  // Input shape (batch_size, sequence_length). The input is required with data type T.
  // Output shape (batch_size * num_beams, sequence_length)

  // When num_beams == 1, we don't need to expand the input, but the expand has a side effect of copying from
  // CPU memory to device memory, so we can skip if the p_device_inputs_ is the CPU device
  if (num_beams == 1 && p_device_inputs_ == GetDeviceInterface(DeviceType::CPU))
    return std::move(input);

  auto input_type_info = input->GetTensorTypeAndShapeInfo();
  auto element_type = input_type_info->GetElementType();
  auto input_shape = input_type_info->GetShape();
  const int64_t batch_size = input_shape[0];
  const int64_t data_size_bytes = input_type_info->GetElementCount() * SizeOf(element_type) / batch_size;

  input_shape[0] *= num_beams;

  auto input_span = ByteWrapTensor(*GetDeviceInterface(DeviceType::CPU), *input);
  auto expanded = OrtValue::CreateTensor(p_device_inputs_->GetAllocator(), input_shape, element_type);
  auto expanded_span = ByteWrapTensor(*p_device_inputs_, *expanded);

  for (int i = 0; i < batch_size; i++) {
    for (int j = 0; j < num_beams; j++) {
      expanded_span.subspan((i * num_beams + j) * data_size_bytes, data_size_bytes).CopyFrom(input_span.subspan(i * data_size_bytes, data_size_bytes));
    }
  }
  return expanded;
}

MultiModalProcessor::MultiModalProcessor(Config& config, const SessionInfo& session_info)
    : tokenizer_{std::make_shared<Tokenizer>(config)} {
  if (config.model.type == "phi3v") {
    image_processor_ = std::make_shared<ImageProcessor>(config, session_info);
  } else if (config.model.type == "whisper") {
    audio_processor_ = std::make_shared<AudioProcessor>(config, session_info);
  } else {
    throw std::runtime_error("MultiModalProcessor cannot be created. Expected a multimodal model. Actual: " + config.model.type);
  }
}

}  // namespace Generators
