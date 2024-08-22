// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#include <algorithm>
#include <thread>

#include "../generators.h"
#include "../search.h"
#include "model.h"
#include "gpt.h"
#include "decoder_only.h"
#include "whisper.h"
#include "kernels.h"
#include "multi_modal_vision_model.h"
#if USE_DML
#include <wil/wrl.h>
#include "dml_provider_factory.h"
#include "../dml/dml_helpers.h"

EXTERN_C IMAGE_DOS_HEADER __ImageBase;

static std::string CurrentModulePath() {
  char path[MAX_PATH];
  GetModuleFileNameA((HINSTANCE)&__ImageBase, path, _countof(path));

  char absolute_path[MAX_PATH];
  char* name;
  GetFullPathNameA(path, _countof(path), absolute_path, &name);

  auto idx = std::distance(absolute_path, name);
  auto out_path = std::string(absolute_path);
  out_path.resize(idx);

  return out_path;
}
#endif

namespace Generators {

State::State(const GeneratorParams& params, const Model& model)
    : params_{params.shared_from_this()},
      model_{model} {}

void State::Run(OrtSession& session, OrtRunOptions& run_options, int new_batch_size) {
  if (first_run_) {
    if (params_->use_cuda_graph) {
      model_.run_options_->AddConfigEntry("gpu_graph_id", "-1");
    }
    first_run_ = false;
  } else if (params_->use_cuda_graph && new_batch_size != current_batch_size_) {
    assert(GetCapturedGraphInfo() != nullptr);
    current_batch_size_ = new_batch_size;
    auto annotation_id = std::to_string(GetCapturedGraphInfo()->GenerateUniqueAnnotationID(new_batch_size));
    model_.run_options_->AddConfigEntry("gpu_graph_id", annotation_id.c_str());
  }

  if (g_log.enabled && g_log.model_input_values) {
    auto& stream = Log("model_input_values");
    stream << std::endl;
    DumpTensors(stream, inputs_.data(), input_names_.data(), input_names_.size(), true);
  }

  if (g_log.enabled && g_log.model_output_shapes) {
    auto& stream = Log("model_output_shapes");
    stream << std::endl;
    DumpTensors(stream, outputs_.data(), output_names_.data(), output_names_.size(), false);
  }

  session.Run(&run_options, input_names_.data(), inputs_.data(), input_names_.size(), output_names_.data(), outputs_.data(), output_names_.size());

  if (g_log.enabled && g_log.model_output_values) {
    auto& stream = Log("model_output_values");
    stream << std::endl;
    DumpTensors(stream, outputs_.data(), output_names_.data(), output_names_.size(), true);
  }
}

OrtValue* State::GetOutput(const char* name) {
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

#if USE_CUDA
// Since Python/Others can and will hold onto a generator object past the model object's lifetime we need to ensure
// the allocator used is not destroyed until last. This keeps the allocator around until exit, after all other memory
// has been destroyed. Without this, we will crash in the Onnxruntime BFCArena code when deleting tensors due to the
// arena already being destroyed.
Ort::Allocator* GetCudaAllocator(OrtSession& session) {
  auto& globals = *GetOrtGlobals();
  if (!globals.allocator_cuda_) {
    globals.memory_info_cuda_ = OrtMemoryInfo::Create("Cuda", OrtAllocatorType::OrtDeviceAllocator, 0, OrtMemType::OrtMemTypeDefault);
    globals.allocator_cuda_ = Ort::Allocator::Create(session, *globals.memory_info_cuda_);
  }
  return globals.allocator_cuda_.get();
}
#endif

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

Model::Model(std::unique_ptr<Config> config) : config_{std::move(config)} {
  // TODO: add function to create run options
  run_options_ = OrtRunOptions::Create();

  CreateSessionOptions();
}

Model::~Model() = default;

void Model::InitDeviceAllocator([[maybe_unused]] OrtSession& session) {
  allocator_device_ = &allocator_cpu_;
#if USE_CUDA
  if (device_type_ == DeviceType::CUDA) {
    allocator_device_ = GetCudaAllocator(session);
  }
#elif USE_DML
  if (device_type_ == DeviceType::DML) {
    memory_info_device_ = OrtMemoryInfo::Create("DML", OrtAllocatorType::OrtDeviceAllocator, 0, OrtMemType::OrtMemTypeDefault);
    dml_owned_allocator_ = Ort::Allocator::Create(session, *memory_info_device_);
    allocator_device_ = dml_owned_allocator_.get();
  }
#endif

  session_info_ = std::make_unique<SessionInfo>(session);
  captured_graph_pool_ = std::make_shared<CapturedGraphPool>(config_.get(), session_info_.get(), allocator_device_);
}

void Model::CreateSessionOptions() {
  session_options_ = OrtSessionOptions::Create();
  auto& ort_options = *session_options_;
  auto& options = config_->model.decoder.session_options;

  // Default to a limit of 16 threads to optimize performance
  constexpr int min_thread_nums = 1;
  constexpr int max_thread_nums = 16;
  int num_of_cores = std::max(min_thread_nums, static_cast<int>(std::thread::hardware_concurrency() / 2));
  ort_options.SetIntraOpNumThreads(std::min(num_of_cores, max_thread_nums));

  if (options.intra_op_num_threads.has_value()) {
    ort_options.SetIntraOpNumThreads(options.intra_op_num_threads.value());
  }

  if (options.inter_op_num_threads.has_value()) {
    ort_options.SetInterOpNumThreads(options.inter_op_num_threads.value());
  }

  if (options.enable_cpu_mem_arena.has_value()) {
    if (options.enable_cpu_mem_arena.value())
      ort_options.EnableCpuMemArena();
    else
      ort_options.DisableCpuMemArena();
  }

  if (options.enable_mem_pattern.has_value()) {
    if (options.enable_cpu_mem_arena.value())
      ort_options.EnableMemPattern();
    else
      ort_options.DisableMemPattern();
  }

  if (options.log_id.has_value()) {
    ort_options.SetLogId(options.log_id.value().c_str());
  }

  if (options.log_severity_level.has_value()) {
    ort_options.SetLogSeverityLevel(options.log_severity_level.value());
  }

  if (options.enable_profiling.has_value()) {
    fs::path profile_file_prefix{options.enable_profiling.value()};
    ort_options.EnableProfiling(profile_file_prefix.c_str());
  }

  if (options.disable_cpu_ep_fallback.has_value()) {
    if (options.disable_cpu_ep_fallback.value())
      ort_options.DisableCpuEpFallback();
    else
      ort_options.EnableCpuEpFallback();
  }

  if (options.disable_quant_qdq.has_value()) {
    if (options.disable_quant_qdq.value())
      ort_options.DisableQuantQdq();
    else
      ort_options.EnableQuantQdq();
  }

  if (options.enable_quant_qdq_cleanup.has_value()) {
    if (options.enable_quant_qdq_cleanup.value())
      ort_options.EnableQuantQdqCleanup();
    else
      ort_options.DisableQuantQdqCleanup();
  }

  if (options.ep_context_enable.has_value()) {
    if (options.ep_context_enable.value())
      ort_options.SetEpContextEnable();
  }

  if (options.ep_context_embed_mode.has_value()) {
    ort_options.SetEpContextEmbedMode(options.ep_context_embed_mode.value().c_str());
  }

  if (options.ep_context_file_path.has_value()) {
    ort_options.SetEpContextFilePath(options.ep_context_file_path.value().c_str());
  }

  for (auto& provider_options : options.provider_options) {
    if (provider_options.name == "cuda") {
      auto ort_provider_options = OrtCUDAProviderOptionsV2::Create();
      std::vector<const char*> keys, values;
      for (auto& option : provider_options.options) {
        keys.emplace_back(option.first.c_str());
        values.emplace_back(option.second.c_str());
      }
      ort_provider_options->Update(keys.data(), values.data(), keys.size());

      // Create and set our cudaStream_t
      cuda_stream_.Create();
      ort_provider_options->UpdateValue("user_compute_stream", cuda_stream_.get());

      ort_options.AppendExecutionProvider_CUDA_V2(*ort_provider_options);
      device_type_ = DeviceType::CUDA;  // Scoring will use CUDA
    } else if (provider_options.name == "rocm") {
      OrtROCMProviderOptions ort_provider_options;

      std::vector<const char*> keys, values;
      for (auto& option : provider_options.options) {
        keys.emplace_back(option.first.c_str());
        values.emplace_back(option.second.c_str());
      }

      Ort::ThrowOnError(Ort::api->UpdateROCMProviderOptions(&ort_provider_options, keys.data(), values.data(), keys.size()));
      ort_options.AppendExecutionProvider_ROCM(ort_provider_options);
#if USE_DML
    } else if (provider_options.name == "dml") {
      auto current_module_path = CurrentModulePath();
      dml_objects_ = DmlHelpers::CreateDmlObjects(current_module_path);

      constexpr auto directml_dll = "DirectML.dll";
      wil::unique_hmodule smart_directml_dll(LoadLibraryEx(directml_dll, nullptr, 0));
      THROW_LAST_ERROR_IF(!smart_directml_dll);

      if (LoadLibraryEx(directml_dll, nullptr, 0) == NULL) {
        throw std::runtime_error("DirectML.dll not found");
      }

      auto dml_create_device1_fn = reinterpret_cast<decltype(&DMLCreateDevice1)>(GetProcAddress(smart_directml_dll.get(), "DMLCreateDevice1"));
      THROW_LAST_ERROR_IF(!dml_create_device1_fn);
      THROW_IF_FAILED(dml_create_device1_fn(dml_objects_.d3d12_device.Get(), DML_CREATE_DEVICE_FLAG_NONE, DML_FEATURE_LEVEL_5_0, IID_PPV_ARGS(&dml_device_)));

      Ort::ThrowOnError(Ort::api->GetExecutionProviderApi("DML", ORT_API_VERSION, reinterpret_cast<const void**>(&p_dml_api_)));
      if (!p_dml_api_) {
        throw std::runtime_error("Unexpected nullptr getting OrtDmlApi");
      }

      dml_execution_context_ = std::make_unique<DmlExecutionContext>(
          dml_objects_.d3d12_device.Get(),
          dml_device_.Get(),
          dml_objects_.command_queue.Get(),
          *allocator_device_,
          p_dml_api_);

      dml_pooled_upload_heap_ = std::make_unique<DmlPooledUploadHeap>(dml_objects_.d3d12_device.Get(), dml_execution_context_.get());
      dml_readback_heap_ = std::make_unique<DmlReadbackHeap>(dml_objects_.d3d12_device.Get(), dml_execution_context_.get());

      // The vision model doesn't support graph capture because of dynamic shapes, so don't enable graph capture for it
      if (!config_->model.vision.filename.empty()) {
        vision_session_options_ = ort_options.Clone();
        p_dml_api_->SessionOptionsAppendExecutionProvider_DML1(vision_session_options_.get(), dml_device_.Get(), dml_objects_.command_queue.Get());
      }

      ort_options.AddConfigEntry("ep.dml.enable_graph_capture", "1");
      ort_options.AddConfigEntry("ep.dml.disable_memory_arena", "1");
      p_dml_api_->SessionOptionsAppendExecutionProvider_DML1(&ort_options, dml_device_.Get(), dml_objects_.command_queue.Get());
      is_intel_device_ = DmlHelpers::IsIntelDevice(dml_objects_.d3d12_device.Get());

      device_type_ = DeviceType::DML;  // We use a DML allocator for input/output caches, but other tensors will use CPU tensors
#endif
    } else if (provider_options.name == "qnn") {
      std::unordered_map<std::string, std::string> opts;
      for (auto& option : provider_options.options) {
        opts.emplace(option.first, option.second);
      }

      ort_options.AppendExecutionProvider("QNN", opts);
    } else
      throw std::runtime_error("Unknown provider type: " + provider_options.name);
  }
}

std::shared_ptr<Tokenizer> Model::CreateTokenizer() const {
  return std::make_shared<Tokenizer>(*config_);
}

std::shared_ptr<MultiModalProcessor> Model::CreateMultiModalProcessor() const {
  return std::make_shared<MultiModalProcessor>(*config_, *session_info_);
}

std::shared_ptr<Model> CreateModel(OrtEnv& ort_env, const char* config_path) {
  auto config = std::make_unique<Config>(fs::path(config_path));

  if (config->model.type == "gpt2")
    return std::make_shared<Gpt_Model>(std::move(config), ort_env);
  if (config->model.type == "llama" || config->model.type == "gemma" || config->model.type == "gemma2" || config->model.type == "mistral" || config->model.type == "phi" || config->model.type == "phi3" || config->model.type == "phi3small" || config->model.type == "phimoe" || config->model.type == "qwen2")
    return std::make_shared<DecoderOnly_Model>(std::move(config), ort_env);
  if (config->model.type == "whisper")
    return std::make_shared<Whisper_Model>(std::move(config), ort_env);
  if (config->model.type == "phi3v")
    return std::make_shared<MultiModalVisionModel>(std::move(config), ort_env);

  throw std::runtime_error("Unsupported model_type in config.json: " + config->model.type);
}

std::shared_ptr<GeneratorParams> CreateGeneratorParams(const Model& model) {
  return std::make_shared<GeneratorParams>(model);
}

// Used by benchmarking tests only, should not be used normally
std::shared_ptr<GeneratorParams> CreateGeneratorParams() {
  return std::make_shared<GeneratorParams>();
}

void ConvertFp16ToFp32(OrtAllocator& allocator, OrtValue& in, std::unique_ptr<OrtValue>& p_out, DeviceType device_type, cudaStream_t stream) {
  auto shape_info = in.GetTensorTypeAndShapeInfo();
  auto shape = shape_info->GetShape();
  assert(shape_info->GetElementType() == Ort::TypeToTensorType<Ort::Float16_t>);

  bool allocate_p_out = p_out == nullptr;
  if (p_out) {
    auto out_shape_info = p_out->GetTensorTypeAndShapeInfo();
    auto out_shape = out_shape_info->GetShape();
    allocate_p_out = shape != out_shape;
  }

  if (allocate_p_out)
    p_out = OrtValue::CreateTensor<float>(allocator, shape);

  int count = static_cast<int>(shape_info->GetElementCount());
  auto* fp16 = in.GetTensorData<uint16_t>();
  auto* fp32 = p_out->GetTensorMutableData<float>();

  switch (device_type) {
    case DeviceType::DML:
      // DML doesn't currently support on-device scoring, so we fall back to the CPU
    case DeviceType::CPU:
      for (int i = 0; i < count; i++)
        fp32[i] = FastFloat16ToFloat32(fp16[i]);
      break;

#if USE_CUDA
    case DeviceType::CUDA:
      cuda::LaunchFp16ToFp32(fp16, fp32, count, stream);
      break;
#endif

    default:
      throw std::runtime_error("ConvertFp16ToFp32 - Unsupported device type");
  }
}

void ConvertFp32ToFp16(OrtAllocator& allocator, OrtValue& in, std::unique_ptr<OrtValue>& p_out,
                       DeviceType device_type, cudaStream_t stream) {
  auto shape_info = in.GetTensorTypeAndShapeInfo();
  auto shape = shape_info->GetShape();
  assert(shape_info->GetElementType() == Ort::TypeToTensorType<float>);

  bool allocate_p_out = p_out == nullptr;
  if (p_out) {
    auto out_shape_info = p_out->GetTensorTypeAndShapeInfo();
    auto out_shape = out_shape_info->GetShape();
    allocate_p_out = shape != out_shape;
  }

  if (allocate_p_out)
    p_out = OrtValue::CreateTensor<float>(allocator, shape);

  int count = static_cast<int>(shape_info->GetElementCount());
  auto* fp32 = in.GetTensorData<float>();
  auto* fp16 = p_out->GetTensorMutableData<uint16_t>();

  switch (device_type) {
    case DeviceType::DML:
    case DeviceType::CPU:
      for (int i = 0; i < count; i++)
        fp16[i] = FastFloat32ToFloat16(fp32[i]);
      break;

#if USE_CUDA
    case DeviceType::CUDA:
      // TODO: Implement for CUDA. For now, fallthrough and report an error.
#endif

    default:
      throw std::runtime_error("ConvertFp32ToFp16 - Unsupported device type");
  }
}

std::unique_ptr<OrtValue> Model::ExpandInputs(std::unique_ptr<OrtValue>& input, int num_beams) const {
  // Input shape (batch_size, sequence_length). The input is required with data type T.
  // Output shape (batch_size * num_beams, sequence_length)

  // If we're on CUDA, we still want to do the copy to move the data over to CUDA memory where we will read from it later.
  // DML doesn't currently support on-device scoring, so we go the same route as the CPU
  if (num_beams == 1 && (device_type_ == DeviceType::CPU || device_type_ == DeviceType::DML)) {
    return std::move(input);
  }

  auto input_type_info = input->GetTensorTypeAndShapeInfo();
  auto element_type = input_type_info->GetElementType();
  auto element_size = SizeOf(element_type);
  auto input_shape = input_type_info->GetShape();
  const int64_t batch_size = input_shape[0];
  const int64_t data_size_bytes = input_type_info->GetElementCount() * element_size / batch_size;

  input_shape[0] *= num_beams;

  auto& allocator = device_type_ == DeviceType::DML ? allocator_cpu_ : *allocator_device_;
  auto expanded = OrtValue::CreateTensor(allocator, input_shape, element_type);
  const auto* input_data = reinterpret_cast<const uint8_t*>(input->GetTensorRawData());
  auto* expanded_data = reinterpret_cast<uint8_t*>(expanded->GetTensorMutableRawData());
  auto* target = expanded_data;

  switch (device_type_) {
    case DeviceType::DML:
      // DML doesn't currently support on-device scoring, so we use the CPU for non-cache inputs/outputs
    case DeviceType::CPU:
      for (int i = 0; i < batch_size; i++) {
        for (int j = 0; j < num_beams; j++) {
          memcpy(target, input_data + i * data_size_bytes, data_size_bytes);
          target += data_size_bytes;
        }
      }
      break;

#if USE_CUDA
    case DeviceType::CUDA:
      for (int i = 0; i < batch_size; i++) {
        for (int j = 0; j < num_beams; j++) {
          cudaMemcpyAsync(target, input_data + i * data_size_bytes, data_size_bytes, cudaMemcpyHostToDevice, cuda_stream_);
          target += data_size_bytes;
        }
      }
      break;
#endif
    default:
      throw std::runtime_error("ExpandInputs - Unsupported device type");
  }
  return expanded;
}

MultiModalProcessor::MultiModalProcessor(Config& config, const SessionInfo& session_info)
    : tokenizer_{std::make_shared<Tokenizer>(config)} {
  if (config.model.type == "phi3v") {
    image_processor_ = std::make_shared<ImageProcessor>(config, session_info);
  } else {
    throw std::runtime_error("MultiModalProcessor cannot be created. Expected a multimodal model. Actual: " + config.model.type);
  }
}

}  // namespace Generators
