#include <algorithm>
#include <thread>

#include "../generators.h"
#include "../search.h"
#include "model.h"
#include "gpt.h"
#include "decoder_only.h"
#include "whisper.h"
#include "kernels.h"
#ifdef USE_DML
//  Because dml_provider_factory includes windows headers that #define min and max, this next line will prevent this from happening
#define NOMINMAX
#include "dml_provider_factory.h"

EXTERN_C IMAGE_DOS_HEADER __ImageBase;

static std::wstring CurrentModulePath() {
  wchar_t path[MAX_PATH];
  GetModuleFileNameW((HINSTANCE)&__ImageBase, path, _countof(path));

  wchar_t absolute_path[MAX_PATH];
  wchar_t* name;
  GetFullPathNameW(path, _countof(path), absolute_path, &name);

  auto idx = std::distance(absolute_path, name);
  auto out_path = std::wstring(absolute_path);
  out_path.resize(idx);

  return out_path;
}
#endif

namespace Generators {

State::State(const GeneratorParams& params) : params_{params.shared_from_this()} {
}

void State::Run(OrtSession& session, OrtRunOptions& run_options) {
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

#ifdef NO_TOKENIZER
const std::string& TokenizerStream::Decode(int32_t token) {
  throw std::runtime_error("Tokenizer not enabled");
}

std::unique_ptr<TokenizerStream> Tokenizer::CreateStream() const {
  return std::make_unique<TokenizerStream>();
}

Tokenizer::Tokenizer(Config& config) {
}

std::vector<int32_t> Tokenizer::Encode(const char* text) const {
  throw std::runtime_error("Tokenizer not enabled");
}

std::string Tokenizer::Decode(std::span<int32_t> tokens) const {
  throw std::runtime_error("Tokenizer not enabled");
}
#else
void CheckResult(tfmError_t error) {
  if (error != kTfmOK)
    throw std::runtime_error(TfmGetLastErrorMessage());
}

TokenizerStream::TokenizerStream(const Tokenizer& tokenizer)
    : tokenizer_{tokenizer.shared_from_this()} {
  CheckResult(TfmCreate(kTfmKindDetokenizerCache, cache_.Address()));
}

const std::string& TokenizerStream::Decode(int32_t token) {
  const char* string;
  CheckResult(TfmDetokenizeCached(tokenizer_->tokenizer_, cache_, token, &string));
  chunk_ = string;
  return chunk_;
}

Tokenizer::Tokenizer(Config& config) : pad_token_id_{config.model.pad_token_id} {
  CheckResult(TfmCreateTokenizer(tokenizer_.Address(), reinterpret_cast<const char*>(config.config_path.u8string().c_str())));
}

std::unique_ptr<TokenizerStream> Tokenizer::CreateStream() const {
  return std::make_unique<TokenizerStream>(*this);
}

std::vector<int32_t> Tokenizer::Encode(const char* text) const {
  TfmPtr<TfmTokenId2DArray> ids;
  CheckResult(TfmTokenize(tokenizer_, &text, 1, ids.Address()));

  const tfmTokenId_t* tokens;
  size_t count;
  CheckResult(TfmTokenId2DArrayGetItem(ids, 0, &tokens, &count));
  return {tokens, tokens + count};
}

std::string Tokenizer::Decode(std::span<const int32_t> tokens) const {
  TfmPtr<TfmStringArray> tfm_string_array;
  CheckResult(TfmDetokenize1D(tokenizer_, reinterpret_cast<const uint32_t*>(tokens.data()), tokens.size(), tfm_string_array.Address()));

  const char* string;
  CheckResult(TfmStringArrayGetItem(tfm_string_array, 0, &string));
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

#endif

#if USE_CUDA
// Since Python/Others can and will hold onto a generator object past the model object's lifetime we need to ensure
// the allocator used is not destroyed until last. This keeps the allocator around until exit, after all other memory
// has been destroyed. Without this, we will crash in the Onnxruntime BFCArena code when deleting tensors due to the
// arena already being destroyed.
Ort::Allocator* GetCudaAllocator(OrtSession& session) {
  static std::unique_ptr<OrtMemoryInfo> memory_info_cuda_;
  static std::unique_ptr<Ort::Allocator> allocator_cuda_;

  if (!allocator_cuda_) {
    memory_info_cuda_ = OrtMemoryInfo::Create("Cuda", OrtAllocatorType::OrtDeviceAllocator, 0, OrtMemType::OrtMemTypeDefault);
    allocator_cuda_ = Ort::Allocator::Create(session, *memory_info_cuda_);
  }
  return allocator_cuda_.get();
}
#endif

#if USE_DML
// Since Python/Others can and will hold onto a generator object past the model object's lifetime we need to ensure
// the allocator used is not destroyed until last. This keeps the allocator around until exit, after all other memory
// has been destroyed.
Ort::Allocator* GetDmlAllocator(OrtSession& session) {
  static std::unique_ptr<OrtMemoryInfo> memory_info_dml_;
  static std::unique_ptr<Ort::Allocator> allocator_dml_;

  if (!allocator_dml_) {
    memory_info_dml_ = OrtMemoryInfo::Create("DML", OrtAllocatorType::OrtDeviceAllocator, 0, OrtMemType::OrtMemTypeDefault);
    allocator_dml_ = Ort::Allocator::Create(session, *memory_info_dml_);
  }
  return allocator_dml_.get();
}
#endif

SessionInfo::SessionInfo(OrtSession& session) {
  auto input_names = session.GetInputNames();
  std::vector<ONNXTensorElementDataType> input_types(input_names.size());
  for (size_t i = 0; i < input_types.size(); i++) {
    auto input_type = session.GetInputTypeInfo(i)->GetTensorTypeAndShapeInfo().GetElementType();
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
    allocator_device_ = GetDmlAllocator(session);
  }
#endif

  session_info_ = std::make_unique<SessionInfo>(session);
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
    std::filesystem::path profile_file_prefix{options.enable_profiling.value()};
    ort_options.EnableProfiling(profile_file_prefix.c_str());
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
#ifdef USE_DML
    } else if (provider_options.name == "dml") {
      device_type_ = DeviceType::DML;  // We use a DML allocator for input/output caches, but other tensors will use CPU tensors
      const OrtDmlApi* p_dml_api{};
      Ort::ThrowOnError(Ort::api->GetExecutionProviderApi("DML", ORT_API_VERSION, reinterpret_cast<const void**>(&p_dml_api)));
      if (!p_dml_api)
        throw std::runtime_error("Unexpected nullptr getting OrtDmlApi");
      auto directml_dll = CurrentModulePath() + L"DirectML.dll";
      if (LoadLibraryExW(directml_dll.c_str(), nullptr, 0) == NULL)
        throw std::runtime_error("DirectML.dll not found");
      p_dml_api->SessionOptionsAppendExecutionProvider_DML(&ort_options, 0);
#endif
    } else
      throw std::runtime_error("Unknown provider type: " + provider_options.name);
  }
}

std::shared_ptr<Tokenizer> Model::CreateTokenizer() const {
  return std::make_shared<Tokenizer>(*config_);
}

std::shared_ptr<Model> CreateModel(OrtEnv& ort_env, const char* config_path) {
  auto config = std::make_unique<Config>(config_path);

  if (config->model.type == "gpt2")
    return std::make_shared<Gpt_Model>(std::move(config), ort_env);
  if (config->model.type == "llama" || config->model.type == "gemma" || config->model.type == "mistral" || config->model.type == "phi" || config->model.type == "phi_longrope" || config->model.type == "phi3mini")
    return std::make_shared<DecoderOnly_Model>(std::move(config), ort_env);
  if (config->model.type == "whisper")
    return std::make_shared<Whisper_Model>(std::move(config), ort_env);

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
  assert(shape_info->GetElementType() == Ort::TypeToTensorType<Ort::Float16_t>::type);

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
        fp32[i] = Float16ToFloat32(fp16[i]);
      break;

#ifdef USE_CUDA
    case DeviceType::CUDA:
      cuda::LaunchFp16ToFp32(fp16, fp32, count, stream);
      break;
#endif

    default:
      throw std::runtime_error("ConvertFp16ToFp32 - Unsupported device type");
  }
}

size_t GetOrtTypeSize(ONNXTensorElementDataType type) {
  switch (type) {
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT:
      return sizeof(float);
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16:
      return sizeof(Ort::Float16_t);
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_BFLOAT16:
      return sizeof(Ort::BFloat16_t);
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE:
      return sizeof(double);
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8:
      return sizeof(int8_t);
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8:
      return sizeof(uint8_t);
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16:
      return sizeof(int16_t);
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16:
      return sizeof(uint16_t);
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32:
      return sizeof(int32_t);
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT32:
      return sizeof(uint32_t);
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64:
      return sizeof(int64_t);
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT64:
      return sizeof(uint64_t);
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL:
      return sizeof(bool);
    default:
      throw std::runtime_error("Unsupported ONNXTensorElementDataType in GetTypeSize");
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
  auto element_size = GetOrtTypeSize(element_type);
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

void Model::GetMaxBatchSizeFromGeneratorParams(const GeneratorParams& params) {
  bool is_cuda_graph_enabled = IsCudaGraphEnabled(config_->model.decoder.session_options);
  max_batch_size_ = params.max_batch_size;

  if (DeviceType::CUDA == device_type_) {
    if (max_batch_size_ == 0 && is_cuda_graph_enabled) {
      throw std::runtime_error("CUDA graph is enabled, but max_batch_size is not set.");
    }
    if (max_batch_size_ > 0) {
      if (!is_cuda_graph_enabled) {
        throw std::runtime_error("CUDA graph is not enabled.");
      }
      use_cuda_graph_ = true;
    }
  } else {
    if (is_cuda_graph_enabled || max_batch_size_ > 0) {
      throw std::runtime_error("CUDA graph is not supported on this device");
    }
  }
}

}  // namespace Generators
