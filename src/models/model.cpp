#include <algorithm>
#include <thread>

#include "../generators.h"
#include "../search.h"
#include "model.h"
#include "debugging.h"
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

static ComPtr<ID3D12Resource> CreateD3D12ResourceOfByteSize(
    ID3D12Device* d3dDevice,
    size_t resourceByteSize,
    D3D12_HEAP_TYPE heapType = D3D12_HEAP_TYPE_DEFAULT,
    D3D12_RESOURCE_STATES resourceState = D3D12_RESOURCE_STATE_COMMON,
    D3D12_RESOURCE_FLAGS resourceFlags = D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS) {
  resourceByteSize = std::max(resourceByteSize, size_t(DML_MINIMUM_BUFFER_TENSOR_ALIGNMENT));

  // DML needs the resources' sizes to be a multiple of 4 bytes
  (resourceByteSize += 3) &= ~3;

  D3D12_HEAP_PROPERTIES heapProperties = {};
  heapProperties.Type = heapType;
  heapProperties.CPUPageProperty = D3D12_CPU_PAGE_PROPERTY_UNKNOWN;
  heapProperties.MemoryPoolPreference = D3D12_MEMORY_POOL_UNKNOWN;
  heapProperties.CreationNodeMask = 1;
  heapProperties.VisibleNodeMask = 1;

  D3D12_RESOURCE_DESC resourceDesc = {};
  resourceDesc.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
  resourceDesc.Alignment = 0;
  resourceDesc.Width = static_cast<uint64_t>(resourceByteSize);
  resourceDesc.Height = 1;
  resourceDesc.DepthOrArraySize = 1;
  resourceDesc.MipLevels = 1;
  resourceDesc.Format = DXGI_FORMAT_UNKNOWN;
  resourceDesc.SampleDesc = {1, 0};
  resourceDesc.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
  resourceDesc.Flags = resourceFlags;

  ComPtr<ID3D12Resource> gpuResource;
  THROW_IF_FAILED(d3dDevice->CreateCommittedResource(
      &heapProperties,
      D3D12_HEAP_FLAG_NONE,
      &resourceDesc,
      resourceState,
      nullptr,
      IID_PPV_ARGS(&gpuResource)));

  return gpuResource;
}

static void UploadDataToDml(
    DmlObjects& dml_objects,
    ID3D12Resource* destinationResource,
    uint64_t dst_offset,
    std::span<const uint8_t> sourceData) {
  // Get the size of the resource.
  D3D12_RESOURCE_DESC resourceDesc = destinationResource->GetDesc();
  assert(resourceDesc.Dimension == D3D12_RESOURCE_DIMENSION_BUFFER);
  const size_t dataSizeInBytes = static_cast<size_t>(resourceDesc.Width);

  // Create intermediate upload resource visible to both CPU and GPU.
  dml_objects.upload_buffer = CreateD3D12ResourceOfByteSize(dml_objects.d3d12Device.Get(), dataSizeInBytes, D3D12_HEAP_TYPE_UPLOAD, D3D12_RESOURCE_STATE_GENERIC_READ, D3D12_RESOURCE_FLAG_NONE);

  // Copy CPU-side data to shared memory that is both CPU and GPU visible.
  size_t clampedDataByteSize = std::min(dataSizeInBytes, sourceData.size());
  uint8_t* uploadBufferData = nullptr;
  THROW_IF_FAILED(dml_objects.upload_buffer->Map(0, nullptr, reinterpret_cast<void**>(&uploadBufferData)));
  memcpy(uploadBufferData, sourceData.data(), clampedDataByteSize);
  dml_objects.upload_buffer->Unmap(0, nullptr);

  D3D12_RESOURCE_BARRIER resourceBarrier{};
  resourceBarrier.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
  resourceBarrier.Flags = D3D12_RESOURCE_BARRIER_FLAG_NONE;
  resourceBarrier.Transition = {};
  resourceBarrier.Transition.pResource = destinationResource;
  resourceBarrier.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
  resourceBarrier.Transition.StateBefore = D3D12_RESOURCE_STATE_COPY_DEST;
  resourceBarrier.Transition.StateAfter = D3D12_RESOURCE_STATE_UNORDERED_ACCESS;

  // Issue deferred command to copy from the intermediate shared resource to the final GPU resource,
  // and then execute the commands.
  dml_objects.commandList->CopyBufferRegion(destinationResource, dst_offset, dml_objects.upload_buffer.Get(), dst_offset, sourceData.size());
  dml_objects.commandList->ResourceBarrier(1, &resourceBarrier);
  THROW_IF_FAILED(dml_objects.commandList->Close());
  ID3D12CommandList* commandLists[] = {dml_objects.commandList.Get()};
  dml_objects.commandQueue->ExecuteCommandLists(ARRAYSIZE(commandLists), commandLists);

  THROW_IF_FAILED(dml_objects.commandAllocator->Reset());
  THROW_IF_FAILED(dml_objects.commandList->Reset(dml_objects.commandAllocator.Get(), nullptr));
}

#endif

namespace Generators {

State::State(const GeneratorParams& params) : params_{params.shared_from_this()} {
}

void State::Run(OrtSession& session) {
#if 0
  // To show input values, enable this block (output values will be shapes only at this point)
  printf("**Inputs:\r\n");
  DumpTensors(inputs_.data(), input_names_.data(), input_names_.size(), true);
  printf("**Outputs:\r\n");
  DumpTensors(outputs_.data(), output_names_.data(), output_names_.size(), false);
#endif

  session.Run(nullptr, input_names_.data(), inputs_.data(), input_names_.size(), output_names_.data(), outputs_.data(), output_names_.size());

#if 0
  // To show the output values, enable this block
  printf("**Outputs:\r\n");
  DumpTensors(outputs_.data(), output_names_.data(), output_names_.size(), true);
#endif
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
  CreateSessionOptions();
}

Model::~Model() = default;

void Model::InitDeviceAllocator([[maybe_unused]] OrtSession& session) {
  allocator_device_ = &allocator_cpu_;
#if USE_CUDA
  if (device_type_ == DeviceType::CUDA) {
    allocator_device_ = GetCudaAllocator(session);
  }
#endif

#if USE_DML
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
      dml_objects_ = CreateDmlObjects();
      dml_execution_context_ = std::make_unique<DmlExecutionContext>(dml_objects_.d3d12Device.Get(), dml_objects_.commandQueue.Get());
      dml_pooled_upload_heap_ = std::make_unique<DmlPooledUploadHeap>(dml_objects_.d3d12Device.Get(), dml_execution_context_.get());
      dml_readback_heap_ = std::make_unique<DmlReadbackHeap>(dml_objects_.d3d12Device.Get());

      Ort::ThrowOnError(Ort::api->GetExecutionProviderApi("DML", ORT_API_VERSION, reinterpret_cast<const void**>(&p_dml_api_)));
      if (!p_dml_api_)
        throw std::runtime_error("Unexpected nullptr getting OrtDmlApi");
      auto directml_dll = CurrentModulePath() + L"DirectML.dll";
      if (LoadLibraryExW(directml_dll.c_str(), nullptr, 0) == NULL)
        throw std::runtime_error("DirectML.dll not found");
      p_dml_api_->SessionOptionsAppendExecutionProvider_DML1(&ort_options, nullptr, dml_objects_.commandQueue.Get());
      device_type_ = DeviceType::DML;  // Scoring will use DML
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
  if (config->model.type == "llama" || config->model.type == "gemma" || config->model.type == "mistral" || config->model.type == "phi")
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

#if USE_CUDA
void ConvertFp16ToFp32(OrtAllocator& allocator, cudaStream_t stream, OrtValue& in, std::unique_ptr<OrtValue>& p_out) {
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

  cuda::LaunchFp16ToFp32(fp16, fp32, count, stream);
}
#endif

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

  // If we're on CUDA, we still want to do the copy to move the data over to CUDA memory where we will read from it later
  if (num_beams == 1 && device_type_ == DeviceType::CPU) {
    return std::move(input);
  }

  auto input_type_info = input->GetTensorTypeAndShapeInfo();
  auto element_type = input_type_info->GetElementType();
  auto element_size = GetOrtTypeSize(element_type);
  auto input_shape = input_type_info->GetShape();
  const int64_t batch_size = input_shape[0];
  const int64_t data_size_bytes = input_type_info->GetElementCount() * element_size / batch_size;

  input_shape[0] *= num_beams;

  auto expanded = OrtValue::CreateTensor(*allocator_device_, input_shape, element_type);

  const auto* input_data = reinterpret_cast<const uint8_t*>(input->GetTensorRawData());

  switch (device_type_) {
    case DeviceType::CPU: {
      auto* expanded_data = reinterpret_cast<uint8_t*>(expanded->GetTensorMutableRawData());
      auto* target = expanded_data;

      for (int i = 0; i < batch_size; i++) {
        for (int j = 0; j < num_beams; j++) {
          memcpy(target, input_data + i * data_size_bytes, data_size_bytes);
          target += data_size_bytes;
        }
      }
    } break;

#if USE_CUDA
    case DeviceType::CUDA: {
      auto* expanded_data = reinterpret_cast<uint8_t*>(expanded->GetTensorMutableRawData());
      auto* target = expanded_data;

      for (int i = 0; i < batch_size; i++) {
        for (int j = 0; j < num_beams; j++) {
          cudaMemcpyAsync(target, input_data + i * data_size_bytes, data_size_bytes, cudaMemcpyHostToDevice, cuda_stream_);
          target += data_size_bytes;
        }
      }

    } break;
#endif

#if USE_DML
    case DeviceType::DML: {
      ComPtr<ID3D12Resource> target_resource;
      Ort::ThrowOnError(p_dml_api_->GetD3D12ResourceFromAllocation(allocator_device_, expanded->GetTensorMutableRawData(), &target_resource));
      uint64_t target_offset = 0;

      for (int i = 0; i < batch_size; i++) {
        for (int j = 0; j < num_beams; j++) {
          // TODO (pavignol): Batch the uploads into a single command list and/or copy
          auto data_span = std::span<const uint8_t>(reinterpret_cast<const uint8_t*>(input_data + i * data_size_bytes), data_size_bytes);
          dml_pooled_upload_heap_->BeginUploadToGpu(target_resource.Get(), target_offset, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, data_span);
          target_offset += data_size_bytes;
        }
      }
    } break;
#endif

    default:
      throw std::runtime_error("ExpandInputs - Unsupported device type");
  }
  return expanded;
}

}  // namespace Generators
