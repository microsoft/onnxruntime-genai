#include "../generators.h"
#include "../search.h"
#include "model.h"
#include "debugging.h"
#include "gpt.h"
#include "llama.h"
#include "mistral.h"
#include "phi2.h"
#include "whisper.h"

namespace Generators {

State::State(const GeneratorParams& search_params) : search_params_{search_params} {
}

void State::Run(OrtSession& session) {
#if 0
    printf("**Inputs:\r\n");
    DumpTensors(inputs_.data(), input_names_.data(), input_names_.size(), true);
    printf("**Outputs:\r\n");
    DumpTensors(outputs_.data(), output_names_.data(), output_names_.size(), false);
#endif

  session.Run(nullptr, input_names_.data(), inputs_.data(), input_names_.size(), output_names_.data(), outputs_.data(), output_names_.size());
}

void State::ClearIO() {
  input_names_.clear();
  output_names_.clear();
  inputs_.clear();
  outputs_.clear();
}

#ifdef NO_TOKENIZER
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

Tokenizer::Tokenizer(Config& config) {
  CheckResult(TfmCreateTokenizer(tokenizer_.Address(), reinterpret_cast<const char*>(config.config_path.u8string().c_str())));
}

std::vector<int32_t> Tokenizer::Encode(const char* text) const {
  TfmPtr<TfmTokenId2DArray> ids;
  CheckResult(TfmTokenize(tokenizer_, &text, 1, ids.Address()));

  const tfmTokenId_t* tokens;
  size_t count;
  CheckResult(TfmTokenId2DArrayGetItem(ids, 0, &tokens, &count));
  return {tokens, tokens + count};
}

std::string Tokenizer::Decode(std::span<int32_t> tokens) const {
  TfmPtr<TfmStringArray> tfm_string_array;
  CheckResult(TfmDetokenize1D(tokenizer_, reinterpret_cast<const uint32_t*>(tokens.data()), tokens.size(), tfm_string_array.Address()));

  const char* string;
  CheckResult(TfmStringArrayGetItem(tfm_string_array, 0, &string));
  return string;
}
#endif

Model::Model(std::unique_ptr<Config> config, const ProviderOptions* provider_options) : config_{std::move(config)} {
  session_options_ = OrtSessionOptions::Create();

  if (provider_options != nullptr) {
#if USE_CUDA
    if (auto* options = std::get_if<OrtCUDAProviderOptions>(provider_options)) {
      cuda_stream_ = reinterpret_cast<cudaStream_t>(options->user_compute_stream);
      session_options_->AppendExecutionProvider_CUDA(*options);
      device_type_ = DeviceType::CUDA;
    }
#endif
  }
}

Model::~Model() = default;

void Model::InitDeviceAllocator([[maybe_unused]] OrtSession& session) {
  allocator_device_ = &allocator_cpu_;
#if USE_CUDA
  if (device_type_ == DeviceType::CUDA) {
    memory_info_cuda_ = OrtMemoryInfo::Create("Cuda", OrtAllocatorType::OrtDeviceAllocator, 0, OrtMemType::OrtMemTypeDefault);
    allocator_cuda_ = Ort::Allocator::Create(session, *memory_info_cuda_);
    allocator_device_ = allocator_cuda_.get();
  }
#endif
}

std::unique_ptr<Tokenizer> Model::CreateTokenizer() const {
  return std::make_unique<Tokenizer>(*config_);
}

std::unique_ptr<Model> CreateModel(OrtEnv& ort_env, const char* config_path, const ProviderOptions* provider_options) {
  auto config = std::make_unique<Config>(config_path);

  if (config->model.type == "gpt2")
    return std::make_unique<Gpt_Model>(std::move(config), ort_env, provider_options);
  if (config->model.type == "llama")
    return std::make_unique<Llama_Model>(std::move(config), ort_env, provider_options);
  if (config->model.type == "mistral")
    return std::make_unique<Mistral_Model>(std::move(config), ort_env, provider_options);
  if (config->model.type == "phi2")
    return std::make_unique<Phi2_Model>(std::move(config), ort_env, provider_options);
  if (config->model.type == "whisper")
    return std::make_unique<Whisper_Model>(std::move(config), ort_env, provider_options);

  throw std::runtime_error("Unsupported model_type in config.json: " + config->model.type);
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
  auto* expanded_data = reinterpret_cast<uint8_t*>(expanded->GetTensorMutableRawData());
  auto* target = expanded_data;

  switch (device_type_) {
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

}  // namespace Generators
