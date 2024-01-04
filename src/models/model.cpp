#include "../generators.h"
#include "../search.h"
#if USE_CUDA
#include "../search_cuda.h"
#endif
#include "model.h"
#include "gpt.h"
#include "llama.h"
#include "whisper.h"

namespace Generators {

Model::Model(OrtEnv& ort_env, const char* config_path, const ProviderOptions* provider_options) : config_{config_path} {
  auto session_options = OrtSessionOptions::Create();

  if (provider_options) {
#if USE_CUDA
    if (auto* options = std::get_if<OrtCUDAProviderOptions>(provider_options)) {
      cuda_stream_ = reinterpret_cast<cudaStream_t>(options->user_compute_stream);
      session_options->AppendExecutionProvider_CUDA(*options);
      device_type_ = DeviceType::CUDA;
    }
#endif
  }

  if (config_.model_type == "gpt2")
    arch_ = std::make_unique<Gpt_Model>(*this, ort_env, *session_options);
  else if (config_.model_type == "llama")
    arch_ = std::make_unique<Llama_Model>(*this, ort_env, *session_options);
  else if (config_.model_type == "whisper")
    arch_ = std::make_unique<Whisper_Model>(*this, ort_env, *session_options);
  else
    throw std::runtime_error("Unsupported model_type in config.json: " + config_.model_type);
}

Model::~Model() = default;

std::unique_ptr<State> Model::CreateState(RoamingArray<int32_t> sequence_lengths, const SearchParams& params) {
  return arch_->CreateState(sequence_lengths, params);
}

std::vector<int32_t> Model::Generate(const SearchParams& params) {
  auto search = params.CreateSearch();
  auto state = CreateState(search->GetSequenceLengths(), params);

  while (!search->IsDone()) {
    search->SetLogits(state->Run(search->GetSequenceLength(), search->GetNextTokens()));

    if (config_.top_p < 1.0f) {
      search->SampleTopP(config_.top_p, config_.temperature);
    } else if (config_.top_k > 1) {
      search->SampleTopK(config_.top_k, config_.temperature);
    } else
      search->SelectTop();
  }

  auto results = search->GetSequence(0);
  auto results_cpu = results.GetCPU();

  std::vector<int32_t> v;
  v.assign(results_cpu.begin(), results_cpu.end());
  return v;
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

std::unique_ptr<OrtValue> ExpandInputs(std::unique_ptr<OrtValue>& input, int num_beams, OrtAllocator& allocator, DeviceType device_type, cudaStream_t cuda_stream) {
  // Input shape (batch_size, sequence_length). The input is required with data type T.
  // Output shape (batch_size * num_beams, sequence_length)

  // If we're on CUDA, we still want to do the copy to move the data over to CUDA memory where we will read from it later
  if (num_beams == 1 && device_type==DeviceType::CPU)
    return std::move(input);

  auto input_type_info = input->GetTensorTypeAndShapeInfo();
  auto element_type = input_type_info->GetElementType();
  auto element_size = GetOrtTypeSize(element_type);
  auto input_shape = input_type_info->GetShape();
  const int64_t batch_size = input_shape[0];
  const int64_t data_size_bytes = input_type_info->GetElementCount() * element_size / batch_size;

  input_shape[0] *= num_beams;

  auto expanded = OrtValue::CreateTensor(allocator, input_shape, element_type);

  auto input_data = reinterpret_cast<const uint8_t*>(input->GetTensorRawData());
  auto expanded_data = reinterpret_cast<uint8_t*>(expanded->GetTensorMutableRawData());
  auto target = expanded_data;

  switch (device_type) {
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
          cudaMemcpyAsync(target, input_data + i * data_size_bytes, data_size_bytes, cudaMemcpyHostToDevice, cuda_stream);
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
