#pragma once

namespace Generators {

struct Logits {

  Logits(const SearchParams& search_params, DeviceType device_type, Ort::Allocator& allocator, cudaStream_t cuda_stream, ONNXTensorElementDataType score_type, bool uses_seq_length);
  void Update();
  RoamingArray<float> Get();

  DeviceType device_type_;
  cudaStream_t cuda_stream_;
  Ort::Allocator& allocator_;
  ONNXTensorElementDataType score_type_;

  std::array<int64_t, 3> logits_shape_;
  std::unique_ptr<OrtValue> logits_;
  std::unique_ptr<OrtValue> logits32_;  // When model output is fp16, this holds the fp32 conversion of them
};

}  // namespace Generators
