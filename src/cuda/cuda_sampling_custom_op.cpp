// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#define ORT_API_MANUAL_INIT
#include "onnxruntime_cxx_api.h"
#undef ORT_API_MANUAL_INIT

#include <algorithm>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <vector>

#include "cuda_sampling.h"

namespace Generators::cuda {
namespace {

constexpr const char* kDomain = "com.microsoft.genai";

class FusedTopKSampleKernel {
 public:
  explicit FusedTopKSampleKernel(const OrtKernelInfo* info) {
    Ort::ConstKernelInfo kernel_info{info};
    top_k_ = static_cast<int>(kernel_info.GetAttribute<int64_t>("top_k"));
    top_p_ = kernel_info.GetAttribute<float>("top_p");
    temperature_ = kernel_info.GetAttribute<float>("temperature");
    if (top_k_ < 1 || top_k_ > kFusedSamplingMaxK) {
      throw std::invalid_argument("FusedTopKSample top_k must be in [1, 256]");
    }
    if (!(top_p_ > 0.0f && top_p_ <= 1.0f)) {
      throw std::invalid_argument("FusedTopKSample top_p must be in (0, 1]");
    }
    if (!(temperature_ > 0.0f)) {
      throw std::invalid_argument("FusedTopKSample temperature must be positive");
    }
  }

  void Compute(OrtKernelContext* raw_context) {
    std::lock_guard<std::mutex> lock{workspace_mutex_};
    Ort::KernelContext context{raw_context};
    const auto logits = context.GetInput(0);
    const auto uniforms = context.GetInput(1);
    const auto logits_shape = logits.GetTensorTypeAndShapeInfo().GetShape();
    const auto uniform_shape = uniforms.GetTensorTypeAndShapeInfo().GetShape();
    if (logits_shape.size() != 2 || uniform_shape.size() != 1 ||
        uniform_shape[0] != logits_shape[0]) {
      throw std::invalid_argument(
          "FusedTopKSample expects logits [batch,vocab] and uniforms [batch]");
    }

    const int batch_size = static_cast<int>(logits_shape[0]);
    const int vocab_size = static_cast<int>(logits_shape[1]);
    if (batch_size < 1 || vocab_size < top_k_) {
      throw std::invalid_argument("FusedTopKSample received an invalid batch or vocabulary size");
    }

    auto* stream = static_cast<cudaStream_t>(context.GetGPUComputeStream());
    if (!sampling_data_ || batch_size > workspace_batch_ || vocab_size != workspace_vocab_) {
      sampling_data_ = std::make_unique<SamplingData>(0, batch_size, vocab_size, stream);
      workspace_batch_ = batch_size;
      workspace_vocab_ = vocab_size;
    }

    const std::vector<int64_t> token_shape{batch_size};
    const std::vector<int64_t> sparse_shape{batch_size, top_k_};
    auto token = context.GetOutput(0, token_shape);
    auto sparse_indices = context.GetOutput(1, sparse_shape);
    auto sparse_probs = context.GetOutput(2, sparse_shape);

    RunTopK(sampling_data_.get(), stream, logits.GetTensorData<float>(), vocab_size,
            batch_size, top_k_);
    LaunchFusedSampleKernelWithOutput(
        sampling_data_.get(), stream, sampling_data_->topk_scores,
        sampling_data_->topk_indices, uniforms.GetTensorData<float>(),
        token.GetTensorMutableData<int64_t>(), sparse_indices.GetTensorMutableData<int64_t>(),
        sparse_probs.GetTensorMutableData<float>(), top_k_, batch_size, top_p_, temperature_,
        sampling_data_->topk_stride);
  }

 private:
  int top_k_{};
  float top_p_{};
  float temperature_{};
  int workspace_batch_{};
  int workspace_vocab_{};
  std::unique_ptr<SamplingData> sampling_data_;
  std::mutex workspace_mutex_;
};

struct FusedTopKSampleOp : Ort::CustomOpBase<FusedTopKSampleOp, FusedTopKSampleKernel> {
  void* CreateKernel(const OrtApi&, const OrtKernelInfo* info) const {
    return new FusedTopKSampleKernel(info);
  }

  const char* GetName() const { return "FusedTopKSample"; }
  const char* GetExecutionProviderType() const { return "CUDAExecutionProvider"; }

  size_t GetInputTypeCount() const { return 2; }
  ONNXTensorElementDataType GetInputType(size_t) const {
    return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;
  }

  size_t GetOutputTypeCount() const { return 3; }
  ONNXTensorElementDataType GetOutputType(size_t index) const {
    return index < 2 ? ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64
                     : ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;
  }
};

void RetainDomain(Ort::CustomOpDomain&& domain) {
  static std::vector<Ort::CustomOpDomain> domains;
  static std::mutex mutex;
  std::lock_guard<std::mutex> lock{mutex};
  domains.push_back(std::move(domain));
}

}  // namespace
}  // namespace Generators::cuda

extern "C" OrtStatus* ORT_API_CALL RegisterCustomOps(OrtSessionOptions* options,
                                                       const OrtApiBase* api_base) {
  const OrtApi* api = api_base->GetApi(ORT_API_VERSION);
  Ort::InitApi(api);
  try {
    static Generators::cuda::FusedTopKSampleOp op;
    Ort::CustomOpDomain domain{Generators::cuda::kDomain};
    domain.Add(&op);
    Ort::UnownedSessionOptions{options}.Add(domain);
    Generators::cuda::RetainDomain(std::move(domain));
    return nullptr;
  } catch (const std::exception& error) {
    return api->CreateStatus(ORT_FAIL, error.what());
  }
}