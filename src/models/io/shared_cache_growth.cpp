// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "shared_cache_growth.h"
#include "../graph_builder.h"

#include <algorithm>

namespace Generators {

bool SharedCacheGrowth::IsSupported(const State& state, std::span<const std::string> input_names,
                                    std::span<const std::string> output_names) {
  if (Ort::api->GetModelEditorApi() == nullptr) return false;
  const auto& model = state.model_;
  if (model.p_device_kvcache_->GetType() != DeviceType::WEBGPU ||
      !state.params_->IsPastPresentShareBufferEnabled(model.config_->model.type) ||
      state.params_->search.num_beams != 1 || model.config_->model.decoder.sliding_window ||
      model.p_device_kvcache_->GetKeyValueCacheQuantizationBits(model.config_->model.decoder.session_options) != 0) {
    return false;
  }
#if ORT_API_VERSION < 27
  if (state.params_->use_graph_capture) return false;
#endif
  if (input_names.empty() || input_names.size() != output_names.size()) return false;
  for (size_t i = 0; i < input_names.size(); ++i) {
    const auto input_shape = model.session_info_.GetInputShape(input_names[i]);
    const auto output_shape = model.session_info_.GetOutputShape(output_names[i]);
    const auto type = model.session_info_.GetInputDataType(input_names[i]);
    if (input_shape.size() != 4 || output_shape.size() != 4 || input_shape[2] >= 0 || output_shape[2] >= 0 ||
        (type != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16 && type != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT)) {
      return false;
    }
  }
  return true;
}

void SharedCacheGrowth::Migrate(OrtValue& source, OrtValue& destination, int64_t extra_capacity) {
  if (!migration_session_) {
    const auto type = source.GetTensorTypeAndShapeInfo()->GetElementType();
    ModelConfig config("Pad");
    config.inputs.emplace_back("cache", type, std::vector<int64_t>{-1, -1, -1, -1});
    config.inputs.emplace_back("pads", ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64, std::vector<int64_t>{8});
    config.outputs.emplace_back("grown", type, std::vector<int64_t>{-1, -1, -1, -1});
    auto graph = GraphBuilder::Build(config);
    auto options = state_.model_.session_options_->Clone();
    options->SetIntraOpNumThreads(1);
    options->AddConfigEntry("session.disable_cpu_ep_fallback", "1");
    OrtSession* session = nullptr;
    Ort::ThrowOnError(Ort::GetModelEditorApi().CreateSessionFromModel(
        &GetOrtEnv(), graph.get(), options.get(), &session));
    migration_session_.reset(session);
  }

  auto pads = OrtValue::CreateTensor<int64_t>(state_.model_.allocator_cpu_, std::array<int64_t, 1>{8});
  auto values = std::span<int64_t>{pads->GetTensorMutableData<int64_t>(), 8};
  std::fill(values.begin(), values.end(), 0);
  values[6] = extra_capacity;
  const char* input_names[] = {"cache", "pads"};
  const char* output_names[] = {"grown"};
  const OrtValue* inputs[] = {&source, pads.get()};
  OrtValue* outputs[] = {&destination};
  auto options = OrtRunOptions::Create();
  options->AddConfigEntry("gpu_graph_id", "-1");
  migration_session_->Run(options.get(), input_names, inputs, 2, output_names, outputs, 1);
}

int64_t SharedCacheGrowth::EnsureCapacity(int total_length, bool has_content,
                                          std::vector<std::unique_ptr<OrtValue>>& tensors,
                                          size_t input_index, size_t output_index) {
  const int64_t limit = state_.params_->search.max_length;
  if (total_length < 0 || total_length > limit) {
    throw std::runtime_error("Shared KV cache length exceeds search.max_length.");
  }
  // Reserve the first decode token with prefill so an exact capacity boundary
  // does not immediately require copying the cache and invalidating capture.
  const int64_t required_capacity = total_length + (!has_content && total_length < limit ? 1 : 0);
  const int64_t old_capacity = tensors.front()->GetTensorTypeAndShapeInfo()->GetShape()[2];
  if (required_capacity <= old_capacity) return old_capacity;

  int64_t capacity = std::max<int64_t>(1, old_capacity);
  while (capacity < required_capacity) capacity += std::min(capacity, limit - capacity);

  // Release only this generator's captures before changing any bound addresses.
  // A failure leaves the old tensors bound and usable; never replay a stale graph.
  state_.InvalidateCapturedGraphs();
  std::vector<std::unique_ptr<OrtValue>> grown;
  grown.reserve(tensors.size());
  for (auto& tensor : tensors) {
    const auto info = tensor->GetTensorTypeAndShapeInfo();
    auto shape = info->GetShape();
    shape[2] = capacity;
    auto value = OrtValue::CreateTensor(state_.model_.p_device_kvcache_->GetAllocator(), shape, info->GetElementType());
    if (has_content) Migrate(*tensor, *value, capacity - old_capacity);
    grown.push_back(std::move(value));
  }
  // Commit all bindings together so a failed allocation/copy can be retried.
  tensors.swap(grown);
  for (size_t i = 0; i < tensors.size(); ++i) {
    state_.inputs_[input_index + i] = tensors[i].get();
    state_.outputs_[output_index + i] = tensors[i].get();
  }
  return capacity;
}

}  // namespace Generators
