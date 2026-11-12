// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "kv_cache.h"
#include "../../models/graph_builder.h"

#include <algorithm>

namespace Generators::WebGPU {

bool SharedCacheGrowth::IsSupported(const State& state, std::span<const std::string> input_names,
                                    std::span<const std::string> output_names) {
  const auto& model = state.model_;
  if (model.p_device_kvcache_->GetType() != DeviceType::WEBGPU ||
      !model.config_->model.decoder.pipeline.empty() ||
      !state.params_->search.past_present_share_buffer ||
      state.params_->search.num_beams != 1 || model.config_->model.decoder.sliding_window ||
      model.p_device_kvcache_->GetKeyValueCacheQuantizationBits(model.config_->model.decoder.session_options) != 0) {
    return false;
  }
#if ORT_API_VERSION < 27
  if (state.params_->use_graph_capture) return false;
#endif
  if (Ort::api_version < 22 || (state.params_->use_graph_capture && Ort::api_version < 27) ||
      Ort::api->GetModelEditorApi() == nullptr) return false;
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

void SharedCacheGrowth::InvalidateCapturedGraphs() {
  if (state_.graph_capture_session_ && Ort::api_version < 27) {
    throw std::runtime_error("WebGPU graph invalidation requires ORT 1.27 or newer.");
  }
#if ORT_API_VERSION >= 27
  if (state_.graph_capture_session_) {
    state_.graph_ids_.ForEachAssignedId([&](int id) { state_.graph_capture_session_->ReleaseCapturedGraph(id); });
  }
#else
  if (state_.graph_capture_session_) throw std::runtime_error("WebGPU graph invalidation requires ORT 1.27 or newer.");
#endif
  // IDs are never recycled. Recurrent fix-up already treats a new ID as a new
  // capture, so no epoch field is needed in the shared State or decoder layout.
  state_.graph_ids_.Clear();
  state_.graph_capture_fallback_logged_ = false;
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
  int64_t required_capacity = total_length + (!has_content && total_length < limit ? 1 : 0);
  // Keep short decodes within their initial allocation to avoid migration and
  // recapture, while leaving large prompts on the existing geometric policy.
  if (!has_content) required_capacity = std::max(required_capacity, std::min<int64_t>(512, limit));
  const int64_t old_capacity = tensors.front()->GetTensorTypeAndShapeInfo()->GetShape()[2];
  if (required_capacity <= old_capacity) return old_capacity;

  int64_t capacity = std::max<int64_t>(1, old_capacity);
  while (capacity < required_capacity) capacity += std::min(capacity, limit - capacity);

  // Release only this generator's captures before changing any bound addresses.
  // A failure leaves the old tensors bound and usable; never replay a stale graph.
  InvalidateCapturedGraphs();
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

SharedKeyValueCache::SharedKeyValueCache(State& state) : Generators::SharedKeyValueCache{state} {
  if (past_present_share_buffer_ && SharedCacheGrowth::IsSupported(state, input_name_strings_, output_name_strings_)) {
    growth_ = std::make_unique<SharedCacheGrowth>(state);
  }
}

void SharedKeyValueCache::Update(DeviceSpan<int32_t> beam_indices, int total_length) {
  if (growth_) {
    shape_[2] = growth_->EnsureCapacity(total_length, current_length_ > 0, presents_, input_index_, output_index_);
    for (auto& shape : layer_shapes_) shape[2] = shape_[2];
  }
  Generators::SharedKeyValueCache::Update(beam_indices, total_length);
}

ConvKeyValueCache::ConvKeyValueCache(State& state) : Generators::ConvKeyValueCache{state} {
  if (kv_share_buffer_ && SharedCacheGrowth::IsSupported(state, kv_input_name_strings_, kv_output_name_strings_)) {
    growth_ = std::make_unique<SharedCacheGrowth>(state);
  }
}

void ConvKeyValueCache::Update(DeviceSpan<int32_t> beam_indices, int total_length) {
  if (growth_) {
    kv_shape_[2] = growth_->EnsureCapacity(total_length, !kv_is_first_update_, kv_presents_, kv_input_index_, kv_output_index_);
    kv_is_first_update_ = false;
  }
  Generators::ConvKeyValueCache::Update(beam_indices, total_length);
}

}  // namespace Generators::WebGPU
