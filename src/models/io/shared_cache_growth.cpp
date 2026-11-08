// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "shared_cache_growth.h"
#include "../graph_builder.h"
#include "kv_cache.h"

#include <algorithm>
#include <charconv>

namespace Generators {

bool SharedCacheGrowth::CanAutoShare(const Model& model, const GeneratorParams& params) {
  const auto& config = *model.config_;
  const auto& decoder = config.model.decoder;
  if (model.p_device_kvcache_->GetType() != DeviceType::WEBGPU || params.search.num_beams != 1 ||
      !decoder.pipeline.empty() || decoder.sliding_window || !decoder.inputs.past_names.empty() ||
      !decoder.inputs.cross_past_key_names.empty() ||
      model.p_device_kvcache_->GetKeyValueCacheQuantizationBits(decoder.session_options) != 0 ||
      Ort::api->GetModelEditorApi() == nullptr || decoder.filename.empty() ||
      config.memory_assets_only_ || config.model_data_spans_.contains(decoder.filename)) return false;
#if ORT_API_VERSION < 27
  if (params.use_graph_capture) return false;
#endif
  // A symbolic sequence axis alone does not establish in-place attention semantics.
  // Inspect the file once per Model, without loading its initializer payloads.
  std::call_once(model.webgpu_shared_kv_once_, [&] {
    model.webgpu_shared_kv_bindings_ = ReadOnnxSharedKvBindings(
        std::filesystem::path{(config.config_path / decoder.filename).c_str()});
  });
  const auto& bindings = model.webgpu_shared_kv_bindings_;
  if (bindings.empty()) return false;

  const auto& key_template = decoder.inputs.past_key_names;
  const auto placeholder = key_template.find("%d");
  if (placeholder == std::string::npos) return false;
  const auto prefix = key_template.substr(0, placeholder);
  const auto suffix = key_template.substr(placeholder + 2);
  size_t cache_count = 0;
  ONNXTensorElementDataType cache_type = ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED;
  for (const auto& name : model.session_info_.GetInputNames()) {
    if (!name.starts_with(prefix) || !name.ends_with(suffix) || name.size() <= prefix.size() + suffix.size()) continue;
    const auto digits = std::string_view{name}.substr(prefix.size(), name.size() - prefix.size() - suffix.size());
    int layer = -1;
    const auto parsed = std::from_chars(digits.data(), digits.data() + digits.size(), layer);
    if (parsed.ec != std::errc{} || parsed.ptr != digits.data() + digits.size() || layer < 0 ||
        ComposeKeyValueName(key_template, layer) != name) return false;
    const std::string inputs[] = {name, ComposeKeyValueName(decoder.inputs.past_value_names, layer)};
    const std::string outputs[] = {ComposeKeyValueName(decoder.outputs.present_key_names, layer),
                                   ComposeKeyValueName(decoder.outputs.present_value_names, layer)};
    std::vector<int64_t> key_shape;
    for (size_t i = 0; i < 2; ++i) {
      if (!model.session_info_.HasInput(inputs[i]) || !model.session_info_.HasOutput(outputs[i]) ||
          std::find(bindings.begin(), bindings.end(), std::pair{inputs[i], outputs[i]}) == bindings.end()) return false;
      const auto input_shape = model.session_info_.GetInputShape(inputs[i]);
      const auto output_shape = model.session_info_.GetOutputShape(outputs[i]);
      const auto type = model.session_info_.GetInputDataType(inputs[i]);
      if (input_shape.size() != 4 || output_shape.size() != 4 || input_shape[2] >= 0 || output_shape[2] >= 0 ||
          (type != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16 && type != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) ||
          model.session_info_.GetOutputDataType(outputs[i]) != type) return false;
      if (cache_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED) cache_type = type;
      if (type != cache_type) return false;  // One typed migration session serves every cache.
      for (size_t axis : {size_t{0}, size_t{1}, size_t{3}}) {
        int64_t resolved = params.search.batch_size;
        if (axis != 0) {
          resolved = input_shape[axis] > 0 ? input_shape[axis] : axis == 1 ? decoder.num_key_value_heads
                                                                           : decoder.head_size;
        }
        if (resolved <= 0 || (input_shape[axis] >= 0 && input_shape[axis] != resolved) ||
            (output_shape[axis] >= 0 && output_shape[axis] != resolved)) return false;
        if (i == 1 && input_shape[axis] != key_shape[axis]) return false;
      }
      if (i == 0) key_shape = input_shape;
      ++cache_count;
    }
  }
  return cache_count == bindings.size();
}

bool SharedCacheGrowth::IsSupported(const State& state, std::span<const std::string> input_names,
                                    std::span<const std::string> output_names) {
  if (Ort::api->GetModelEditorApi() == nullptr) return false;
  const auto& model = state.model_;
  if (model.p_device_kvcache_->GetType() != DeviceType::WEBGPU ||
      !model.config_->model.decoder.pipeline.empty() ||
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
