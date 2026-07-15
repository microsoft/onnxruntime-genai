// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "../generators.h"
#include "nemotron_parse.h"
#include "input_ids.h"
#include "kv_cache.h"
#include "logits.h"

namespace Generators {
namespace {

constexpr const char* kNvProfileMinShapes =
    "ep.nvtensorrtrtxexecutionprovider.nv_profile_min_shapes";
constexpr const char* kNvProfileOptShapes =
    "ep.nvtensorrtrtxexecutionprovider.nv_profile_opt_shapes";
constexpr const char* kNvProfileMaxShapes =
    "ep.nvtensorrtrtxexecutionprovider.nv_profile_max_shapes";

void AppendProfileShape(std::ostringstream& profile, bool& first,
                        const std::string& name,
                        std::initializer_list<int64_t> dimensions) {
  if (!first) {
    profile << ',';
  }
  first = false;
  profile << name << ':';
  bool first_dimension = true;
  for (int64_t dimension : dimensions) {
    if (!first_dimension) {
      profile << 'x';
    }
    first_dimension = false;
    profile << dimension;
  }
}

void SetFixedNvTensorRtRtxProfile(OrtSessionOptions& options,
                                  const std::string& profile) {
  options.AddConfigEntry(kNvProfileMinShapes, profile.c_str());
  options.AddConfigEntry(kNvProfileOptShapes, profile.c_str());
  options.AddConfigEntry(kNvProfileMaxShapes, profile.c_str());
}

DeviceInterface& DeviceFor(const OrtValue& value, DeviceInterface& model_device) {
  const bool on_cpu =
      value.GetTensorMemoryInfo().GetDeviceType() == OrtMemoryInfoDeviceType_CPU;
  return on_cpu ? *GetDeviceInterface(DeviceType::CPU) : model_device;
}

std::string MakePrefillProfile(const Config& config) {
  const auto& decoder = config.model.decoder;
  std::ostringstream profile;
  bool first = true;
  AppendProfileShape(profile, first, decoder.inputs.input_ids,
                     {1, decoder.prefill_sequence_length});
  AppendProfileShape(profile, first, decoder.inputs.attention_mask,
                     {1, decoder.prefill_sequence_length});
  AppendProfileShape(profile, first, decoder.inputs.encoder_hidden_states,
                     {1, config.model.vision.num_visual_tokens,
                      decoder.hidden_size});
  return profile.str();
}

std::string MakeDecodeProfile(const Config& config) {
  const auto& decoder = config.model.decoder;
  std::ostringstream profile;
  bool first = true;
  AppendProfileShape(profile, first, decoder.inputs.input_ids, {1, 1});
  AppendProfileShape(profile, first, decoder.inputs.attention_mask,
                     {1, config.model.context_length});

  for (int layer = 0; layer < decoder.num_hidden_layers; ++layer) {
    AppendProfileShape(profile, first,
                       ComposeKeyValueName(decoder.inputs.past_key_names, layer),
                       {1, decoder.num_key_value_heads,
                        config.model.context_length, decoder.head_size});
    AppendProfileShape(profile, first,
                       ComposeKeyValueName(decoder.inputs.past_value_names, layer),
                       {1, decoder.num_key_value_heads,
                        config.model.context_length, decoder.head_size});
    AppendProfileShape(profile, first,
                       ComposeKeyValueName(decoder.inputs.cross_past_key_names, layer),
                       {1, decoder.num_key_value_heads,
                        config.model.vision.num_visual_tokens,
                        decoder.head_size});
    AppendProfileShape(profile, first,
                       ComposeKeyValueName(decoder.inputs.cross_past_value_names, layer),
                       {1, decoder.num_key_value_heads,
                        config.model.vision.num_visual_tokens,
                        decoder.head_size});
  }

  AppendProfileShape(profile, first, decoder.inputs.cache_write_indices, {1});
  return profile.str();
}

struct ContextCaches {
  std::vector<std::unique_ptr<OrtValue>> self;
  std::vector<std::unique_ptr<OrtValue>> cross;
};

void ValidateTensorType(ONNXTensorElementDataType actual,
                        ONNXTensorElementDataType expected,
                        const std::string& name) {
  if (actual != expected) {
    throw std::runtime_error(name + " has type " + TypeToString(actual) +
                             ", expected " + TypeToString(expected));
  }
}

class DynamicAttentionMask {
 public:
  DynamicAttentionMask(State& state, const SessionInfo& session_info)
      : state_{state},
        name_{state.model_.config_->model.decoder.inputs.attention_mask},
        type_{session_info.GetInputDataType(name_)},
        value_{state.model_.p_device_inputs_, type_} {
    if (type_ != Ort::TypeToTensorType<int32_t> &&
        type_ != Ort::TypeToTensorType<int64_t>) {
      throw std::runtime_error("Nemotron Parse attention mask must be int32 or int64");
    }
  }

  void Add() {
    input_index_ = state_.inputs_.size();
    state_.input_names_.push_back(name_.c_str());
    state_.inputs_.push_back(nullptr);
  }

  void Update(size_t sequence_length) {
    const std::array<int64_t, 2> shape{
        state_.params_->search.batch_size,
        static_cast<int64_t>(sequence_length)};
    value_.CreateTensor(shape);

    auto cpu_value = OrtValue::CreateTensor(
        state_.model_.allocator_cpu_, shape, type_);
    if (type_ == Ort::TypeToTensorType<int32_t>) {
      std::fill_n(cpu_value->GetTensorMutableData<int32_t>(),
                  cpu_value->GetTensorTypeAndShapeInfo()->GetElementCount(), 1);
    } else {
      std::fill_n(cpu_value->GetTensorMutableData<int64_t>(),
                  cpu_value->GetTensorTypeAndShapeInfo()->GetElementCount(), 1);
    }

    ByteWrapTensor(*state_.model_.p_device_inputs_, *value_.GetOrtTensor())
        .CopyFrom(ByteWrapTensor(*GetDeviceInterface(DeviceType::CPU), *cpu_value));
    state_.inputs_[input_index_] = value_.GetOrtTensor();
  }

 private:
  State& state_;
  std::string name_;
  ONNXTensorElementDataType type_;
  Tensor value_;
  size_t input_index_{~0U};
};

class StaticAttentionMask {
 public:
  StaticAttentionMask(State& state, const SessionInfo& session_info)
      : state_{state},
        name_{state.model_.config_->model.decoder.inputs.attention_mask},
        type_{session_info.GetInputDataType(name_)},
        value_{state.model_.p_device_inputs_, type_} {
    if (type_ != Ort::TypeToTensorType<int32_t> &&
        type_ != Ort::TypeToTensorType<int64_t>) {
      throw std::runtime_error("Nemotron Parse attention mask must be int32 or int64");
    }

    auto graph_shape = session_info.GetInputShape(name_);
    if (graph_shape.size() != 2 || graph_shape[1] <= 0) {
      throw std::runtime_error(
          "Nemotron Parse decode attention mask must have a static sequence dimension");
    }
    cache_sequence_length_ = static_cast<int>(graph_shape[1]);

    const std::array<int64_t, 2> shape{
        state_.params_->BatchBeamSize(), cache_sequence_length_};
    value_.CreateTensor(shape, true);
    value_.GetByteSpan().Zero();
  }

  void Add() {
    state_.input_names_.push_back(name_.c_str());
    state_.inputs_.push_back(value_.GetOrtTensor());
  }

  void Initialize(size_t prompt_length) {
    Activate(prompt_length, prompt_length);
  }

  void Update(size_t total_length, size_t new_length) {
    Activate(total_length, new_length);
  }

 private:
  void Activate(size_t total_length, size_t new_length) {
    if (new_length > total_length || total_length > static_cast<size_t>(cache_sequence_length_)) {
      throw std::runtime_error("Nemotron Parse attention mask exceeds the fixed cache capacity");
    }

    if (state_.model_.p_device_inputs_->UpdateAttentionMask(
            nullptr, value_.GetMutableRawData(), state_.params_->BatchBeamSize(),
            static_cast<int>(new_length), static_cast<int>(total_length),
            cache_sequence_length_, true, type_)) {
      return;
    }

    auto bytes = value_.GetByteSpan();
    auto cpu = bytes.CopyDeviceToCpu();
    const size_t begin = total_length - new_length;
    if (type_ == Ort::TypeToTensorType<int32_t>) {
      auto* data = reinterpret_cast<int32_t*>(cpu.data());
      std::fill(data + begin, data + total_length, 1);
    } else {
      auto* data = reinterpret_cast<int64_t*>(cpu.data());
      std::fill(data + begin, data + total_length, 1);
    }
    bytes.CopyCpuToDevice();
  }

  State& state_;
  std::string name_;
  ONNXTensorElementDataType type_;
  int cache_sequence_length_;
  Tensor value_;
};

class CacheWriteIndices {
 public:
  CacheWriteIndices(State& state, const SessionInfo& session_info)
      : state_{state},
        name_{state.model_.config_->model.decoder.inputs.cache_write_indices},
        type_{session_info.GetInputDataType(name_)},
        value_{state.model_.p_device_inputs_, type_} {
    if (type_ != Ort::TypeToTensorType<int32_t> &&
        type_ != Ort::TypeToTensorType<int64_t>) {
      throw std::runtime_error("Nemotron Parse cache_write_indices must be int32 or int64");
    }
    const std::array<int64_t, 1> shape{state_.params_->BatchBeamSize()};
    value_.CreateTensor(shape, true);
  }

  void Add() {
    state_.input_names_.push_back(name_.c_str());
    state_.inputs_.push_back(value_.GetOrtTensor());
  }

  void Update(size_t index) {
    if (type_ == Ort::TypeToTensorType<int32_t>) {
      auto values = value_.GetDeviceSpan<int32_t>();
      auto cpu_values = values.CpuSpan();
      std::fill(cpu_values.begin(), cpu_values.end(), static_cast<int32_t>(index));
      values.CopyCpuToDevice();
    } else {
      auto values = value_.GetDeviceSpan<int64_t>();
      auto cpu_values = values.CpuSpan();
      std::fill(cpu_values.begin(), cpu_values.end(), static_cast<int64_t>(index));
      values.CopyCpuToDevice();
    }
  }

 private:
  State& state_;
  std::string name_;
  ONNXTensorElementDataType type_;
  Tensor value_;
};

class TensorScatterKeyValueCache {
 public:
  TensorScatterKeyValueCache(State& state, const SessionInfo& session_info)
      : state_{state},
        layer_count_{state.model_.config_->model.decoder.num_hidden_layers} {
    const auto& config = state_.model_.config_->model.decoder;
    input_names_.reserve(layer_count_ * 2);
    output_names_.reserve(layer_count_ * 2);
    values_.reserve(layer_count_ * 2);

    for (int layer = 0; layer < layer_count_; ++layer) {
      input_names_.push_back(ComposeKeyValueName(config.inputs.past_key_names, layer));
      input_names_.push_back(ComposeKeyValueName(config.inputs.past_value_names, layer));
      output_names_.push_back(ComposeKeyValueName(config.outputs.present_key_names, layer));
      output_names_.push_back(ComposeKeyValueName(config.outputs.present_value_names, layer));
    }

    for (size_t i = 0; i < input_names_.size(); ++i) {
      const auto& input_name = input_names_[i];
      const auto& output_name = output_names_[i];
      if (!session_info.HasInput(input_name) || !session_info.HasOutput(output_name)) {
        throw std::runtime_error("Nemotron Parse decode graph is missing cache pair " +
                                 input_name + " -> " + output_name);
      }

      auto shape = session_info.GetInputShape(input_name);
      auto output_shape = session_info.GetOutputShape(output_name);
      if (shape.size() != 4 || output_shape.size() != 4) {
        throw std::runtime_error("Nemotron Parse self KV cache must be rank 4");
      }
      shape[0] = state_.params_->BatchBeamSize();
      if (shape[1] <= 0 || shape[2] <= 0 || shape[3] <= 0) {
        throw std::runtime_error("Nemotron Parse self KV cache has an invalid fixed shape");
      }
      if (cache_sequence_length_ == 0) {
        cache_sequence_length_ = static_cast<int>(shape[2]);
      } else if (shape[2] != cache_sequence_length_) {
        throw std::runtime_error(
            "Nemotron Parse self KV caches have inconsistent sequence dimensions");
      }
      for (size_t axis = 1; axis < shape.size(); ++axis) {
        if (output_shape[axis] > 0 && output_shape[axis] != shape[axis]) {
          throw std::runtime_error("Nemotron Parse past/present KV cache shapes differ");
        }
      }

      const auto type = session_info.GetInputDataType(input_name);
      ValidateTensorType(session_info.GetOutputDataType(output_name), type, output_name);
      values_.push_back(OrtValue::CreateTensor(
          state_.model_.p_device_kvcache_->GetAllocator(), shape, type));
      ByteWrapTensor(*state_.model_.p_device_kvcache_, *values_.back()).Zero();
    }
  }

  void Add() {
    for (size_t i = 0; i < values_.size(); ++i) {
      state_.input_names_.push_back(input_names_[i].c_str());
      state_.inputs_.push_back(values_[i].get());
      state_.output_names_.push_back(output_names_[i].c_str());
      state_.outputs_.push_back(values_[i].get());
    }
  }

  void Initialize(const std::vector<std::unique_ptr<OrtValue>>& compact_values) {
    if (compact_values.size() != values_.size()) {
      throw std::runtime_error("Nemotron Parse prefill returned an unexpected self-cache count");
    }

    for (size_t i = 0; i < values_.size(); ++i) {
      auto source_info = compact_values[i]->GetTensorTypeAndShapeInfo();
      auto target_info = values_[i]->GetTensorTypeAndShapeInfo();
      auto source_shape = source_info->GetShape();
      auto target_shape = target_info->GetShape();
      if (source_shape.size() != 4 || source_shape[0] != target_shape[0] ||
          source_shape[1] != target_shape[1] || source_shape[3] != target_shape[3] ||
          source_shape[2] > target_shape[2]) {
        throw std::runtime_error("Nemotron Parse prefill self cache does not fit the decode cache");
      }
      ValidateTensorType(source_info->GetElementType(), target_info->GetElementType(),
                         input_names_[i]);

      const size_t row_bytes = static_cast<size_t>(source_shape[2] * source_shape[3]) *
                               Ort::SizeOf(source_info->GetElementType());
      const size_t source_stride = row_bytes;
      const size_t target_stride = static_cast<size_t>(target_shape[2] * target_shape[3]) *
                                   Ort::SizeOf(target_info->GetElementType());
      auto& cache_device = *state_.model_.p_device_kvcache_;
      auto source = ByteWrapTensor(DeviceFor(*compact_values[i], cache_device),
                                   *compact_values[i]);
      auto target = ByteWrapTensor(cache_device, *values_[i]);
      const size_t rows = static_cast<size_t>(source_shape[0] * source_shape[1]);
      for (size_t row = 0; row < rows; ++row) {
        target.subspan(row * target_stride, row_bytes)
            .CopyFrom(source.subspan(row * source_stride, row_bytes));
      }
    }
  }

 private:
  State& state_;
  int cache_sequence_length_{};
  int layer_count_;
  std::vector<std::string> input_names_;
  std::vector<std::string> output_names_;
  std::vector<std::unique_ptr<OrtValue>> values_;
};

class EncoderState : public State {
 public:
  EncoderState(const NemotronParseModel& model, const GeneratorParams& params)
      : State{params, model}, model_{model} {
    output_names_.push_back(
        model_.config_->model.vision.outputs.image_features.c_str());
    outputs_.push_back(nullptr);
  }

  void SetExtraInputs(const std::vector<ExtraInput>& extra_inputs) override {
    const auto& graph_name = model_.config_->model.vision.inputs.pixel_values;
    for (const auto& input : extra_inputs) {
      if (input.name == graph_name || input.name == Config::Defaults::PixelValuesName) {
        pixel_values_ = input.tensor;
        input_names_.push_back(graph_name.c_str());
        inputs_.push_back(pixel_values_->ort_tensor_.get());
        return;
      }
    }
    throw std::runtime_error("Nemotron Parse requires a pixel_values input");
  }

  std::unique_ptr<OrtValue> RunEncoder() {
    if (!pixel_values_) {
      throw std::runtime_error("Nemotron Parse pixel_values were not set");
    }
    if (model_.config_->model.vision.run_options.has_value()) {
      State::SetRunOptions(*model_.config_->model.vision.run_options);
    }
    State::Run(*model_.encoder_session_);
    std::unique_ptr<OrtValue> result{outputs_[0]};
    outputs_[0] = nullptr;
    if (!result) {
      throw std::runtime_error("Nemotron Parse encoder did not return encoder_hidden_states");
    }
    return result;
  }

  DeviceSpan<float> Run(int, DeviceSpan<int32_t>&, DeviceSpan<int32_t>) override {
    throw std::runtime_error("Use EncoderState::RunEncoder for Nemotron Parse");
  }

 private:
  const NemotronParseModel& model_;
  std::shared_ptr<Tensor> pixel_values_;
};

class PrefillState : public State {
 public:
  struct Result {
    DeviceSpan<float> logits;
    ContextCaches caches;
  };

  PrefillState(const NemotronParseModel& model, const GeneratorParams& params)
      : State{params, model},
        model_{model},
        input_ids_{*this},
        attention_mask_{*this, model_.prefill_session_info_},
        logits_{*this} {
    input_ids_.Add();
    attention_mask_.Add();
    encoder_hidden_states_input_index_ = inputs_.size();
    input_names_.push_back(
        model_.config_->model.decoder.inputs.encoder_hidden_states.c_str());
    inputs_.push_back(nullptr);
    logits_.Add();

    const auto& decoder = model_.config_->model.decoder;
    const int layer_count = decoder.num_hidden_layers;
    self_output_indices_.reserve(layer_count * 2);
    cross_output_indices_.reserve(layer_count * 2);
    cache_output_names_.reserve(layer_count * 4);
    for (int layer = 0; layer < layer_count; ++layer) {
      AddCacheOutput(ComposeKeyValueName(decoder.outputs.present_key_names, layer),
                     self_output_indices_);
      AddCacheOutput(ComposeKeyValueName(decoder.outputs.present_value_names, layer),
                     self_output_indices_);
      AddCacheOutput(ComposeKeyValueName(decoder.outputs.cross_present_key_names, layer),
                     cross_output_indices_);
      AddCacheOutput(ComposeKeyValueName(decoder.outputs.cross_present_value_names, layer),
                     cross_output_indices_);
    }
  }

  Result RunPrefill(DeviceSpan<int32_t>& tokens, OrtValue& encoder_hidden_states) {
    const size_t sequence_length = tokens.size() / params_->search.batch_size;
    input_ids_.Update(tokens);
    attention_mask_.Update(sequence_length);
    inputs_[encoder_hidden_states_input_index_] = &encoder_hidden_states;
    logits_.Update(tokens, sequence_length);
    if (model_.config_->model.decoder.run_options.has_value()) {
      State::SetRunOptions(*model_.config_->model.decoder.run_options);
    }
    State::Run(*model_.prefill_session_);
    inputs_[encoder_hidden_states_input_index_] = nullptr;

    Result result{logits_.Get(), {}};
    result.caches.self = TakeOutputs(self_output_indices_);
    result.caches.cross = TakeOutputs(cross_output_indices_);
    return result;
  }

  DeviceSpan<float> Run(int, DeviceSpan<int32_t>&, DeviceSpan<int32_t>) override {
    throw std::runtime_error("Use PrefillState::RunPrefill for Nemotron Parse");
  }

 private:
  void AddCacheOutput(std::string name, std::vector<size_t>& indices) {
    cache_output_names_.push_back(std::move(name));
    indices.push_back(outputs_.size());
    output_names_.push_back(cache_output_names_.back().c_str());
    outputs_.push_back(nullptr);
  }

  std::vector<std::unique_ptr<OrtValue>> TakeOutputs(
      const std::vector<size_t>& indices) {
    std::vector<std::unique_ptr<OrtValue>> values;
    values.reserve(indices.size());
    for (size_t index : indices) {
      if (!outputs_[index]) {
        throw std::runtime_error("Nemotron Parse prefill did not return a required cache output");
      }
      values.emplace_back(outputs_[index]);
      outputs_[index] = nullptr;
    }
    return values;
  }

  const NemotronParseModel& model_;
  DefaultInputIDs input_ids_;
  DynamicAttentionMask attention_mask_;
  Logits logits_;
  size_t encoder_hidden_states_input_index_{~0U};
  std::vector<std::string> cache_output_names_;
  std::vector<size_t> self_output_indices_;
  std::vector<size_t> cross_output_indices_;
};

class DecodeState : public State {
 public:
  DecodeState(const NemotronParseModel& model, const GeneratorParams& params)
      : State{params, model},
        model_{model},
        input_ids_{*this},
        attention_mask_{*this, model_.decoder_session_info_},
        write_indices_{*this, model_.decoder_session_info_},
        self_cache_{*this, model_.decoder_session_info_},
        logits_{*this} {
    input_ids_.Add();
    attention_mask_.Add();
    write_indices_.Add();

    const auto& decoder = model_.config_->model.decoder;
    if (model_.decoder_session_info_.HasInput(decoder.inputs.encoder_hidden_states)) {
      encoder_hidden_states_input_index_ = inputs_.size();
      input_names_.push_back(decoder.inputs.encoder_hidden_states.c_str());
      inputs_.push_back(nullptr);
    }

    cross_input_names_.reserve(decoder.num_hidden_layers * 2);
    cross_input_indices_.reserve(decoder.num_hidden_layers * 2);
    for (int layer = 0; layer < decoder.num_hidden_layers; ++layer) {
      AddCrossInput(ComposeKeyValueName(decoder.inputs.cross_past_key_names, layer));
      AddCrossInput(ComposeKeyValueName(decoder.inputs.cross_past_value_names, layer));
    }

    self_cache_.Add();
    logits_.Add();
  }

  void Initialize(size_t prompt_length,
                  std::unique_ptr<OrtValue> encoder_hidden_states,
                  ContextCaches caches) {
    self_cache_.Initialize(caches.self);
    if (caches.cross.size() != cross_input_indices_.size()) {
      throw std::runtime_error("Nemotron Parse prefill returned an unexpected cross-cache count");
    }

    std::vector<DeviceSpan<uint8_t>> cross_cache_sources;
    std::vector<DeviceSpan<uint8_t>> cross_cache_targets;
    cross_cache_sources.reserve(caches.cross.size());
    cross_cache_targets.reserve(caches.cross.size());
    cross_cache_.clear();
    cross_cache_.reserve(caches.cross.size());
    auto& cache_device = *model_.p_device_kvcache_;
    for (size_t i = 0; i < caches.cross.size(); ++i) {
      auto source_info = caches.cross[i]->GetTensorTypeAndShapeInfo();
      const auto expected_type = model_.decoder_session_info_.GetInputDataType(
          cross_input_names_[i]);
      ValidateTensorType(source_info->GetElementType(), expected_type,
                         cross_input_names_[i]);

      const auto source_shape = source_info->GetShape();
      const auto input_shape = model_.decoder_session_info_.GetInputShape(
          cross_input_names_[i]);
      if (source_shape.size() != input_shape.size()) {
        throw std::runtime_error("Nemotron Parse cross cache has an unexpected rank for " +
                                 cross_input_names_[i]);
      }
      for (size_t axis = 0; axis < input_shape.size(); ++axis) {
        if (input_shape[axis] > 0 && input_shape[axis] != source_shape[axis]) {
          throw std::runtime_error("Nemotron Parse cross cache has an unexpected shape for " +
                                   cross_input_names_[i]);
        }
      }

      cross_cache_sources.push_back(ByteWrapTensor(
          DeviceFor(*caches.cross[i], cache_device), *caches.cross[i]));
      cross_cache_.push_back(OrtValue::CreateTensor(
          cache_device.GetAllocator(), source_shape, expected_type));
      cross_cache_targets.push_back(
          ByteWrapTensor(cache_device, *cross_cache_.back()));
      cross_cache_targets.back().CopyFrom(cross_cache_sources.back());
      inputs_[cross_input_indices_[i]] = cross_cache_[i].get();
    }

    // Prefill outputs may be pageable CPU tensors. Copy them once into persistent
    // KV-cache-device allocations so ORT does not stage every cross cache each token.
    cache_device.Synchronize();

    if (encoder_hidden_states_input_index_ != ~0U) {
      encoder_hidden_states_ = std::move(encoder_hidden_states);
      inputs_[encoder_hidden_states_input_index_] = encoder_hidden_states_.get();
    }
    attention_mask_.Initialize(prompt_length);
  }

  DeviceSpan<float> Run(int total_length, DeviceSpan<int32_t>& next_tokens,
                        DeviceSpan<int32_t>) override {
    const size_t new_length = next_tokens.size() / params_->BatchBeamSize();
    if (new_length != 1) {
      throw std::runtime_error(
          "Nemotron Parse TensorScatter decode accepts exactly one token per step");
    }
    if (total_length <= 0 ||
        total_length > model_.config_->model.context_length) {
      throw std::runtime_error("Nemotron Parse decode length exceeds the cache capacity");
    }

    input_ids_.Update(next_tokens);
    attention_mask_.Update(total_length, new_length);
    write_indices_.Update(static_cast<size_t>(total_length) - new_length);
    logits_.Update(next_tokens, new_length);
    if (model_.config_->model.decoder.run_options.has_value()) {
      State::SetRunOptions(*model_.config_->model.decoder.run_options);
    }
    State::Run(*model_.decoder_session_, params_->use_graph_capture);
    return logits_.Get();
  }

 private:
  void AddCrossInput(std::string name) {
    if (!model_.decoder_session_info_.HasInput(name)) {
      throw std::runtime_error("Nemotron Parse decode graph is missing " + name);
    }
    cross_input_names_.push_back(std::move(name));
    cross_input_indices_.push_back(inputs_.size());
    input_names_.push_back(cross_input_names_.back().c_str());
    inputs_.push_back(nullptr);
  }

  const NemotronParseModel& model_;
  DefaultInputIDs input_ids_;
  StaticAttentionMask attention_mask_;
  CacheWriteIndices write_indices_;
  TensorScatterKeyValueCache self_cache_;
  Logits logits_;

  size_t encoder_hidden_states_input_index_{~0U};
  std::unique_ptr<OrtValue> encoder_hidden_states_;
  std::vector<std::string> cross_input_names_;
  std::vector<size_t> cross_input_indices_;
  std::vector<std::unique_ptr<OrtValue>> cross_cache_;
};

class NemotronParseState : public State {
 public:
  NemotronParseState(const NemotronParseModel& model,
                     const GeneratorParams& params)
      : State{params, model},
        model_{model},
        encoder_state_{std::make_unique<EncoderState>(model, params)},
        prefill_state_{std::make_unique<PrefillState>(model, params)},
        decode_state_{model, params} {
    if (params_->search.batch_size != 1 || params_->search.num_beams != 1) {
      throw std::runtime_error(
          "Nemotron Parse native TensorScatter currently supports batch_size=1 and num_beams=1");
    }
    if (params_->search.max_length > model_.config_->model.context_length) {
      throw std::runtime_error(
          "Nemotron Parse max_length exceeds the TensorScatter cache capacity");
    }
  }

  void SetExtraInputs(const std::vector<ExtraInput>& extra_inputs) override {
    if (!encoder_state_) {
      throw std::runtime_error("Nemotron Parse inputs cannot be changed after prefill");
    }
    encoder_state_->SetExtraInputs(extra_inputs);
  }

  DeviceSpan<float> Run(int total_length, DeviceSpan<int32_t>& next_tokens,
                        DeviceSpan<int32_t> next_indices) override {
    if (!context_complete_) {
      if (total_length <= 0 ||
          total_length > model_.config_->model.context_length) {
        throw std::runtime_error("Nemotron Parse prompt exceeds the cache capacity");
      }
      if (model_.p_device_->GetType() == DeviceType::NvTensorRtRtx &&
          total_length != model_.config_->model.decoder.prefill_sequence_length) {
        throw std::runtime_error(
            "Nemotron Parse TRT-RTX prompt length must match prefill_sequence_length");
      }
      auto encoder_hidden_states = encoder_state_->RunEncoder();
      auto prefill = prefill_state_->RunPrefill(next_tokens, *encoder_hidden_states);
      decode_state_.Initialize(static_cast<size_t>(total_length),
                               std::move(encoder_hidden_states),
                               std::move(prefill.caches));
      encoder_state_.reset();
      context_complete_ = true;
      return prefill.logits;
    }
    // Prefill logits have been consumed by sampling before the next Run call.
    prefill_state_.reset();
    return decode_state_.Run(total_length, next_tokens, next_indices);
  }

  void RewindTo(size_t) override {
    throw std::runtime_error("Nemotron Parse native TensorScatter does not support rewind");
  }

 private:
  const NemotronParseModel& model_;
  std::unique_ptr<EncoderState> encoder_state_;
  std::unique_ptr<PrefillState> prefill_state_;
  DecodeState decode_state_;
  bool context_complete_{};
};

}  // namespace

NemotronParseModel::NemotronParseModel(std::unique_ptr<Config> config,
                                       OrtEnv& ort_env)
    : Model{std::move(config)} {
  const auto& decoder = config_->model.decoder;
  if (decoder.cache_update_mode != "tensor_scatter") {
    throw std::runtime_error(
        "Nemotron Parse native OGA execution requires cache_update_mode=tensor_scatter");
  }
  if (config_->model.vision.filename.empty() || decoder.filename.empty() ||
      decoder.prefill_filename.empty() || config_->model.context_length <= 0 ||
      decoder.prefill_sequence_length <= 0 ||
      decoder.prefill_sequence_length > config_->model.context_length ||
      decoder.hidden_size <= 0 || decoder.num_hidden_layers <= 0 ||
      decoder.num_key_value_heads <= 0 || decoder.head_size <= 0 ||
      decoder.inputs.cache_write_indices.empty() ||
      decoder.inputs.past_key_names.empty() ||
      decoder.inputs.past_value_names.empty() ||
      decoder.inputs.cross_past_key_names.empty() ||
      decoder.inputs.cross_past_value_names.empty() ||
      decoder.outputs.present_key_names.empty() ||
      decoder.outputs.present_value_names.empty() ||
      decoder.outputs.cross_present_key_names.empty() ||
      decoder.outputs.cross_present_value_names.empty() ||
      config_->model.vision.num_visual_tokens <= 0) {
    throw std::runtime_error(
        "Nemotron Parse TensorScatter config is missing prefill or cache metadata");
  }

  config_->AddMapping(std::string(Config::Defaults::PixelValuesName),
                      config_->model.vision.inputs.pixel_values);

  encoder_session_options_ = OrtSessionOptions::Create();
  const auto& encoder_session_config =
      config_->model.vision.session_options.has_value()
          ? *config_->model.vision.session_options
          : decoder.session_options;
  CreateSessionOptionsFromConfig(encoder_session_config,
                                 *encoder_session_options_, true,
                                 /*disable_graph_capture=*/true);
  prefill_session_options_ = OrtSessionOptions::Create();
  CreateSessionOptionsFromConfig(decoder.session_options,
                                 *prefill_session_options_, true,
                                 /*disable_graph_capture=*/true);

  if (p_device_->GetType() == DeviceType::NvTensorRtRtx) {
    SetFixedNvTensorRtRtxProfile(*prefill_session_options_,
                                 MakePrefillProfile(*config_));
    SetFixedNvTensorRtRtxProfile(*session_options_,
                                 MakeDecodeProfile(*config_));
  }

  encoder_session_ = CreateSession(ort_env, config_->model.vision.filename,
                                   encoder_session_options_.get());
  prefill_session_ = CreateSession(ort_env, decoder.prefill_filename,
                                   prefill_session_options_.get());
  decoder_session_ = CreateSession(ort_env, decoder.filename,
                                   session_options_.get());

  encoder_session_info_.Add(*encoder_session_);
  prefill_session_info_.Add(*prefill_session_);
  decoder_session_info_.Add(*decoder_session_);

  const auto pixel_values_shape = encoder_session_info_.GetInputShape(
      config_->model.vision.inputs.pixel_values);
  if (pixel_values_shape.size() != 4 || pixel_values_shape[0] != 1 ||
      pixel_values_shape[1] != 3 || pixel_values_shape[2] <= 0 ||
      pixel_values_shape[3] <= 0) {
    throw std::runtime_error(
        "Nemotron Parse encoder pixel_values must have static shape [1, 3, H, W]");
  }

  const auto attention_mask_shape = decoder_session_info_.GetInputShape(
      decoder.inputs.attention_mask);
  if (attention_mask_shape.size() != 2 || attention_mask_shape[1] <= 0 ||
      attention_mask_shape[1] != config_->model.context_length) {
    throw std::runtime_error(
        "Nemotron Parse context_length must match the static decode attention-mask shape");
  }

  // Generic input/logits helpers must resolve the cached decoder's static shapes.
  session_info_.Add(*decoder_session_);
}

std::unique_ptr<State> NemotronParseModel::CreateState(
    DeviceSpan<int32_t>, const GeneratorParams& params) const {
  return std::make_unique<NemotronParseState>(*this, params);
}

}  // namespace Generators
