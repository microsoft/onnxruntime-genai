// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "../generators.h"
#include "nemotron_parse.h"
#include "input_ids.h"
#include "kv_cache.h"
#include "logits.h"
#include "position_inputs.h"

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

void SetFixedProfile(OrtSessionOptions& session_options,
                     const std::string& profile) {
  session_options.AddConfigEntry(kNvProfileMinShapes, profile.c_str());
  session_options.AddConfigEntry(kNvProfileOptShapes, profile.c_str());
  session_options.AddConfigEntry(kNvProfileMaxShapes, profile.c_str());
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
    AppendProfileShape(
        profile, first,
        ComposeKeyValueName(decoder.inputs.cross_past_key_names, layer),
        {1, decoder.num_key_value_heads,
         config.model.vision.num_visual_tokens, decoder.head_size});
    AppendProfileShape(
        profile, first,
        ComposeKeyValueName(decoder.inputs.cross_past_value_names, layer),
        {1, decoder.num_key_value_heads,
         config.model.vision.num_visual_tokens, decoder.head_size});
  }

  AppendProfileShape(profile, first, decoder.inputs.cache_write_indices, {1});
  return profile.str();
}

DeviceInterface& DeviceFor(const OrtValue& value, DeviceInterface& model_device) {
  const bool on_cpu =
      value.GetTensorMemoryInfo().GetDeviceType() == OrtMemoryInfoDeviceType_CPU;
  return on_cpu ? *GetDeviceInterface(DeviceType::CPU) : model_device;
}

bool HasName(const std::vector<std::string>& input_names,
             const std::string& name) {
  return std::find(input_names.begin(), input_names.end(), name) !=
         input_names.end();
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

  PrefillState(const NemotronParseModel& model,
               DeviceSpan<int32_t> sequence_lengths,
               const GeneratorParams& params)
      : State{params, model},
        model_{model},
        input_ids_{*this},
        attention_mask_{
            model, *this, sequence_lengths,
            model.config_->model.decoder.inputs.attention_mask,
            {AttentionMaskMode::Dynamic}},
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
    attention_mask_.Update(tokens, static_cast<int>(sequence_length),
                           static_cast<int>(sequence_length));
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
  DefaultPositionInputs attention_mask_;
  Logits logits_;
  size_t encoder_hidden_states_input_index_{~0U};
  std::vector<std::string> cache_output_names_;
  std::vector<size_t> self_output_indices_;
  std::vector<size_t> cross_output_indices_;
};

class DecodeState : public State {
 public:
  DecodeState(const NemotronParseModel& model,
              DeviceSpan<int32_t> sequence_lengths,
              const GeneratorParams& params)
      : State{params, model},
        model_{model},
        input_ids_{*this},
        attention_mask_{
            model, *this, sequence_lengths,
            model.config_->model.decoder.inputs.attention_mask,
            {AttentionMaskMode::Static, model.config_->model.context_length}},
        self_cache_{*this},
        logits_{*this} {
    input_ids_.Add();
    attention_mask_.Add();

    const auto& decoder = model_.config_->model.decoder;
    // session_info_ is the union of all three graphs. Use the decoder's own
    // names for phase membership so prefill-only inputs are not fed to decode.
    const auto decoder_input_names = model_.decoder_session_->GetInputNames();
    if (HasName(decoder_input_names, decoder.inputs.encoder_hidden_states)) {
      encoder_hidden_states_input_index_ = inputs_.size();
      input_names_.push_back(decoder.inputs.encoder_hidden_states.c_str());
      inputs_.push_back(nullptr);
    }

    cross_input_names_.reserve(decoder.num_hidden_layers * 2);
    cross_input_indices_.reserve(decoder.num_hidden_layers * 2);
    for (int layer = 0; layer < decoder.num_hidden_layers; ++layer) {
      AddCrossInput(
          ComposeKeyValueName(decoder.inputs.cross_past_key_names, layer),
          decoder_input_names);
      AddCrossInput(
          ComposeKeyValueName(decoder.inputs.cross_past_value_names, layer),
          decoder_input_names);
    }

    self_cache_.Add();
    logits_.Add();
  }

  void Initialize(size_t prompt_length,
                  DeviceSpan<int32_t> prompt_tokens,
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
      const auto expected_type = model_.session_info_.GetInputDataType(
          cross_input_names_[i]);
      ValidateTensorType(source_info->GetElementType(), expected_type,
                         cross_input_names_[i]);

      const auto source_shape = source_info->GetShape();
      const auto input_shape = model_.session_info_.GetInputShape(
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
    attention_mask_.Update(prompt_tokens, static_cast<int>(prompt_length),
                           static_cast<int>(prompt_length));
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
    attention_mask_.Update(next_tokens, total_length,
                           static_cast<int>(new_length));
    self_cache_.Update({}, total_length);
    logits_.Update(next_tokens, new_length);
    if (model_.config_->model.decoder.run_options.has_value()) {
      State::SetRunOptions(*model_.config_->model.decoder.run_options);
    }
    State::Run(*model_.decoder_session_, params_->use_graph_capture);
    return logits_.Get();
  }

 private:
  void AddCrossInput(std::string name,
                     const std::vector<std::string>& decoder_input_names) {
    if (!HasName(decoder_input_names, name)) {
      throw std::runtime_error("Nemotron Parse decode graph is missing " + name);
    }
    cross_input_names_.push_back(std::move(name));
    cross_input_indices_.push_back(inputs_.size());
    input_names_.push_back(cross_input_names_.back().c_str());
    inputs_.push_back(nullptr);
  }

  const NemotronParseModel& model_;
  DefaultInputIDs input_ids_;
  DefaultPositionInputs attention_mask_;
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
                     DeviceSpan<int32_t> sequence_lengths,
                     const GeneratorParams& params)
      : State{params, model},
        model_{model},
        encoder_state_{std::make_unique<EncoderState>(model, params)},
        prefill_state_{
            std::make_unique<PrefillState>(model, sequence_lengths, params)},
        decode_state_{model, sequence_lengths, params} {
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
      if (total_length !=
          model_.config_->model.decoder.prefill_sequence_length) {
        throw std::runtime_error(
            "Nemotron Parse prompt length must match prefill_sequence_length");
      }
      auto encoder_hidden_states = encoder_state_->RunEncoder();
      auto prefill = prefill_state_->RunPrefill(next_tokens, *encoder_hidden_states);
      decode_state_.Initialize(static_cast<size_t>(total_length),
                               next_tokens,
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
  if (config_->model.vision.filename.empty() || decoder.filename.empty() ||
      decoder.prefill_filename.empty() || config_->model.context_length <= 0 ||
      decoder.prefill_sequence_length <= 0 ||
      decoder.prefill_sequence_length >= config_->model.context_length ||
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
    SetFixedProfile(*prefill_session_options_, MakePrefillProfile(*config_));
    SetFixedProfile(*session_options_, MakeDecodeProfile(*config_));
  }

  encoder_session_ = CreateSession(ort_env, config_->model.vision.filename,
                                   encoder_session_options_.get());
  prefill_session_ = CreateSession(ort_env, decoder.prefill_filename,
                                   prefill_session_options_.get());
  decoder_session_ = CreateSession(ort_env, decoder.filename,
                                   session_options_.get());

  // Shared names have phase-specific shapes. Keep decode metadata for those
  // names, matching Whisper's decoder-first SessionInfo convention.
  session_info_.Add(*decoder_session_);
  session_info_.Add(*prefill_session_);
  session_info_.Add(*encoder_session_);

  const auto pixel_values_shape = session_info_.GetInputShape(
      config_->model.vision.inputs.pixel_values);
  if (pixel_values_shape.size() != 4 || pixel_values_shape[0] != 1 ||
      pixel_values_shape[1] != 3 || pixel_values_shape[2] <= 0 ||
      pixel_values_shape[3] <= 0) {
    throw std::runtime_error(
        "Nemotron Parse encoder pixel_values must have static shape [1, 3, H, W]");
  }

  const auto attention_mask_shape = session_info_.GetInputShape(
      decoder.inputs.attention_mask);
  if (attention_mask_shape.size() != 2 || attention_mask_shape[1] <= 0 ||
      attention_mask_shape[1] != config_->model.context_length) {
    throw std::runtime_error(
        "Nemotron Parse context_length must match the static decode attention-mask shape");
  }
}

std::unique_ptr<State> NemotronParseModel::CreateState(
    DeviceSpan<int32_t> sequence_lengths, const GeneratorParams& params) const {
  return std::make_unique<NemotronParseState>(*this, sequence_lengths, params);
}

}  // namespace Generators
