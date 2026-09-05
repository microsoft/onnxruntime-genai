// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "generator/generators.h"
#include "nemotron_parse.h"
#include "models/io/cross_kv_cache.h"
#include "models/io/default_position_inputs.h"
#include "models/io/input_ids.h"
#include "models/io/logits.h"
#include "models/io/tensor_scatter_kv_cache.h"

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

std::string MakeDecoderProfile(const Config& config, int sequence_length) {
  const auto& decoder = config.model.decoder;
  std::ostringstream profile;
  bool first = true;
  AppendProfileShape(profile, first, decoder.inputs.input_ids,
                     {1, sequence_length});
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

void SpecializeDecoderSession(OrtSessionOptions& session_options,
                              const Config& config,
                              int sequence_length) {
  // Fixed TRT profiles alone do not make symbolic dimensions static during
  // ORT graph optimization. Override every free dimension so TRT-RTX can
  // compile each decoder phase as a static graph.
  session_options.AddFreeDimensionOverrideByName("batch_size", 1);
  session_options.AddFreeDimensionOverrideByName(
      "encoder_sequence_length", config.model.vision.num_visual_tokens);
  session_options.AddFreeDimensionOverrideByName("sequence_length",
                                                 sequence_length);

  const auto profile = MakeDecoderProfile(config, sequence_length);
  session_options.AddConfigEntry(kNvProfileMinShapes, profile.c_str());
  session_options.AddConfigEntry(kNvProfileOptShapes, profile.c_str());
  session_options.AddConfigEntry(kNvProfileMaxShapes, profile.c_str());
}

class EncoderState : public State {
 public:
  EncoderState(const NemotronParseModel& model, const GeneratorParams& params)
      : State{params, model}, model_{model} {}

  void AddCrossCache(CrossCache& cross_cache) {
    cross_cache.AddOutputs(*this);
  }

  void SetExtraInputs(const std::vector<ExtraInput>& extra_inputs) override {
    const auto& graph_name = model_.config_->model.vision.inputs.pixel_values;
    for (const auto& input : extra_inputs) {
      if (input.name == graph_name ||
          input.name == Config::Defaults::PixelValuesName) {
        pixel_values_ = input.tensor;
        input_names_.push_back(graph_name.c_str());
        inputs_.push_back(pixel_values_->ort_tensor_.get());
        return;
      }
    }
    throw std::runtime_error("Nemotron Parse requires a pixel_values input");
  }

  void RunEncoder() {
    if (!pixel_values_) {
      throw std::runtime_error("Nemotron Parse pixel_values were not set");
    }
    if (model_.config_->model.vision.run_options.has_value()) {
      State::SetRunOptions(*model_.config_->model.vision.run_options);
    }
    State::Run(*model_.encoder_session_);
  }

  DeviceSpan<float> Run(int, DeviceSpan<int32_t>&, DeviceSpan<int32_t>) override {
    throw std::runtime_error("Use EncoderState::RunEncoder for Nemotron Parse");
  }

 private:
  const NemotronParseModel& model_;
  std::shared_ptr<Tensor> pixel_values_;
};

class DecoderState : public State {
 public:
  DecoderState(const NemotronParseModel& model,
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
    self_cache_.Add();
    logits_.Add();
  }

  void AddCrossCache(CrossCache& cross_cache) {
    cross_cache.AddInputs(*this);
  }

  DeviceSpan<float> Run(int total_length, DeviceSpan<int32_t>& next_tokens,
                        DeviceSpan<int32_t>) override {
    const size_t new_length = next_tokens.size() / params_->BatchBeamSize();
    const bool is_prompt = first_run_;
    if (is_prompt) {
      if (total_length !=
              model_.config_->model.decoder.prefill_sequence_length ||
          new_length != static_cast<size_t>(total_length)) {
        throw std::runtime_error(
            "Nemotron Parse prompt length must match prefill_sequence_length");
      }
    } else if (new_length != 1) {
      throw std::runtime_error(
          "Nemotron Parse TensorScatter decode accepts one token per step");
    }
    if (total_length <= 0 ||
        total_length > model_.config_->model.context_length) {
      throw std::runtime_error(
          "Nemotron Parse sequence length exceeds the cache capacity");
    }

    input_ids_.Update(next_tokens);
    attention_mask_.Update(next_tokens, total_length,
                           static_cast<int>(new_length));
    self_cache_.Update({}, total_length);
    logits_.Update(next_tokens, new_length);
    auto& decoder_session =
        is_prompt && model_.prefill_decoder_session_
            ? *model_.prefill_decoder_session_
            : *model_.decoder_session_;
    UpdateIoBinding(decoder_session, new_length);
    if (model_.config_->model.decoder.run_options.has_value()) {
      State::SetRunOptions(*model_.config_->model.decoder.run_options);
    }
    State::Run(decoder_session,
               params_->use_graph_capture && !is_prompt,
               static_cast<int>(new_length), 0, io_binding_.get());
    return logits_.Get();
  }

 private:
  void UpdateIoBinding(OrtSession& session, size_t sequence_length) {
    if (io_binding_ && bound_session_ == &session &&
        bound_sequence_length_ == sequence_length) {
      return;
    }

    // TRT-RTX lowers TensorScatter to an in-place layer, so each past/present
    // pair must alias the same preallocated OrtValue. Rebuild the binding only
    // when prompt-to-token decoding changes the input and logits shapes.
    io_binding_ = OrtIoBinding::Create(session);
    for (size_t i = 0; i < input_names_.size(); ++i) {
      io_binding_->BindInput(input_names_[i], *inputs_[i]);
    }
    for (size_t i = 0; i < output_names_.size(); ++i) {
      if (!outputs_[i]) {
        throw std::runtime_error(
            "Nemotron Parse requires preallocated decoder outputs");
      }
      io_binding_->BindOutput(output_names_[i], *outputs_[i]);
    }
    bound_session_ = &session;
    bound_sequence_length_ = sequence_length;
  }

  const NemotronParseModel& model_;
  DefaultInputIDs input_ids_;
  DefaultPositionInputs attention_mask_;
  TensorScatterKeyValueCache self_cache_;
  Logits logits_;
  std::unique_ptr<OrtIoBinding> io_binding_;
  OrtSession* bound_session_{};
  size_t bound_sequence_length_{};
};

class NemotronParseState : public State {
 public:
  NemotronParseState(const NemotronParseModel& model,
                     DeviceSpan<int32_t> sequence_lengths,
                     const GeneratorParams& params)
      : State{params, model},
        model_{model},
        encoder_state_{std::make_unique<EncoderState>(model, params)},
        decoder_state_{model, sequence_lengths, params} {
    if (params_->search.batch_size != 1 || params_->search.num_beams != 1) {
      throw std::runtime_error(
          "Nemotron Parse TensorScatter supports batch_size=1 and num_beams=1");
    }
    if (params_->search.max_length > model_.config_->model.context_length) {
      throw std::runtime_error(
          "Nemotron Parse max_length exceeds the TensorScatter cache capacity");
    }

    cross_cache_ = std::make_unique<CrossCache>(
        *this, model_.config_->model.vision.num_visual_tokens);
    encoder_state_->AddCrossCache(*cross_cache_);
    decoder_state_.AddCrossCache(*cross_cache_);
  }

  void SetExtraInputs(const std::vector<ExtraInput>& extra_inputs) override {
    if (!encoder_state_) {
      throw std::runtime_error(
          "Nemotron Parse inputs cannot be changed after prompt processing");
    }
    encoder_state_->SetExtraInputs(extra_inputs);
  }

  DeviceSpan<float> Run(int total_length, DeviceSpan<int32_t>& next_tokens,
                        DeviceSpan<int32_t> next_indices) override {
    if (encoder_state_) {
      encoder_state_->RunEncoder();
      encoder_state_.reset();
    }
    return decoder_state_.Run(total_length, next_tokens, next_indices);
  }

  void RewindTo(size_t) override {
    throw std::runtime_error(
        "Nemotron Parse TensorScatter does not support rewind");
  }

 private:
  const NemotronParseModel& model_;
  std::unique_ptr<EncoderState> encoder_state_;
  DecoderState decoder_state_;
  std::unique_ptr<CrossCache> cross_cache_;
};

}  // namespace

NemotronParseModel::NemotronParseModel(std::unique_ptr<Config> config,
                                       OrtEnv& ort_env)
    : Model{std::move(config)} {
  const auto& decoder = config_->model.decoder;
  if (config_->model.vision.filename.empty() || decoder.filename.empty() ||
      config_->model.context_length <= 0 ||
      decoder.prefill_sequence_length <= 0 ||
      decoder.prefill_sequence_length >= config_->model.context_length ||
      decoder.num_hidden_layers <= 0 || decoder.num_key_value_heads <= 0 ||
      decoder.head_size <= 0 || decoder.inputs.cache_write_indices.empty() ||
      decoder.inputs.past_key_names.empty() ||
      decoder.inputs.past_value_names.empty() ||
      decoder.inputs.cross_past_key_names.empty() ||
      decoder.inputs.cross_past_value_names.empty() ||
      decoder.outputs.present_key_names.empty() ||
      decoder.outputs.present_value_names.empty() ||
      config_->model.encoder.outputs.cross_present_key_names.empty() ||
      config_->model.encoder.outputs.cross_present_value_names.empty() ||
      config_->model.vision.num_visual_tokens <= 0) {
    throw std::runtime_error(
        "Nemotron Parse TensorScatter config is missing cache metadata");
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

  if (p_device_->GetType() == DeviceType::NvTensorRtRtx) {
    // Reuse one ONNX file, but create independently optimized prefill and
    // decode sessions. A single dynamic engine is materially slower at Q=1.
    prefill_decoder_session_options_ = OrtSessionOptions::Create();
    CreateSessionOptionsFromConfig(
        decoder.session_options, *prefill_decoder_session_options_,
        /*is_primary_session_options=*/false);
    SpecializeDecoderSession(*session_options_, *config_, 1);
    SpecializeDecoderSession(*prefill_decoder_session_options_, *config_,
                             decoder.prefill_sequence_length);
  }

  encoder_session_ = CreateSession(ort_env, config_->model.vision.filename,
                                   encoder_session_options_.get());
  decoder_session_ = CreateSession(ort_env, decoder.filename,
                                   session_options_.get());
  if (prefill_decoder_session_options_) {
    prefill_decoder_session_ = CreateSession(
        ort_env, decoder.filename, prefill_decoder_session_options_.get());
  }

  session_info_.Add(*decoder_session_);
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
        "Nemotron Parse context_length must match the static attention-mask shape");
  }
}

std::unique_ptr<State> NemotronParseModel::CreateState(
    DeviceSpan<int32_t> sequence_lengths, const GeneratorParams& params) const {
  return std::make_unique<NemotronParseState>(*this, sequence_lengths, params);
}

}  // namespace Generators
