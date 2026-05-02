// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "../generators.h"
#include "pipeline_config.h"

namespace Generators {

// ============================================================================
// PipelineConfigModel — loads sessions from pipeline config
// ============================================================================

PipelineConfigModel::PipelineConfigModel(
    std::unique_ptr<Config> config, OrtEnv& ort_env)
    : Model{std::move(config)} {
  const auto& pipeline_config = config_->pipeline_config;

  // Load each named session from the pipeline config
  for (const auto& [name, session_config] : pipeline_config.sessions) {
    if (session_config.file.empty()) continue;

    OrtSessionOptions* opts = session_options_.get();

    // Non-decoder sessions get separate options with graph capture disabled.
    // Vision/embedding/encoder sessions often have control flow nodes that
    // are incompatible with CUDA graph capture.
    if (session_config.role != "decoder") {
      auto custom_opts = OrtSessionOptions::Create();
      CreateSessionOptionsFromConfig(
          config_->model.decoder.session_options,
          *custom_opts, /*is_primary_session_options=*/true,
          /*disable_graph_capture=*/true);
      non_decoder_session_options_[name] = std::move(custom_opts);
      opts = non_decoder_session_options_[name].get();
    }

    sessions_[name] = CreateSession(ort_env, session_config.file, opts);
    session_info_.Add(*sessions_[name]);
  }

  flow_interpreter_ = std::make_unique<FlowInterpreter>(pipeline_config);
}

std::unique_ptr<State> PipelineConfigModel::CreateState(
    DeviceSpan<int32_t> sequence_lengths,
    const GeneratorParams& params) const {
  return std::make_unique<PipelineConfigState>(
      *this, sequence_lengths, params);
}

// ============================================================================
// PipelineConfigState — orchestrates pipeline execution
// ============================================================================

PipelineConfigState::PipelineConfigState(
    const PipelineConfigModel& model,
    DeviceSpan<int32_t> sequence_lengths,
    const GeneratorParams& params)
    : State{params, model},
      model_{model},
      kv_cache_(CreateKeyValueCache(*this)),
      position_inputs_{CreatePositionInputs(
          *this, sequence_lengths,
          model_.config_->model.decoder.inputs.attention_mask)} {
  input_ids_.Add();
  position_inputs_->Add();
  logits_.Add();
  if (kv_cache_) kv_cache_->Add();
}

void PipelineConfigState::SetExtraInputs(
    const std::vector<ExtraInput>& extra_inputs) {
  // Route extra inputs to the decoder session
  if (model_.sessions_.count("decoder")) {
    extra_inputs_.Add(extra_inputs,
                      model_.sessions_.at("decoder")->GetInputNames());
  }

  // Route extra inputs to non-decoder sessions
  for (const auto& [name, session] : model_.sessions_) {
    if (name == "decoder") continue;
    auto& io = non_decoder_io_[name];
    auto session_input_names = session->GetInputNames();
    for (const auto& extra : extra_inputs) {
      for (const auto& session_input : session_input_names) {
        if (extra.name == session_input) {
          io.input_names.push_back(extra.name.c_str());
          io.inputs.push_back(extra.tensor->ort_tensor_.get());
        }
      }
    }
  }
}

void PipelineConfigState::RunNonDecoderSession(
    const std::string& session_name) {
  auto session_it = model_.sessions_.find(session_name);
  if (session_it == model_.sessions_.end() || !session_it->second) {
    throw std::runtime_error(
        "Pipeline flow error: session '" + session_name + "' not loaded");
  }

  auto& session = *session_it->second;

  // Build input arrays from extra inputs + wired intermediates.
  // We maintain string ownership in input_name_strings to avoid dangling ptrs.
  std::vector<std::string> input_name_strings;
  std::vector<OrtValue*> input_values;

  // Add extra inputs that were set for this session
  auto io_it = non_decoder_io_.find(session_name);
  if (io_it != non_decoder_io_.end()) {
    for (size_t i = 0; i < io_it->second.input_names.size(); ++i) {
      input_name_strings.push_back(io_it->second.input_names[i]);
      input_values.push_back(io_it->second.inputs[i]);
    }
  }

  // Add wired intermediate inputs from upstream sessions
  auto wired_inputs = model_.flow_interpreter_->GetWiredInputs(session_name);
  for (auto& [input_name, value] : wired_inputs) {
    input_name_strings.push_back(input_name);
    input_values.push_back(value);
  }

  // Build const char* array from owned strings
  std::vector<const char*> input_name_ptrs;
  input_name_ptrs.reserve(input_name_strings.size());
  for (const auto& name : input_name_strings) {
    input_name_ptrs.push_back(name.c_str());
  }

  // Set up outputs: query session for expected output names
  auto output_name_strings = session.GetOutputNames();
  std::vector<const char*> output_name_ptrs;
  output_name_ptrs.reserve(output_name_strings.size());
  for (const auto& name : output_name_strings) {
    output_name_ptrs.push_back(name.c_str());
  }
  std::vector<OrtValue*> output_values(output_name_strings.size(), nullptr);

  // Run session
  session.Run(nullptr,
              input_name_ptrs.data(),
              reinterpret_cast<const OrtValue* const*>(input_values.data()),
              input_name_ptrs.size(),
              output_name_ptrs.data(), output_values.data(),
              output_name_ptrs.size());

  // Store outputs as intermediates for downstream sessions.
  // Take ownership of ORT-allocated output tensors.
  for (size_t i = 0; i < output_name_strings.size(); ++i) {
    if (output_values[i]) {
      auto key = session_name + "." + output_name_strings[i];
      model_.flow_interpreter_->StoreIntermediate(
          session_name, output_name_strings[i], output_values[i]);
      intermediate_store_[key] =
          std::unique_ptr<OrtValue>(output_values[i]);
    }
  }
}

void PipelineConfigState::RunFlowStep(
    const PipelineConfig::FlowStep& step,
    int total_length,
    DeviceSpan<int32_t>& next_tokens,
    DeviceSpan<int32_t> next_indices) {
  if (step.run == "decoder") {
    // Decoder runs through the main state with managed components.
    // Wire any intermediate inputs (e.g. inputs_embeds from embedding session)
    // into the decoder's input bindings.
    auto wired = model_.flow_interpreter_->GetWiredInputs("decoder");
    for (const auto& [input_name, value] : wired) {
      // Find if this input is already bound in the state
      bool found = false;
      for (size_t i = 0; i < input_names_.size(); ++i) {
        if (std::strcmp(input_names_[i], input_name.c_str()) == 0) {
          inputs_[i] = value;
          found = true;
          break;
        }
      }
      if (!found) {
        // Store name persistently and add new input binding
        wired_input_names_.push_back(input_name);
        input_names_.push_back(wired_input_names_.back().c_str());
        inputs_.push_back(value);
      }
    }

    if (model_.config_->model.decoder.run_options.has_value()) {
      State::SetRunOptions(model_.config_->model.decoder.run_options.value());
    }

    bool graph_capture = params_->use_graph_capture &&
                         input_ids_.GetShape()[1] == 1;
    State::Run(*model_.sessions_.at("decoder"), graph_capture);
    return;
  }

  // Non-decoder session
  RunNonDecoderSession(step.run);
}

DeviceSpan<float> PipelineConfigState::Run(
    int total_length, DeviceSpan<int32_t>& next_tokens,
    DeviceSpan<int32_t> next_indices) {
  UpdateInputsOutputs(next_tokens, next_indices, total_length);

  if (!model_.flow_interpreter_->IsMultiSession()) {
    // Single session (decoder-only): simple path matching DecoderOnly_State
    if (model_.config_->model.decoder.run_options.has_value()) {
      State::SetRunOptions(model_.config_->model.decoder.run_options.value());
    }
    bool graph_capture = params_->use_graph_capture &&
                         input_ids_.GetShape()[1] == 1;
    State::Run(*model_.sessions_.at("decoder"), graph_capture);
    return logits_.Get();
  }

  // Multi-session pipeline execution
  if (is_prompt_) {
    // Run prompt-only steps (vision, encoder, etc.)
    for (const auto& step : model_.flow_interpreter_->prompt_steps()) {
      RunFlowStep(step, total_length, next_tokens, next_indices);
    }
  }

  // Run always-steps (embedding, decoder)
  for (const auto& step : model_.flow_interpreter_->always_steps()) {
    RunFlowStep(step, total_length, next_tokens, next_indices);
  }

  if (is_prompt_) {
    is_prompt_ = false;
    model_.flow_interpreter_->ClearPromptIntermediates();
  }

  return logits_.Get();
}

void PipelineConfigState::UpdateInputsOutputs(
    DeviceSpan<int32_t>& next_tokens,
    DeviceSpan<int32_t> beam_indices,
    int total_length) {
  input_ids_.Update(next_tokens);
  size_t new_length = static_cast<size_t>(input_ids_.GetShape()[1]);
  position_inputs_->Update(next_tokens, total_length,
                           static_cast<int>(new_length));
  if (kv_cache_) kv_cache_->Update(beam_indices, total_length);
  logits_.Update(next_tokens, new_length);
}

void PipelineConfigState::RewindTo(size_t index) {
  if (kv_cache_) kv_cache_->RewindTo(index);
  position_inputs_->RewindTo(index);
}

OrtValue* PipelineConfigState::GetInput(const char* name) {
  // Check non-decoder I/O stores
  for (const auto& [session_name, io] : non_decoder_io_) {
    for (size_t i = 0; i < io.input_names.size(); ++i) {
      if (std::strcmp(io.input_names[i], name) == 0) {
        return io.inputs[i];
      }
    }
  }
  return State::GetInput(name);
}

OrtValue* PipelineConfigState::GetOutput(const char* name) {
  // Check intermediate store for non-decoder outputs
  for (const auto& [key, value] : intermediate_store_) {
    // Extract tensor name from "session.tensor" key
    auto dot = key.find('.');
    if (dot != std::string::npos && key.substr(dot + 1) == name) {
      return value.get();
    }
  }
  return State::GetOutput(name);
}

}  // namespace Generators
