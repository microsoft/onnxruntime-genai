// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "../generators.h"
#include "pipeline_config.h"

namespace Generators {

// ============================================================================
// PipelineConfigModel
// ============================================================================

PipelineConfigModel::PipelineConfigModel(
    std::unique_ptr<Config> config, OrtEnv& ort_env)
    : DecoderOnly_Model{std::move(config), ort_env} {
  // DecoderOnly_Model already loaded session_decoder_ and registered it
  // with session_info_.  Now load any additional sessions for multi-session
  // pipelines (vision, embedding, encoder, etc.).
  //
  // Note: For v2 configs, the v1 translator or preset resolution already
  // syncs pipeline_config.sessions["decoder"].file to model.decoder.filename
  // so that DecoderOnly_Model's constructor loads the correct session.
  const auto& pipeline_config = config_->pipeline_config;

  for (const auto& [name, session_config] : pipeline_config.sessions) {
    if (name == "decoder" || session_config.file.empty()) continue;

    // Non-decoder sessions get separate options with graph capture disabled.
    // Vision/embedding/encoder sessions often have control flow nodes that
    // are incompatible with CUDA graph capture.
    auto custom_opts = OrtSessionOptions::Create();
    CreateSessionOptionsFromConfig(
        config_->model.decoder.session_options,
        *custom_opts, /*is_primary_session_options=*/true,
        /*disable_graph_capture=*/true);
    non_decoder_session_options_[name] = std::move(custom_opts);

    extra_sessions_[name] = CreateSession(
        ort_env, session_config.file,
        non_decoder_session_options_[name].get());
    session_info_.Add(*extra_sessions_[name]);
  }

  flow_interpreter_ = std::make_unique<FlowInterpreter>(pipeline_config);
}

std::unique_ptr<State> PipelineConfigModel::CreateState(
    DeviceSpan<int32_t> sequence_lengths,
    const GeneratorParams& params) const {
  // For single-session (decoder-only) configs, use DecoderOnly_State directly.
  // Zero code duplication — full decoder parity including sliding window,
  // chunking, graph capture, and all future DecoderOnly_State improvements.
  if (!flow_interpreter_->IsMultiSession()) {
    return std::make_unique<DecoderOnly_State>(
        *this, sequence_lengths, params);
  }
  return std::make_unique<PipelineConfigState>(
      *this, sequence_lengths, params);
}

// ============================================================================
// PipelineConfigState — multi-session orchestration
// ============================================================================

PipelineConfigState::PipelineConfigState(
    const PipelineConfigModel& model,
    DeviceSpan<int32_t> sequence_lengths,
    const GeneratorParams& params)
    : State{params, model},
      model_{model},
      decoder_state_{std::make_unique<DecoderOnly_State>(
          model_, sequence_lengths, params)} {
}

void PipelineConfigState::SetExtraInputs(
    const std::vector<ExtraInput>& extra_inputs) {
  // Route extra inputs to the decoder state
  decoder_state_->SetExtraInputs(extra_inputs);

  // Route extra inputs to non-decoder sessions (store owned copies)
  for (const auto& [name, session] : model_.extra_sessions_) {
    auto& io = non_decoder_io_[name];
    auto session_input_names = session->GetInputNames();
    for (const auto& extra : extra_inputs) {
      for (const auto& session_input : session_input_names) {
        if (extra.name == session_input) {
          io.input_name_strings.push_back(extra.name);
          io.inputs.push_back(extra.tensor->ort_tensor_.get());
        }
      }
    }
  }
}

void PipelineConfigState::RunNonDecoderSession(
    const std::string& session_name) {
  auto session_it = model_.extra_sessions_.find(session_name);
  if (session_it == model_.extra_sessions_.end() || !session_it->second) {
    throw std::runtime_error(
        "Pipeline flow error: session '" + session_name + "' not loaded");
  }

  auto& session = *session_it->second;

  // Build input arrays from extra inputs + wired intermediates.
  // Maintain string ownership in input_name_strings to avoid dangling ptrs.
  std::vector<std::string> input_name_strings;
  std::vector<OrtValue*> input_values;

  auto io_it = non_decoder_io_.find(session_name);
  if (io_it != non_decoder_io_.end()) {
    for (size_t i = 0; i < io_it->second.input_name_strings.size(); ++i) {
      input_name_strings.push_back(io_it->second.input_name_strings[i]);
      input_values.push_back(io_it->second.inputs[i]);
    }
  }

  auto wired_inputs = model_.flow_interpreter_->GetWiredInputs(
      session_name, intermediates_);
  for (auto& [input_name, value] : wired_inputs) {
    input_name_strings.push_back(input_name);
    input_values.push_back(value);
  }

  std::vector<const char*> input_name_ptrs;
  input_name_ptrs.reserve(input_name_strings.size());
  for (const auto& name : input_name_strings) {
    input_name_ptrs.push_back(name.c_str());
  }

  auto output_name_strings = session.GetOutputNames();
  std::vector<const char*> output_name_ptrs;
  output_name_ptrs.reserve(output_name_strings.size());
  for (const auto& name : output_name_strings) {
    output_name_ptrs.push_back(name.c_str());
  }
  std::vector<OrtValue*> output_values(output_name_strings.size(), nullptr);

  session.Run(nullptr,
              input_name_ptrs.data(),
              reinterpret_cast<const OrtValue* const*>(input_values.data()),
              input_name_ptrs.size(),
              output_name_ptrs.data(), output_values.data(),
              output_name_ptrs.size());

  // Store outputs as intermediates (owned by this State, not Model).
  // Also update output_by_tensor_name_ so GetOutput returns the last-
  // executed session's output (flow order, not map order).
  for (size_t i = 0; i < output_name_strings.size(); ++i) {
    if (output_values[i]) {
      auto key = session_name + "." + output_name_strings[i];
      intermediates_[key] = output_values[i];
      intermediate_owned_[key] =
          std::unique_ptr<OrtValue>(output_values[i]);
      output_by_tensor_name_[output_name_strings[i]] = output_values[i];
    }
  }
}

void PipelineConfigState::RunFlowStep(
    const PipelineConfig::FlowStep& step,
    int total_length,
    DeviceSpan<int32_t>& next_tokens,
    DeviceSpan<int32_t> next_indices) {
  if (step.run == "decoder") {
    // Wire any intermediate inputs (e.g. inputs_embeds from embedding session)
    // into the decoder state's input bindings before running.
    // Save original size so we can restore after the run — this prevents
    // dangling pointers when prompt intermediates are freed later.
    const size_t original_input_count = decoder_state_->input_names_.size();

    auto wired = model_.flow_interpreter_->GetWiredInputs(
        "decoder", intermediates_);
    for (const auto& [input_name, value] : wired) {
      bool found = false;
      for (size_t i = 0; i < original_input_count; ++i) {
        if (std::strcmp(decoder_state_->input_names_[i],
                        input_name.c_str()) == 0) {
          decoder_state_->inputs_[i] = value;
          found = true;
          break;
        }
      }
      if (!found) {
        wired_decoder_input_names_.push_back(input_name);
        decoder_state_->input_names_.push_back(
            wired_decoder_input_names_.back().c_str());
        decoder_state_->inputs_.push_back(value);
      }
    }

    // Delegate to DecoderOnly_State::Run which handles everything:
    // KV cache, position inputs, logits, sliding window, chunking,
    // graph capture, run options.
    last_logits_ = decoder_state_->Run(total_length, next_tokens, next_indices);

    // Restore decoder input bindings to original size so that
    // prompt-only intermediate pointers don't persist.
    decoder_state_->input_names_.resize(original_input_count);
    decoder_state_->inputs_.resize(original_input_count);
    return;
  }

  // Handle non-decoder step.loop if specified
  if (!step.loop.empty() && step.loop == "per_image") {
    // TODO: Implement per-image looping for vision sessions.
    // For now, run the session once (single-image support).
    RunNonDecoderSession(step.run);
  } else {
    RunNonDecoderSession(step.run);
  }
}

DeviceSpan<float> PipelineConfigState::Run(
    int total_length, DeviceSpan<int32_t>& next_tokens,
    DeviceSpan<int32_t> next_indices) {
  // Multi-session pipeline execution
  if (is_prompt_) {
    for (const auto& step : model_.flow_interpreter_->init_steps()) {
      RunFlowStep(step, total_length, next_tokens, next_indices);
    }
  }

  for (const auto& step : model_.flow_interpreter_->step_steps()) {
    RunFlowStep(step, total_length, next_tokens, next_indices);
  }

  if (is_prompt_) {
    is_prompt_ = false;
    // Clear init-only intermediates from both maps
    const auto& init_sessions = model_.flow_interpreter_->init_only_sessions();
    for (auto it = intermediates_.begin(); it != intermediates_.end();) {
      auto dot = it->first.find('.');
      if (dot != std::string::npos &&
          init_sessions.count(it->first.substr(0, dot))) {
        intermediate_owned_.erase(it->first);
        it = intermediates_.erase(it);
      } else {
        ++it;
      }
    }
  }

  // Logits were populated by DecoderOnly_State::Run inside RunFlowStep("decoder")
  return last_logits_;
}

void PipelineConfigState::Finalize(int current_length) {
  // Run final-phase steps (e.g., TTS vocoder, diffusion VAE decode).
  // These execute once after the generation loop completes.
  DeviceSpan<int32_t> empty_tokens;
  for (const auto& step : model_.flow_interpreter_->final_steps()) {
    RunFlowStep(step, current_length, empty_tokens, {});
  }
  decoder_state_->Finalize(current_length);
}

void PipelineConfigState::RewindTo(size_t index) {
  decoder_state_->RewindTo(index);
}

OrtValue* PipelineConfigState::GetInput(const char* name) {
  for (const auto& [session_name, io] : non_decoder_io_) {
    for (size_t i = 0; i < io.input_name_strings.size(); ++i) {
      if (io.input_name_strings[i] == name) {
        return io.inputs[i];
      }
    }
  }
  return decoder_state_->GetInput(name);
}

OrtValue* PipelineConfigState::GetOutput(const char* name) {
  // Return the last-executed session's output for this tensor name.
  auto it = output_by_tensor_name_.find(name);
  if (it != output_by_tensor_name_.end()) return it->second;
  return decoder_state_->GetOutput(name);
}

}  // namespace Generators
