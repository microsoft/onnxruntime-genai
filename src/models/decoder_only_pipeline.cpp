// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "../generators.h"
#include "decoder_only_pipeline.h"

namespace Generators {

DecoderOnlyPipelineModel::DecoderOnlyPipelineModel(std::unique_ptr<Config> config, OrtEnv& ort_env)
    : Model{std::move(config)} {
  for (const auto& model : config_->model.decoder.pipeline) {
    sessions_.emplace_back(OrtSession::Create(ort_env, (config_->config_path / fs::path(model.filename)).c_str(),
                                              GetSessionOptions(model.model_id)));

    if (!allocator_device_ && model.session_options.has_value()) {
      const auto& provider_options = (*model.session_options).provider_options;
      if (std::any_of(provider_options.begin(), provider_options.end(),
                      [](const auto& elem) { return !elem.name.empty(); })) {
        InitDeviceAllocator(*sessions_.back());
      }
    }
  }

  if (!allocator_device_) {
    // If the device allocator has not been created, it implies all
    // sessions are configured to run on CPU.
    // Pick any session to create the device allocator.
    // Device allocator is guaranteed to be the cpu allocator.
    InitDeviceAllocator(*sessions_.front());
  }

  for (auto& session : sessions_) {
    session_info_->Add(*session);
  }
}

std::unique_ptr<State> DecoderOnlyPipelineModel::CreateState(RoamingArray<int32_t> sequence_lengths,
                                                             const GeneratorParams& params) const {
  return std::make_unique<DecoderOnlyPipelineState>(*this, sequence_lengths, params);
}

IntermediatePipelineState::IntermediatePipelineState(const DecoderOnlyPipelineModel& model, const GeneratorParams& params,
                                                     size_t pipeline_state_index)
    : State{params, model},
      id_{pipeline_state_index},
      model_{model} {}

bool IntermediatePipelineState::HasInput(std::string_view name) const {
  return std::any_of(model_.config_->model.decoder.pipeline[id_].inputs.begin(),
                     model_.config_->model.decoder.pipeline[id_].inputs.end(),
                     [&name](const std::string& elem) { return elem == name; });
}

bool IntermediatePipelineState::HasOutput(std::string_view name) const {
  return std::any_of(model_.config_->model.decoder.pipeline[id_].outputs.begin(),
                     model_.config_->model.decoder.pipeline[id_].outputs.end(),
                     [&name](const std::string& elem) { return elem == name; });
}

bool IntermediatePipelineState::SupportsPrimaryDevice() const {
  if (model_.device_type_ == DeviceType::CPU) {
    return true;
  } else if (model_.device_type_ == DeviceType::CUDA) {
    if (!model_.config_->model.decoder.pipeline[id_].session_options.has_value()) {
      // No session options, so this session uses the default session options.
      // Default session options supports the cuda device type.
      return true;
    } else if (auto& provider_options = (*model_.config_->model.decoder.pipeline[id_].session_options).provider_options;
               std::any_of(provider_options.begin(), provider_options.end(),
                           [](const Config::ProviderOptions& elem) { return elem.name == "cuda"; })) {
      // cuda is listed as one of the providers. This session supports the cuda device type.
      return true;
    } else {
      // cuda is not listed as one of the providers. This session does not support the cuda device type.
      return false;
    }
  }

  return false;
}

RoamingArray<float> IntermediatePipelineState::Run(int current_length, RoamingArray<int32_t> next_tokens,
                                                   RoamingArray<int32_t> next_indices) {
  State::Run(*model_.sessions_[id_], params_->BatchBeamSize());

  return RoamingArray<float>();
}

DecoderOnlyPipelineState::DecoderOnlyPipelineState(const DecoderOnlyPipelineModel& model,
                                                   RoamingArray<int32_t> sequence_lengths,
                                                   const GeneratorParams& params)
    : State{params, model},
      model_{model},
      position_inputs_{model, *this, sequence_lengths} {
  input_ids_.Add();
  position_inputs_.Add();
  logits_.Add();
  if (KV_Cache::IsCacheNeeded(model)) {
    kv_cache_ = std::make_unique<KV_Cache>(*this);
    kv_cache_->Add();
  }
  extra_inputs_.Add();

  for ([[maybe_unused]] const auto& pipeline_model : model_.config_->model.decoder.pipeline) {
    pipeline_states_.emplace_back(std::make_unique<IntermediatePipelineState>(model_, params, pipeline_states_.size()));
  }
}

RoamingArray<float> DecoderOnlyPipelineState::Run(int current_length, RoamingArray<int32_t> next_tokens,
                                                  RoamingArray<int32_t> next_indices) {
  if (!first_run_) {
    UpdateInputsOutputs(next_tokens, next_indices, current_length);
  }

  for (auto& pipeline_state : pipeline_states_) {
    if (first_run_ && !model_.config_->model.decoder.pipeline[pipeline_state->id_].run_on_prompt) {
      continue;
    } else if (!first_run_ && !model_.config_->model.decoder.pipeline[pipeline_state->id_].run_on_token_gen) {
      continue;
    }

    // Clear the intermediate pipeline state outputs from the previous runs.
    // These outputs will be replaced by the outputs from the current run.
    for (const auto& output_name : pipeline_state->output_names_) {
      if (auto iter = ortvalue_store_.find(output_name); iter != ortvalue_store_.end()) {
        ortvalue_store_.erase(iter);
      }
    }
    pipeline_state->ClearIO();

    // Managed inputs and outputs are those inputs and outputs that the
    // Model knows how to create and update from one run to the next.

    // Add all the managed inputs to the intermediate pipeline state
    for (const auto& input_name : input_names_) {
      if (pipeline_state->HasInput(input_name)) {
        if (!pipeline_state->SupportsPrimaryDevice()) {
          std::ostringstream oss;
          oss << "Managed input " << input_name << " resides on the primary device type ("
              << to_string(model_.device_type_) << "). "
              << "But the pipeline model "
              << model_.config_->model.decoder.pipeline[pipeline_state->id_].model_id
              << " is expecting it to reside elsewhere.";
          throw std::runtime_error(oss.str());
        }
        pipeline_state->input_names_.push_back(input_name);
        pipeline_state->inputs_.push_back(State::GetInput(input_name));
      }
    }

    // Add outputs from the previous pipeline states to the current pipeline state
    for (auto& [name, ortvalue] : ortvalue_store_) {
      if (pipeline_state->HasInput(name)) {
        pipeline_state->input_names_.push_back(name.c_str());
        pipeline_state->inputs_.push_back(ortvalue.get());
      }
    }

    // Add all the managed outputs to the intermediate pipeline state
    for (const auto& output_name : output_names_) {
      if (pipeline_state->HasOutput(output_name)) {
        if (!pipeline_state->SupportsPrimaryDevice()) {
          std::ostringstream oss;
          oss << "Managed output " << output_name << " resides on the primary device type ("
              << to_string(model_.device_type_) << "). "
              << "But the pipeline model "
              << model_.config_->model.decoder.pipeline[pipeline_state->id_].model_id
              << " is expecting it to reside elsewhere.";
          throw std::runtime_error(oss.str());
        }
        pipeline_state->output_names_.push_back(output_name);
        pipeline_state->outputs_.push_back(State::GetOutput(output_name));
      }
    }

    // Output of pipeline models could also be managed inputs.
    // For example, the output of a pipeline model could be the key-value cache.
    // In such cases, use the managed output buffers and register them with the pipeline model as outputs.
    for (const auto& input_name : input_names_) {
      if (pipeline_state->HasOutput(input_name)) {
        if (!pipeline_state->SupportsPrimaryDevice()) {
          std::ostringstream oss;
          oss << "Managed input " << input_name << " resides on the primary device type ("
              << to_string(model_.device_type_) << "). "
              << "But the pipeline model "
              << model_.config_->model.decoder.pipeline[pipeline_state->id_].model_id
              << " is expecting it to reside elsewhere.";
          throw std::runtime_error(oss.str());
        }
        pipeline_state->output_names_.push_back(input_name);
        pipeline_state->outputs_.push_back(State::GetInput(input_name));
      }
    }

    // Add all the remaining outputs for the intermediate pipeline state
    for (const auto& output_name : model_.config_->model.decoder.pipeline[pipeline_state->id_].outputs) {
      if (std::none_of(pipeline_state->output_names_.begin(), pipeline_state->output_names_.end(),
                       [&](const std::string& elem) { return elem == output_name; })) {
        pipeline_state->output_names_.push_back(output_name.c_str());
        pipeline_state->outputs_.push_back(nullptr);
      }
    }

    // Run the intermediate pipeline state
    pipeline_state->Run(current_length, next_tokens, next_indices);

    // Transfer ownership of all the non-managed outputs from the current pipeline state to the ortvalue store.
    // All non managed outputs are assumed to be on CPU
    for (size_t i = 0; i < pipeline_state->output_names_.size(); ++i) {
      if (std::none_of(output_names_.begin(), output_names_.end(),
                       [&](const std::string& elem) { return elem == pipeline_state->output_names_[i]; }) &&
          std::none_of(input_names_.begin(), input_names_.end(),
                       [&](const std::string& elem) { return elem == pipeline_state->output_names_[i]; })) {
        auto forwarded_output = model_.config_->model.decoder.pipeline[pipeline_state->id_].output_names_forwarder.find(pipeline_state->output_names_[i]);
        if (forwarded_output != model_.config_->model.decoder.pipeline[pipeline_state->id_].output_names_forwarder.end()) {
          ortvalue_store_[forwarded_output->second] = std::unique_ptr<OrtValue>(pipeline_state->outputs_[i]);
        } else {
          ortvalue_store_[pipeline_state->output_names_[i]] = std::unique_ptr<OrtValue>(pipeline_state->outputs_[i]);
        }
      }
    }
  }

  // Clear the outputs of the pipeline models that are only run on prompt since this cannot happen earlier.
  if (!first_run_) {
    for (auto& pipeline_state : pipeline_states_) {
      if (!model_.config_->model.decoder.pipeline[pipeline_state->id_].run_on_token_gen) {
        for (const auto& output_name : pipeline_state->output_names_) {
          if (auto iter = ortvalue_store_.find(output_name); iter != ortvalue_store_.end()) {
            ortvalue_store_.erase(iter);
          }
        }
      }
    }
  }

  first_run_ = false;

  return logits_.Get();
}

void DecoderOnlyPipelineState::UpdateInputsOutputs(const RoamingArray<int32_t>& next_tokens_unk,
                                                   RoamingArray<int32_t> beam_indices, int current_length) {
  input_ids_.Update(next_tokens_unk);
  position_inputs_.Update(current_length);
  if (kv_cache_) kv_cache_->Update(beam_indices.GetCPU(), current_length);
  logits_.Update();
}

OrtValue* DecoderOnlyPipelineState::GetOutput(const char* name) {
  // Check the ortvalue store to search if name is one of the non-managed output.
  auto it = ortvalue_store_.find(name);
  if (it != ortvalue_store_.end()) {
    return it->second.get();
  }

  // Search managed outputs saved in this State.
  return State::GetOutput(name);
}

}  // namespace Generators
