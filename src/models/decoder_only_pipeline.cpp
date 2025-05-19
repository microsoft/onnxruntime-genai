// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "../generators.h"
#include "../logging.h"
#include "../timer.h"
#include "decoder_only_pipeline.h"
#include "windowed_kv_cache.h"

namespace Generators {

DecoderOnlyPipelineModel::DecoderOnlyPipelineModel(std::unique_ptr<Config> config, OrtEnv& ort_env)
    : Model{std::move(config)}, ort_env_{ort_env} {
  for (const auto& model : config_->model.decoder.pipeline) {
    sessions_.emplace_back(OrtSession::Create(ort_env, (config_->config_path / fs::path(model.filename)).c_str(),
                                              GetSessionOptions(model.model_id)));
    auto session_info = std::make_unique<SessionInfo>();
    session_info->Add(*sessions_.back());
    session_infos_.emplace_back(std::move(session_info));
  }

  for (auto& session : sessions_) {
    session_info_.Add(*session);
  }
}

std::unique_ptr<State> DecoderOnlyPipelineModel::CreateState(DeviceSpan<int32_t> sequence_lengths,
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
  if (model_.p_device_->GetType() == DeviceType::CPU || model_.p_device_->GetType() == DeviceType::QNN) {
    return true;
  } else if (model_.p_device_->GetType() == DeviceType::CUDA) {
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

DeviceSpan<float> IntermediatePipelineState::Run(int total_length, DeviceSpan<int32_t>& next_tokens,
                                                 DeviceSpan<int32_t> next_indices) {
  if (!model_.sessions_[id_]) {
    const_cast<DecoderOnlyPipelineModel*>(&model_)->sessions_[id_] =
        OrtSession::Create(model_.ort_env_, (model_.config_->config_path / fs::path(model_.config_->model.decoder.pipeline[id_].filename)).c_str(),
                           model_.GetSessionOptions(model_.config_->model.decoder.pipeline[id_].model_id));
  }

  Timer timer{};
  State::Run(*model_.sessions_[id_]);
  const auto elapsed = timer.Elapsed();
  LogTiming(MakeString("intermediate pipeline state run [", id_, "]"), timer.Elapsed());

  return {};
}

using NameToLayerIdxMap = std::unordered_map<std::string, size_t>;

static NameToLayerIdxMap GeneratePastKeyNameToLayerIdxMap(const Config& config) {
  const size_t num_layers = config.model.decoder.num_hidden_layers;
  const std::string& past_key_name_template = config.model.decoder.inputs.past_key_names;
  NameToLayerIdxMap m{};
  for (size_t i = 0; i < num_layers; ++i) {
    m.emplace(ComposeKeyValueName(past_key_name_template, static_cast<int>(i)), i);
  }
  return m;
}

static std::vector<size_t> DetectLayerIndicesFromPastKeyNameInputs(
    const NameToLayerIdxMap& past_key_name_to_layer_idx, std::span<const std::string> inputs) {
  std::vector<size_t> detected_layer_indices{};
  for (const auto& input_name : inputs) {
    const auto it = past_key_name_to_layer_idx.find(input_name);
    if (it != past_key_name_to_layer_idx.end()) {
      detected_layer_indices.push_back(it->second);
    }
  }
  return detected_layer_indices;
}

DecoderOnlyPipelineState::DecoderOnlyPipelineState(const DecoderOnlyPipelineModel& model,
                                                   DeviceSpan<int32_t> sequence_lengths,
                                                   const GeneratorParams& params)
    : State{params, model},
      model_{model},
      input_ids_{CreateInputIDs(*this)},
      key_value_cache_{CreateKeyValueCache(*this)},
      do_key_value_cache_partial_token_generation_update_{
          key_value_cache_ && key_value_cache_->IsPartialTokenGenerationUpdateSupported()},
      position_inputs_{CreatePositionInputs(*this, sequence_lengths)} {
  input_ids_->Add();
  position_inputs_->Add();
  logits_.Add();
  if (key_value_cache_) {
    key_value_cache_->Add();
  }
  extra_inputs_.Add();

  const auto past_key_name_to_layer_idx = [&]() -> std::optional<NameToLayerIdxMap> {
    if (do_key_value_cache_partial_token_generation_update_) {
      return GeneratePastKeyNameToLayerIdxMap(*model_.config_);
    }
    return std::nullopt;
  }();

  for (const auto& pipeline_model : model_.config_->model.decoder.pipeline) {
    auto pipeline_model_state = std::make_unique<IntermediatePipelineState>(model_, params, pipeline_states_.size());

    auto overlapped_kv_cache_update_record = [&]() -> std::optional<OverlappedKeyValueCacheUpdateRecord> {
      if (do_key_value_cache_partial_token_generation_update_) {
        const bool token_gen_only = !pipeline_model.run_on_prompt && pipeline_model.run_on_token_gen;
        if (token_gen_only) {
          auto layer_indices = DetectLayerIndicesFromPastKeyNameInputs(*past_key_name_to_layer_idx,
                                                                       pipeline_model.inputs);
          if (!layer_indices.empty()) {
            // token generation model with KV cache tensors - we should overlap KV cache update
            auto record = OverlappedKeyValueCacheUpdateRecord{};
            record.layer_indices = std::move(layer_indices);
            return record;
          }
        }
      }
      return std::nullopt;
    }();

    pipeline_states_.emplace_back(std::move(pipeline_model_state));
    pipeline_overlapped_kv_cache_update_records_.emplace_back(std::move(overlapped_kv_cache_update_record));
  }

  if (std::any_of(pipeline_overlapped_kv_cache_update_records_.begin(),
                  pipeline_overlapped_kv_cache_update_records_.end(),
                  [](const auto& record) { return record.has_value(); })) {
    key_value_cache_update_worker_thread_.emplace();
  }
}

void DecoderOnlyPipelineState::RunPipeline(int total_length, DeviceSpan<int32_t>& next_tokens,
                                           DeviceSpan<int32_t> next_indices) {
  for (auto& pipeline_state : pipeline_states_) {
    if (first_run_ && !model_.config_->model.decoder.pipeline[pipeline_state->id_].run_on_prompt) {
      continue;
    } else if (!first_run_ && !model_.config_->model.decoder.pipeline[pipeline_state->id_].run_on_token_gen) {
      continue;
    }

    if (model_.config_->model.decoder.pipeline[pipeline_state->id_].reset_session_idx > -1) {
      if (model_.config_->model.decoder.pipeline[pipeline_state->id_].reset_session_idx >=
          static_cast<int>(model_.sessions_.size())) {
        throw std::runtime_error(
            MakeString("Invalid reset_session_idx ", model_.config_->model.decoder.pipeline[pipeline_state->id_].reset_session_idx,
                       " for pipeline model ", model_.config_->model.decoder.pipeline[pipeline_state->id_].model_id));
      }
      (const_cast<DecoderOnlyPipelineModel*>(&model_))->sessions_[model_.config_->model.decoder.pipeline[pipeline_state->id_].reset_session_idx].reset();
    }

    // Clear the intermediate pipeline state outputs from the previous runs.
    // These outputs will be replaced by the outputs from the current run.
    //for (const auto& output_name : pipeline_state->output_names_) {
    //  if (auto iter = ortvalue_store_.find(output_name); iter != ortvalue_store_.end()) {
    //    ortvalue_store_.erase(iter);
    //  }
    //}
    pipeline_state->ClearIO();

    // Managed inputs and outputs are those inputs and outputs that the
    // Model knows how to create and update from one run to the next.

    // Add all the managed inputs to the intermediate pipeline state
    for (const auto& input_name : input_names_) {
      if (pipeline_state->HasInput(input_name)) {
        if (!pipeline_state->SupportsPrimaryDevice()) {
          throw std::runtime_error(
              MakeString("Managed input ", input_name, " resides on the primary device type (",
                         static_cast<int>(model_.p_device_->GetType()), "). But the pipeline model ",
                         model_.config_->model.decoder.pipeline[pipeline_state->id_].model_id,
                         " is expecting it to reside elsewhere."));
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
          throw std::runtime_error(
              MakeString("Managed output ", output_name, " resides on the primary device type (",
                         static_cast<int>(model_.p_device_->GetType()), "). But the pipeline model ",
                         model_.config_->model.decoder.pipeline[pipeline_state->id_].model_id,
                         " is expecting it to reside elsewhere."));
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
          throw std::runtime_error(
              MakeString("Managed input ", input_name, " resides on the primary device type (",
                         static_cast<int>(model_.p_device_->GetType()), "). But the pipeline model ",
                         model_.config_->model.decoder.pipeline[pipeline_state->id_].model_id,
                         " is expecting it to reside elsewhere."));
        }
        pipeline_state->output_names_.push_back(input_name);
        pipeline_state->outputs_.push_back(State::GetInput(input_name));
      }
    }

    const auto& output_names_forwarder =
        model_.config_->model.decoder.pipeline[pipeline_state->id_].output_names_forwarder;

    auto get_forwarded_output_name = [&output_names_forwarder](std::string output_name) {
      const auto forwarded_name_it = output_names_forwarder.find(output_name);
      if (forwarded_name_it != output_names_forwarder.end()) {
        output_name = forwarded_name_it->second;
      }
      return output_name;
    };

    auto add_output_to_ort_value_store =
        [this, &get_forwarded_output_name](
            std::string output_name, std::unique_ptr<OrtValue> output_value) -> OrtValue* {
      output_name = get_forwarded_output_name(std::move(output_name));
      auto [it, inserted] = ortvalue_store_.insert_or_assign(std::move(output_name),
                                                             std::move(output_value));
      if (!inserted) {
        // log warning?
      }
      return it->second.get();
    };

    // Add all the remaining outputs for the intermediate pipeline state
    std::vector<size_t> intermediate_pipeline_state_outputs_allocated_by_ort_indices{};
    {
      for (const auto& output_name : model_.config_->model.decoder.pipeline[pipeline_state->id_].outputs) {
        if (std::any_of(pipeline_state->output_names_.begin(), pipeline_state->output_names_.end(),
                        [&](const std::string& elem) { return elem == output_name; })) {
          continue;
        }

        pipeline_state->output_names_.push_back(output_name.c_str());

        const auto& output_type_and_shape =
            model_.session_infos_[pipeline_state->id_]->GetOutputTensorTypeAndShapeInfo(output_name);
        const auto output_shape = output_type_and_shape.GetShape();

        const bool is_output_shape_static = std::all_of(output_shape.begin(), output_shape.end(),
                                                        [](int64_t dim) { return dim >= 0; });

        // pre-allocate with device allocator if shape is static
        if (is_output_shape_static) {
          const auto has_same_type_and_shape = [](const OrtTensorTypeAndShapeInfo& a,
                                                  const OrtTensorTypeAndShapeInfo& b) {
            return a.GetElementType() == b.GetElementType() && a.GetShape() == b.GetShape();
          };

          OrtValue* output_ptr{};
          auto ortvalue_it = ortvalue_store_.find(get_forwarded_output_name(output_name));
          if (ortvalue_it != ortvalue_store_.end() &&
              has_same_type_and_shape(output_type_and_shape, *ortvalue_it->second->GetTensorTypeAndShapeInfo())) {
            // re-use existing stored value with the same shape
            output_ptr = ortvalue_it->second.get();
          } else {
            auto device_output = OrtValue::CreateTensor(model_.p_device_->GetAllocator(), output_shape,
                                                        output_type_and_shape.GetElementType());
            output_ptr = add_output_to_ort_value_store(output_name, std::move(device_output));
          }
          pipeline_state->outputs_.push_back(output_ptr);

        } else {
          intermediate_pipeline_state_outputs_allocated_by_ort_indices.push_back(pipeline_state->outputs_.size());
          pipeline_state->outputs_.push_back(nullptr);
        }
      }
    }

    auto& overlapped_kv_update_record = pipeline_overlapped_kv_cache_update_records_[pipeline_state->id_];
    if (overlapped_kv_update_record.has_value()) {
      // wait for any outstanding KV cache update to finish
      if (overlapped_kv_update_record->outstanding_update.valid()) {
        overlapped_kv_update_record->outstanding_update.get();
      }
    }

    // Run the intermediate pipeline state
    pipeline_state->Run(total_length, next_tokens, next_indices);

    if (overlapped_kv_update_record.has_value()) {
      assert(key_value_cache_update_worker_thread_.has_value());
      // enqueue the next KV cache update
      auto update_fn = [&key_value_cache = *key_value_cache_.get(),
                        layer_indices = overlapped_kv_update_record->layer_indices,
                        next_indices, total_length]() {
        key_value_cache.PartialTokenGenerationUpdate(next_indices, total_length, layer_indices);
      };
      overlapped_kv_update_record->outstanding_update = key_value_cache_update_worker_thread_->Enqueue(update_fn);
    }

    // Transfer ownership of all the non-managed outputs from the current pipeline state to the ortvalue store.
    // All non managed outputs are assumed to be on CPU
    for (const auto i : intermediate_pipeline_state_outputs_allocated_by_ort_indices) {
      auto output_name = pipeline_state->output_names_[i];
      add_output_to_ort_value_store(output_name, std::unique_ptr<OrtValue>(pipeline_state->outputs_[i]));
    }
  }
}

DeviceSpan<float> DecoderOnlyPipelineState::Run(int total_length, DeviceSpan<int32_t>& next_tokens,
                                                DeviceSpan<int32_t> next_indices) {
  Timer timer{};

  UpdateInputsOutputs(next_tokens, next_indices, total_length);

  size_t num_chunks{1};
  if (first_run_ && model_.config_->model.decoder.sliding_window.has_value()) {
    int window_size = model_.config_->model.decoder.sliding_window->window_size;
    num_chunks = (next_tokens.size() + window_size - 1) / window_size;
  }

  for (size_t i = 0; i < num_chunks; ++i) {
    RunPipeline(total_length, next_tokens, next_indices);

    if (model_.config_->model.decoder.sliding_window.has_value() && i < num_chunks - 1) {
      // Sliding the window over the input_ids, key_cache, and value_cache, position_ids, and attention_mask
      input_ids_->Update(next_tokens);
      if (key_value_cache_) key_value_cache_->Update(next_indices, total_length);
      position_inputs_->Update(next_tokens, total_length, static_cast<int>(input_ids_->GetShape()[1]));
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

  auto logits = logits_.Get();

  const auto elapsed = timer.Elapsed();
  LogTiming("Run", elapsed);

  return logits;
}

void DecoderOnlyPipelineState::UpdateInputsOutputs(DeviceSpan<int32_t>& next_tokens,
                                                   DeviceSpan<int32_t> beam_indices, int total_length) {
  input_ids_->Update(next_tokens);
  size_t new_length = input_ids_->GetShape()[1];
  position_inputs_->Update(next_tokens, total_length, static_cast<int>(new_length));

  if (key_value_cache_) {
    const bool outstanding_key_value_cache_partial_token_generation_update =
        do_key_value_cache_partial_token_generation_update_ &&
        std::any_of(pipeline_overlapped_kv_cache_update_records_.rbegin(),
                    pipeline_overlapped_kv_cache_update_records_.rend(),
                    [](const std::optional<OverlappedKeyValueCacheUpdateRecord>& record) {
                      return record.has_value() && record->outstanding_update.valid();
                    });

    if (outstanding_key_value_cache_partial_token_generation_update) {
      // If there is any outstanding partial KV cache update, don't update the KV cache here.
    } else {
      key_value_cache_->Update(beam_indices, total_length);
    }
  }

  logits_.Update(next_tokens, new_length);
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
