// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "../generators.h"
#include "../logging.h"
#include "../tracing.h"
#include "decoder_only_pipeline.h"
#include "windowed_kv_cache.h"

#include <functional>
#include <unordered_set>

namespace Generators {

namespace {

// Splits a "session.tensor" dataflow endpoint into {session, tensor}. If no '.' is present the whole
// string is treated as the session and the tensor is empty.
std::pair<std::string, std::string> SplitEndpoint(const std::string& endpoint) {
  const auto dot = endpoint.find('.');
  if (dot == std::string::npos) {
    return {endpoint, std::string{}};
  }
  return {endpoint.substr(0, dot), endpoint.substr(dot + 1)};
}

// DFS-based cycle detection over the dataflow session graph (issue #2114 §3.2 guardrail).
void ThrowIfDataflowHasCycle(const std::unordered_map<std::string, std::vector<std::string>>& adjacency) {
  enum class Color { White, Gray, Black };
  std::unordered_map<std::string, Color> color;

  std::function<void(const std::string&)> visit = [&](const std::string& node) {
    color[node] = Color::Gray;
    if (auto it = adjacency.find(node); it != adjacency.end()) {
      for (const auto& next : it->second) {
        const Color next_color = color.count(next) ? color[next] : Color::White;
        if (next_color == Color::Gray) {
          throw std::runtime_error(
              MakeString("Pipeline dataflow contains a cycle involving session '", next,
                         "'. Dataflow wiring must be acyclic."));
        }
        if (next_color == Color::White) {
          visit(next);
        }
      }
    }
    color[node] = Color::Black;
  };

  for (const auto& [node, _] : adjacency) {
    if (!color.count(node) || color[node] == Color::White) {
      visit(node);
    }
  }
}

}  // namespace

std::string PipelineFlow::WireKey(std::string_view session, std::string_view input) {
  std::string key;
  key.reserve(session.size() + input.size() + 1);
  key.append(session);
  key.push_back('\n');  // '\n' cannot appear in a tensor/session name, so it is a safe separator.
  key.append(input);
  return key;
}

std::pair<std::string, std::string> PipelineFlow::SplitWire(const std::string& endpoint) {
  return SplitEndpoint(endpoint);
}

PipelineFlow::PipelineFlow(const Config& config) {
  const auto& decoder_pipeline = config.model.decoder.pipeline;
  const auto& pipeline = config.pipeline;

  // Max-10-stage guardrail (issue #2114 §3.2).
  const size_t stage_count = std::max(decoder_pipeline.size(), pipeline.flow.size());
  if (stage_count > 10) {
    throw std::runtime_error(
        MakeString("Pipeline exceeds the maximum of 10 stages (got ", stage_count, ")."));
  }

  // Map session name -> decoder.pipeline[] index. The session name is the stage model_id when set,
  // otherwise its filename (mirrors how TranslateV1ToPipeline names passthrough sessions).
  std::unordered_map<std::string, size_t> name_to_stage;
  for (size_t i = 0; i < decoder_pipeline.size(); ++i) {
    const auto& stage = decoder_pipeline[i];
    const std::string& name = !stage.model_id.empty() ? stage.model_id : stage.filename;
    name_to_stage.emplace(name, i);
  }

  // Default every stage to Step; flow steps refine init/final.
  stage_phase_.assign(decoder_pipeline.size(), Phase::Step);
  for (const auto& step : pipeline.flow) {
    Phase phase = Phase::Step;
    if (step.when == "init") {
      phase = Phase::Init;
    } else if (step.when == "final") {
      phase = Phase::Final;
    }
    if (auto it = name_to_stage.find(step.run); it != name_to_stage.end()) {
      stage_phase_[it->second] = phase;
      if (phase == Phase::Final) {
        has_final_ = true;
      }
    }
  }

  // Explicit dataflow wiring (overrides auto-match) + cycle detection.
  std::unordered_map<std::string, std::vector<std::string>> adjacency;
  for (const auto& wire : pipeline.dataflow) {
    const auto [from_session, from_tensor] = SplitWire(wire.from);
    const auto [to_session, to_tensor] = SplitWire(wire.to);
    explicit_wires_[WireKey(to_session, to_tensor)] = from_tensor;
    adjacency[from_session].push_back(to_session);
  }
  ThrowIfDataflowHasCycle(adjacency);
}

PipelineFlow::Phase PipelineFlow::PhaseForStage(size_t stage_id) const {
  return stage_id < stage_phase_.size() ? stage_phase_[stage_id] : Phase::Step;
}

bool PipelineFlow::IsFinal(size_t stage_id) const {
  return stage_id < stage_phase_.size() && stage_phase_[stage_id] == Phase::Final;
}

const std::string* PipelineFlow::ExplicitSource(const std::string& session, const std::string& input) const {
  auto it = explicit_wires_.find(WireKey(session, input));
  return it == explicit_wires_.end() ? nullptr : &it->second;
}

DecoderOnlyPipelineModel::DecoderOnlyPipelineModel(std::unique_ptr<Config> config, OrtEnv& ort_env)
    : Model{std::move(config)}, ort_env_{ort_env} {
  // Validate flow/dataflow guardrails (cycle, max-10-stage) at load time.
  PipelineFlow{*config_};

  for (const auto& model : config_->model.decoder.pipeline) {
    sessions_.emplace_back(CreateSession(ort_env, model.filename, GetSessionOptions(model.model_id)));
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

  if (model_.config_->model.decoder.pipeline[id_].run_options.has_value()) {
    State::SetRunOptions(model_.config_->model.decoder.pipeline[id_].run_options.value());
  }
  State::Run(*model_.sessions_[id_]);
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

static std::vector<size_t> GetLayerIndicesSetFromPastKeyNameInputs(
    const NameToLayerIdxMap& past_key_name_to_layer_idx, std::span<const std::string> inputs) {
  std::vector<size_t> layer_indices{};
  for (const auto& input_name : inputs) {
    const auto it = past_key_name_to_layer_idx.find(input_name);
    if (it != past_key_name_to_layer_idx.end()) {
      layer_indices.push_back(it->second);
    }
  }
  // sort and remove duplicates
  std::sort(layer_indices.begin(), layer_indices.end());
  layer_indices.erase(std::unique(layer_indices.begin(), layer_indices.end()),
                      layer_indices.end());
  return layer_indices;
}

DecoderOnlyPipelineState::DecoderOnlyPipelineState(const DecoderOnlyPipelineModel& model,
                                                   DeviceSpan<int32_t> sequence_lengths,
                                                   const GeneratorParams& params)
    : State{params, model},
      input_ids_{CreateInputIDs(*this)},
      model_{model},
      key_value_cache_{CreateKeyValueCache(*this)},
      do_key_value_cache_partial_update_{key_value_cache_ && key_value_cache_->IsPartialUpdateSupported()},
      recurrent_state_{CreateRecurrentState(*this)},
      position_inputs_{CreatePositionInputs(*this, sequence_lengths, model_.config_->model.decoder.inputs.attention_mask)} {
  input_ids_->Add();
  position_inputs_->Add();
  logits_.Add();

  flow_ = PipelineFlow{*model_.config_};
  if (key_value_cache_) {
    key_value_cache_->Add();
  }
  if (recurrent_state_) {
    recurrent_state_->Add();
  }

  const auto& config_pipeline = model_.config_->model.decoder.pipeline;

  for (size_t i = 0; i < config_pipeline.size(); ++i) {
    auto pipeline_model_state = std::make_unique<IntermediatePipelineState>(model_, params, pipeline_states_.size());
    pipeline_states_.emplace_back(std::move(pipeline_model_state));
  }

  if (do_key_value_cache_partial_update_) {
    const auto past_key_name_to_layer_idx = GeneratePastKeyNameToLayerIdxMap(*model_.config_);

    std::map<std::vector<size_t>, size_t> layer_indices_to_update_record_idx{};
    std::unordered_set<size_t> layer_indices_encountered{};

    for (size_t i = 0; i < config_pipeline.size(); ++i) {
      const auto& pipeline_model = config_pipeline[i];

      const auto layer_indices = GetLayerIndicesSetFromPastKeyNameInputs(past_key_name_to_layer_idx,
                                                                         pipeline_model.inputs);

      if (layer_indices.empty()) {
        continue;
      }

      size_t record_idx{};

      if (auto layer_indices_to_update_record_it = layer_indices_to_update_record_idx.find(layer_indices);
          layer_indices_to_update_record_it != layer_indices_to_update_record_idx.end()) {
        // we have seen this exact set of layer indices before. reuse the existing record.
        record_idx = layer_indices_to_update_record_it->second;
      } else {
        // verify that the new set of layer indices is valid.
        // i.e., it is disjoint with the set of all layer indices we've seen so far.
        const bool layer_indices_valid =
            std::all_of(layer_indices.begin(), layer_indices.end(),
                        [&layer_indices_encountered](size_t layer_idx) {
                          return layer_indices_encountered.find(layer_idx) == layer_indices_encountered.end();
                        });

        if (!layer_indices_valid) {
          throw std::runtime_error(
              "Invalid layer indices. Layer index sets for partial key value cache update must be either an exact "
              "match with another set or disjoint with all other sets.");
        }

        // add a new record
        auto record = PartialKeyValueCacheUpdateRecord{};
        record.layer_indices = layer_indices;

        partial_kv_cache_update_records_.emplace_back(std::move(record));
        record_idx = partial_kv_cache_update_records_.size() - 1;

        // add layer_indices to what we've seen so far
        layer_indices_encountered.insert(layer_indices.begin(), layer_indices.end());
        layer_indices_to_update_record_idx.emplace(layer_indices, record_idx);
      }

      pipeline_state_id_to_partial_kv_cache_update_record_idx_.emplace(i, record_idx);
    }

    if (!partial_kv_cache_update_records_.empty()) {
      key_value_cache_update_worker_thread_.emplace();
    }
  }
}

void DecoderOnlyPipelineState::SetExtraInputs(const std::vector<ExtraInput>& extra_inputs) {
  for (auto& session : model_.sessions_) {
    extra_inputs_.Add(extra_inputs, session->GetInputNames());
  }
}

void DecoderOnlyPipelineState::RunPipeline(int total_length, DeviceSpan<int32_t>& next_tokens,
                                           DeviceSpan<int32_t> next_indices, bool is_last_chunk) {
  for (auto& pipeline_state : pipeline_states_) {
    const size_t id = pipeline_state->id_;

    // Final-phase stages run once after the generation loop (see Finalize), never in the per-token loop.
    if (flow_.IsFinal(id)) {
      continue;
    }

    // Init-vs-step gating is preserved exactly via the run_on_prompt/run_on_token_gen fields that PR1
    // lowering populates (kept consistent with the resolved flow), so v1 behavior is unchanged.
    if (first_run_ && !model_.config_->model.decoder.pipeline[id].run_on_prompt) {
      continue;
    } else if (first_run_ && model_.config_->model.decoder.pipeline[id].is_lm_head && !is_last_chunk) {
      continue;
    } else if (!first_run_ && !model_.config_->model.decoder.pipeline[id].run_on_token_gen) {
      continue;
    }

    RunStage(*pipeline_state, total_length, next_tokens, next_indices);
  }
}

void DecoderOnlyPipelineState::RunStage(IntermediatePipelineState& pipeline_state, int total_length,
                                        DeviceSpan<int32_t>& next_tokens, DeviceSpan<int32_t> next_indices) {
  DurationTrace trace{MakeString("DecoderOnlyPipelineState::RunPipeline[", pipeline_state.id_, "]")};

  const auto& stage_config = model_.config_->model.decoder.pipeline[pipeline_state.id_];
  const std::string& stage_session = !stage_config.model_id.empty() ? stage_config.model_id : stage_config.filename;

  if (stage_config.reset_session_idx > -1) {
    if (stage_config.reset_session_idx >= static_cast<int>(model_.sessions_.size())) {
      throw std::runtime_error(
          MakeString("Invalid reset_session_idx ", stage_config.reset_session_idx,
                     " for pipeline model ", stage_config.model_id));
    }
    (const_cast<DecoderOnlyPipelineModel*>(&model_))->sessions_[stage_config.reset_session_idx].reset();
  }

  auto* const partial_kv_cache_update_record = [&]() -> PartialKeyValueCacheUpdateRecord* {
    auto it = pipeline_state_id_to_partial_kv_cache_update_record_idx_.find(pipeline_state.id_);
    if (it != pipeline_state_id_to_partial_kv_cache_update_record_idx_.end()) {
      return &partial_kv_cache_update_records_[it->second];
    }
    return nullptr;
  }();

  // If there is any outstanding partial KV cache update, wait for it to finish.
  // It is important to synchronize at this point, before setting input/output tensors for this pipeline state run,
  // because a KV cache update may replace the KV cache input/output tensors.
  if (partial_kv_cache_update_record) {
    if (partial_kv_cache_update_record->outstanding_update.valid()) {
      partial_kv_cache_update_record->outstanding_update.get();
    }
  }

  // Clear the intermediate pipeline state outputs from the previous runs.
  // These outputs will be replaced by the outputs from the current run.
  for (const auto& output_name : pipeline_state.output_names_) {
    if (auto iter = ortvalue_store_.find(output_name); iter != ortvalue_store_.end()) {
      ortvalue_store_.erase(iter);
    }
  }
  pipeline_state.ClearIO();

  // Managed inputs and outputs are those inputs and outputs that the
  // Model knows how to create and update from one run to the next.

  // Add all the managed inputs to the intermediate pipeline state
  for (const auto& input_name : input_names_) {
    if (pipeline_state.HasInput(input_name)) {
      if (!pipeline_state.SupportsPrimaryDevice()) {
        throw std::runtime_error(
            MakeString("Managed input ", input_name, " resides on the primary device type (",
                       static_cast<int>(model_.p_device_->GetType()), "). But the pipeline model ",
                       stage_config.model_id,
                       " is expecting it to reside elsewhere."));
      }
      pipeline_state.input_names_.push_back(input_name);
      pipeline_state.inputs_.push_back(State::GetInput(input_name));
    }
  }

  // Explicit dataflow wiring (issue #2114 §3.3) overrides the name-based auto-match below. For each
  // declared input of this stage that an explicit `dataflow[]` wire targets, bind it from the wired
  // producer's stored output tensor (which may have a different name than the input).
  std::unordered_set<std::string> explicitly_bound_inputs;
  for (const auto& input_name : stage_config.inputs) {
    if (const std::string* from_tensor = flow_.ExplicitSource(stage_session, input_name)) {
      if (auto iter = ortvalue_store_.find(*from_tensor); iter != ortvalue_store_.end()) {
        pipeline_state.input_names_.push_back(input_name.c_str());
        pipeline_state.inputs_.push_back(iter->second.get());
        explicitly_bound_inputs.insert(input_name);
      }
    }
  }

  // Add outputs from the previous pipeline states to the current pipeline state (auto-match by name).
  // Skip inputs already satisfied by an explicit dataflow wire (explicit wires win).
  for (auto& [name, ortvalue] : ortvalue_store_) {
    if (pipeline_state.HasInput(name) && explicitly_bound_inputs.find(name) == explicitly_bound_inputs.end()) {
      pipeline_state.input_names_.push_back(name.c_str());
      pipeline_state.inputs_.push_back(ortvalue.get());
    }
  }

  // Add all the managed outputs to the intermediate pipeline state
  for (const auto& output_name : output_names_) {
    if (pipeline_state.HasOutput(output_name)) {
      if (!pipeline_state.SupportsPrimaryDevice()) {
        throw std::runtime_error(
            MakeString("Managed output ", output_name, " resides on the primary device type (",
                       static_cast<int>(model_.p_device_->GetType()), "). But the pipeline model ",
                       stage_config.model_id,
                       " is expecting it to reside elsewhere."));
      }
      pipeline_state.output_names_.push_back(output_name);
      pipeline_state.outputs_.push_back(State::GetOutput(output_name));
    }
  }

  // Output of pipeline models could also be managed inputs.
  // For example, the output of a pipeline model could be the key-value cache.
  // In such cases, use the managed output buffers and register them with the pipeline model as outputs.
  for (const auto& input_name : input_names_) {
    if (pipeline_state.HasOutput(input_name)) {
      if (!pipeline_state.SupportsPrimaryDevice()) {
        throw std::runtime_error(
            MakeString("Managed input ", input_name, " resides on the primary device type (",
                       static_cast<int>(model_.p_device_->GetType()), "). But the pipeline model ",
                       stage_config.model_id,
                       " is expecting it to reside elsewhere."));
      }
      pipeline_state.output_names_.push_back(input_name);
      pipeline_state.outputs_.push_back(State::GetInput(input_name));
    }
  }

  // Add all the remaining outputs for the intermediate pipeline state
  for (const auto& output_name : stage_config.outputs) {
    if (std::none_of(pipeline_state.output_names_.begin(), pipeline_state.output_names_.end(),
                     [&](const std::string& elem) { return elem == output_name; })) {
      pipeline_state.output_names_.push_back(output_name.c_str());
      pipeline_state.outputs_.push_back(nullptr);
    }
  }

  // Run the intermediate pipeline state
  pipeline_state.Run(total_length, next_tokens, next_indices);

  // If there is any partial KV cache update to start, enqueue it.
  if (partial_kv_cache_update_record) {
    assert(key_value_cache_update_worker_thread_.has_value());
    auto update_fn = [&key_value_cache = *key_value_cache_.get(),
                      layer_indices = partial_kv_cache_update_record->layer_indices,
                      next_indices, total_length]() {
      key_value_cache.PartialUpdate(next_indices, total_length, layer_indices);
    };
    partial_kv_cache_update_record->outstanding_update = key_value_cache_update_worker_thread_->Enqueue(update_fn);
  }

  // Transfer ownership of all the non-managed outputs from the current pipeline state to the ortvalue store.
  // All non managed outputs are assumed to be on CPU
  for (size_t i = 0; i < pipeline_state.output_names_.size(); ++i) {
    if (std::none_of(output_names_.begin(), output_names_.end(),
                     [&](const std::string& elem) { return elem == pipeline_state.output_names_[i]; }) &&
        std::none_of(input_names_.begin(), input_names_.end(),
                     [&](const std::string& elem) { return elem == pipeline_state.output_names_[i]; })) {
      auto forwarded_output = stage_config.output_names_forwarder.find(pipeline_state.output_names_[i]);
      if (forwarded_output != stage_config.output_names_forwarder.end()) {
        ortvalue_store_[forwarded_output->second] = std::unique_ptr<OrtValue>(pipeline_state.outputs_[i]);
      } else {
        ortvalue_store_[pipeline_state.output_names_[i]] = std::unique_ptr<OrtValue>(pipeline_state.outputs_[i]);
      }
    }
  }

  // Notify derived classes that this pipeline stage has completed.
  // This allows e.g. Qwen VL to inject vision embeddings after the embeddings stage.
  OnStageComplete(pipeline_state.id_);
}

DeviceSpan<float> DecoderOnlyPipelineState::Run(int total_length, DeviceSpan<int32_t>& next_tokens,
                                                DeviceSpan<int32_t> next_indices) {
  DurationTrace trace{"DecoderOnlyPipelineState::Run"};

  UpdateInputsOutputs(next_tokens, next_indices, total_length);

  // first_run_ should be thought of as prompt_processing_run_.
  // It is true only for the prompt processing part when the provided tokens are more than 1.
  first_run_ = next_tokens.size() > 1;
  size_t num_chunks{1};
  if (first_run_ && model_.config_->model.decoder.sliding_window.has_value()) {
    int window_size = model_.config_->model.decoder.sliding_window->window_size;
    num_chunks = (next_tokens.size() + window_size - 1) / window_size;
  }

  for (size_t i = 0; i < num_chunks; ++i) {
    RunPipeline(total_length, next_tokens, next_indices, (i == num_chunks - 1));

    if (model_.config_->model.decoder.sliding_window.has_value() && i < num_chunks - 1) {
      // Sliding the window over the input_ids, key_cache, and value_cache, position_ids, and attention_mask
      input_ids_->Update(next_tokens);
      UpdateKeyValueCache(next_indices, total_length);
      position_inputs_->Update(next_tokens, total_length, static_cast<int>(input_ids_->GetShape()[1]));
      logits_.Update(WrapTensor<int32_t>(*model_.p_device_inputs_, *input_ids_->Get()),
                     static_cast<int>(input_ids_->GetShape()[1]));
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

  // Remember the managed indices/length so a post-loop Finalize() can replay them into final stages.
  last_next_indices_ = next_indices;
  last_total_length_ = total_length;

  return logits_.Get();
}

void DecoderOnlyPipelineState::Finalize(int current_length) {
  // Execute flow stages whose `when=="final"` exactly once after the generation loop completes.
  // No in-tree model declares final stages today, so this is a no-op for all current models.
  if (!flow_.HasFinalStages()) {
    return;
  }

  DurationTrace trace{"DecoderOnlyPipelineState::Finalize"};

  const int total_length = last_total_length_ != 0 ? last_total_length_ : current_length;
  DeviceSpan<int32_t> next_tokens{};
  for (auto& pipeline_state : pipeline_states_) {
    if (flow_.IsFinal(pipeline_state->id_)) {
      RunStage(*pipeline_state, total_length, next_tokens, last_next_indices_);
    }
  }
}

void DecoderOnlyPipelineState::RewindTo(size_t index) {
  // Before mutating cache state, drain any outstanding asynchronous partial KV cache updates.
  // A partial update worker may otherwise write into KV cache tensors after we rewind them,
  // racing with and corrupting the rolled-back state.
  for (auto& record : partial_kv_cache_update_records_) {
    if (record.outstanding_update.valid()) {
      record.outstanding_update.get();
    }
  }

  // Mirror DecoderOnly_State::RewindTo: roll back the position inputs and, when present, the
  // KV cache and recurrent state. If an owned cache cannot be rewound (e.g. LFM2Cache), its
  // RewindTo throws a clear error rather than leaving the pipeline in a corrupt state.
  position_inputs_->RewindTo(index);
  if (key_value_cache_)
    key_value_cache_->RewindTo(index);
  if (recurrent_state_)
    recurrent_state_->RewindTo(index);
}

void DecoderOnlyPipelineState::UpdateKeyValueCache(DeviceSpan<int32_t> beam_indices, int total_length) {
  if (key_value_cache_) {
    const bool outstanding_key_value_cache_partial_update =
        do_key_value_cache_partial_update_ &&
        std::any_of(partial_kv_cache_update_records_.rbegin(),
                    partial_kv_cache_update_records_.rend(),
                    [](const PartialKeyValueCacheUpdateRecord& record) {
                      return record.outstanding_update.valid();
                    });

    if (outstanding_key_value_cache_partial_update) {
      // If there is any outstanding partial KV cache update, don't update the KV cache here.
    } else {
      key_value_cache_->Update(beam_indices, total_length);
    }
  }
}

void DecoderOnlyPipelineState::UpdateInputsOutputs(DeviceSpan<int32_t>& next_tokens,
                                                   DeviceSpan<int32_t> beam_indices, int total_length) {
  input_ids_->Update(next_tokens);
  size_t new_length = input_ids_->GetShape()[1];
  position_inputs_->Update(next_tokens, total_length, static_cast<int>(new_length));
  UpdateKeyValueCache(beam_indices, total_length);
  if (recurrent_state_) {
    recurrent_state_->Update();
  }

  auto next_windowed_tokens = WrapTensor<int32_t>(*model_.p_device_inputs_, *input_ids_->Get());
  logits_.Update(next_windowed_tokens, new_length);
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
