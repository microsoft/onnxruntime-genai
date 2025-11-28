// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "../generators.h"
#include "../logging.h"
#include "../tracing.h"
#include "decoder_only_pipeline.h"
#include "windowed_kv_cache.h"
#include "vision_pipeline.h"
#include <unordered_set>
#include <limits>
#include <algorithm>

namespace Generators {

DecoderOnlyPipelineModel::DecoderOnlyPipelineModel(std::unique_ptr<Config> config, OrtEnv& ort_env)
    : Model{std::move(config)}, ort_env_{ort_env} {
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
      model_{model},
      input_ids_{CreateInputIDs(*this)},
      key_value_cache_{CreateKeyValueCache(*this)},
      do_key_value_cache_partial_update_{key_value_cache_ && key_value_cache_->IsPartialUpdateSupported()},
      position_inputs_{CreatePositionInputs(*this, sequence_lengths, model_.config_->model.decoder.inputs.attention_mask)} {
  
  // Check if model requires past_seq_len and total_seq_len inputs (for GQA-based sliding window models)
  has_past_seq_len_input_ = !model_.config_->model.decoder.inputs.past_sequence_length.empty() &&
                             model_.session_info_.HasInput(model_.config_->model.decoder.inputs.past_sequence_length);
  has_total_seq_len_input_ = !model_.config_->model.decoder.inputs.total_sequence_length.empty() &&
                              model_.session_info_.HasInput(model_.config_->model.decoder.inputs.total_sequence_length);
  
  if (has_past_seq_len_input_) {
    past_seq_len_tensor_ = std::make_unique<Tensor>(model_.p_device_inputs_, ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32);
    past_seq_len_tensor_->CreateTensor(std::array<int64_t, 2>{1, 1});
    *past_seq_len_tensor_->GetMutableData<int32_t>() = 0;  // Initialize to 0
    past_seq_len_input_index_ = static_cast<int>(inputs_.size());
    input_names_.push_back(model_.config_->model.decoder.inputs.past_sequence_length.c_str());
    inputs_.push_back(past_seq_len_tensor_->GetOrtTensor());
  }
  
  if (has_total_seq_len_input_) {
    total_seq_len_tensor_ = std::make_unique<Tensor>(model_.p_device_inputs_, ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32);
    total_seq_len_tensor_->CreateTensor(std::array<int64_t, 1>{1});
    // Use the KV cache actual size (window_size * 4 = 2048), not config context_length (4096)
    int kv_cache_size = model_.config_->model.decoder.sliding_window->window_size * 4;
    *total_seq_len_tensor_->GetMutableData<int32_t>() = kv_cache_size;
    total_seq_len_input_index_ = static_cast<int>(inputs_.size());
    input_names_.push_back(model_.config_->model.decoder.inputs.total_sequence_length.c_str());
    inputs_.push_back(total_seq_len_tensor_->GetOrtTensor());
  }
  
  input_ids_->Add();
  position_inputs_->Add();
  logits_.Add();
  if (key_value_cache_) {
    key_value_cache_->Add();
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
  // Check if we have vision inputs (pixel_values and image_grid_thw)
  // If so, process them through the vision pipeline
  OrtValue* pixel_values = nullptr;
  OrtValue* image_grid_thw = nullptr;
  
  for (const auto& extra_input : extra_inputs) {
    if (extra_input.name == model_.config_->model.vision.inputs.pixel_values) {
      pixel_values = extra_input.tensor->ort_tensor_.get();
    } else if (extra_input.name == model_.config_->model.vision.inputs.image_grid_thw) {
      image_grid_thw = extra_input.tensor->ort_tensor_.get();
    }
  }
  
  // If we have vision inputs, process them through the vision pipeline
  if (pixel_values != nullptr && image_grid_thw != nullptr && 
      !model_.config_->model.vision.pipeline.empty()) {
    // Create vision pipeline state and process the image
    VisionPipelineState vision_pipeline(model_, *params_);
    auto image_embeddings = vision_pipeline.ProcessImage(pixel_values, image_grid_thw);
    
    // Store the image embeddings for injection during generation
    SetImageEmbeddings(std::move(image_embeddings));
  }
  
  // Add extra inputs to all pipeline sessions (excluding vision inputs which are now processed)
  for (auto& session : model_.sessions_) {
    extra_inputs_.Add(extra_inputs, session->GetInputNames());
  }
}

void DecoderOnlyPipelineState::RunPipeline(int total_length, DeviceSpan<int32_t>& next_tokens,
                                           DeviceSpan<int32_t> next_indices, bool is_last_chunk) {
  for (auto& pipeline_state : pipeline_states_) {
    if (first_run_ && !model_.config_->model.decoder.pipeline[pipeline_state->id_].run_on_prompt) {
      continue;
    } else if (first_run_ && model_.config_->model.decoder.pipeline[pipeline_state->id_].is_lm_head && !is_last_chunk) {
      continue;
    } else if (!first_run_ && !model_.config_->model.decoder.pipeline[pipeline_state->id_].run_on_token_gen) {
      continue;
    }

    DurationTrace trace{MakeString("DecoderOnlyPipelineState::RunPipeline[", pipeline_state->id_, "]")};

    if (model_.config_->model.decoder.pipeline[pipeline_state->id_].reset_session_idx > -1) {
      if (model_.config_->model.decoder.pipeline[pipeline_state->id_].reset_session_idx >=
          static_cast<int>(model_.sessions_.size())) {
        throw std::runtime_error(
            MakeString("Invalid reset_session_idx ", model_.config_->model.decoder.pipeline[pipeline_state->id_].reset_session_idx,
                       " for pipeline model ", model_.config_->model.decoder.pipeline[pipeline_state->id_].model_id));
      }
      (const_cast<DecoderOnlyPipelineModel*>(&model_))->sessions_[model_.config_->model.decoder.pipeline[pipeline_state->id_].reset_session_idx].reset();
    }

    auto* const partial_kv_cache_update_record = [&]() -> PartialKeyValueCacheUpdateRecord* {
      auto it = pipeline_state_id_to_partial_kv_cache_update_record_idx_.find(pipeline_state->id_);
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

    // Add all the remaining outputs for the intermediate pipeline state
    for (const auto& output_name : model_.config_->model.decoder.pipeline[pipeline_state->id_].outputs) {
      if (std::none_of(pipeline_state->output_names_.begin(), pipeline_state->output_names_.end(),
                       [&](const std::string& elem) { return elem == output_name; })) {
        pipeline_state->output_names_.push_back(output_name.c_str());
        pipeline_state->outputs_.push_back(nullptr);
      }
    }

    // Run the intermediate pipeline state
    pipeline_state->Run(total_length, next_tokens, next_indices);

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
        
        // Check if this output is input_hidden_states and we have image embeddings to inject
        const std::string& output_name = pipeline_state->output_names_[i];
        if (output_name == model_.config_->model.decoder.inputs.input_hidden_states && 
            image_embeds_cache_ != nullptr && first_run_) {
          // Inject image embeddings into the hidden states
          // Use the original full input_ids (not windowed chunk) to correctly find image token positions
          if (!original_input_ids_.empty()) {
            InjectImageEmbeddings(*ortvalue_store_[output_name], original_input_ids_);
          }
        }
      }
    }
  }
}

DeviceSpan<float> DecoderOnlyPipelineState::Run(int total_length, DeviceSpan<int32_t>& next_tokens,
                                                DeviceSpan<int32_t> next_indices) {
  DurationTrace trace{"DecoderOnlyPipelineState::Run"};

  // first_run_ should be thought of as prompt_processing_run_.
  // It is true only for the prompt processing part when the provided tokens are more than 1.
  first_run_ = next_tokens.size() > 1;
  
  // Store original full input_ids BEFORE processing any chunks
  // This must happen before the pipeline runs so InjectImageEmbeddings can use it
  if (first_run_ && original_input_ids_.empty()) {
    // Find where real tokens start (skip padding tokens at the beginning)
    auto span = next_tokens.CpuSpan();
    size_t real_start = 0;
    
    // Qwen uses 151643 as pad token (EOS token used for padding)
    // Check first token to determine pad token ID
    int32_t potential_pad_token = span[0];
    
    // Find the first token that differs from the initial token
    // This assumes padding is homogeneous at the start
    for (size_t i = 1; i < span.size(); ++i) {
      if (span[i] != potential_pad_token) {
        real_start = i;
        break;
      }
    }
    
    // Store only the real (non-padded) tokens
    original_input_ids_.assign(span.begin() + real_start, span.end());
  }

  UpdateInputsOutputs(next_tokens, next_indices, total_length);
  
  size_t num_chunks{1};
  int window_size = 512;  // Default
  if (first_run_ && model_.config_->model.decoder.sliding_window.has_value()) {
    window_size = model_.config_->model.decoder.sliding_window->window_size;
    num_chunks = (next_tokens.size() + window_size - 1) / window_size;
  }

  for (size_t i = 0; i < num_chunks; ++i) {
    current_chunk_index_ = i;  // Track which chunk we're processing
    
    // Update past_seq_len and total_seq_len for sliding window GQA models
    if (first_run_ && (has_past_seq_len_input_ || has_total_seq_len_input_)) {
      // Correct past_seq_len: number of real (non-pad) tokens processed up to and including this chunk minus 1.
      // Previous logic over-counted on the final partial chunk (e.g., reported 2047 when only 1583 tokens exist).
      int real_prompt_length = static_cast<int>(original_input_ids_.size());
      int tokens_processed = static_cast<int>(std::min(static_cast<size_t>((i + 1) * window_size), original_input_ids_.size()));
      int past_seq_len = tokens_processed - 1;  // last valid index
      // KV cache capacity (window_size * 4) remains constant
      int total_seq_len = static_cast<int>(window_size * 4);

      if (has_past_seq_len_input_) {
        *past_seq_len_tensor_->GetMutableData<int32_t>() = past_seq_len;
        inputs_[past_seq_len_input_index_] = past_seq_len_tensor_->GetOrtTensor();
      }

      if (has_total_seq_len_input_) {
        *total_seq_len_tensor_->GetMutableData<int32_t>() = total_seq_len;
        inputs_[total_seq_len_input_index_] = total_seq_len_tensor_->GetOrtTensor();
      }

      // Debug: show corrected seq len feeding into model
      std::cout << "[SlidingWindow] chunk=" << i
                << " tokens_processed=" << tokens_processed
                << " past_seq_len=" << past_seq_len
                << " real_prompt_length=" << real_prompt_length
                << " total_seq_len(capacity)=" << total_seq_len << std::endl;
    }
    
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

  // For prompt processing with padding, manually set the correct sequence length
  // so that logits are extracted from the last real token, not the last padded token
  if (first_run_ && !original_input_ids_.empty() && model_.config_->model.decoder.sliding_window.has_value()) {
    // When add_generation_prompt=True is used, the prompt ends with:
    // "...user message</im_end>\n<|im_start|>assistant\n"
    // The model should predict the FIRST token of the assistant's response AFTER this prefix.
    // So we extract logits from the LAST token position (the final newline after "assistant").
    
    // Determine the last valid token position within the final window
    const size_t last_content_token_pos = original_input_ids_.size() - 1;
    const int win_size = model_.config_->model.decoder.sliding_window->window_size;  // typically 512
    const size_t final_window_start = (num_chunks - 1) * win_size;  // e.g., 1536 for window 3
    const size_t within_window_pos = last_content_token_pos - final_window_start; // e.g., 46

    // Effective sequence length is within_window_pos + 1
    const int effective_seq_len = static_cast<int>(within_window_pos + 1);

    for (int b = 0; b < params_->search.batch_size; b++) {
      // Ensure logits extractor respects the effective sequence length
      logits_.SetInputSequenceLength(b, effective_seq_len);
    }

    // Debug: print effective sequence info to compare indexing with baseline
    std::cout << "[Logits Indexing] last_content_token_pos=" << last_content_token_pos
          << ", final_window_start=" << final_window_start
          << ", within_window_pos=" << within_window_pos
          << ", effective_seq_len=" << effective_seq_len << std::endl;
  }
  
  first_run_ = false;

  return logits_.Get();
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

void DecoderOnlyPipelineState::SetImageEmbeddings(std::unique_ptr<OrtValue> image_embeds) {
  image_embeds_cache_ = std::move(image_embeds);
  img_emd_start_idx_ = 0;
  img_emd_end_idx_ = -1;
}

void DecoderOnlyPipelineState::InjectImageEmbeddings(OrtValue& input_embeds, const std::vector<int32_t>& input_ids) {
  if (!image_embeds_cache_ || model_.config_->model.image_token_id == 0) {
    return;  // No image embeddings to inject or no image token configured
  }

  // Removed broken debug stream output

  // Get the current window info from the input_embeds tensor
  auto input_embeds_info = input_embeds.GetTensorTypeAndShapeInfo();
  auto input_shape = input_embeds_info->GetShape();
  size_t window_size = static_cast<size_t>(input_shape[1]);  // Number of tokens in current window
  
  // Calculate the starting position in the full sequence for the current window
  size_t window_start = current_chunk_index_ * window_size;
  
  // Find image token positions in the FULL input_ids sequence
  std::vector<size_t> global_image_positions;
  for (size_t i = 0; i < input_ids.size(); ++i) {
    if (input_ids[i] == model_.config_->model.image_token_id) {
      global_image_positions.push_back(i);
    }
  }
  
  // Removed broken debug stream output
  if (!global_image_positions.empty()) {
  }
  
  if (global_image_positions.empty()) {
    return;  // No image tokens anywhere
  }

  // Filter to only image tokens that fall within the current window
  std::vector<std::pair<size_t, size_t>> window_image_positions;  // <window_relative_pos, global_image_index>
  for (size_t i = 0; i < global_image_positions.size(); ++i) {
    size_t global_pos = global_image_positions[i];
    if (global_pos >= window_start && global_pos < window_start + window_size) {
      size_t window_relative_pos = global_pos - window_start;
      window_image_positions.push_back({window_relative_pos, i});
    }
  }
  

  if (window_image_positions.empty()) {
    return;  // No image tokens in this window
  }

  // Get tensor data
  auto* input_data = input_embeds.GetTensorMutableData<float>();
  auto image_embeds_info = image_embeds_cache_->GetTensorTypeAndShapeInfo();
  auto image_shape = image_embeds_info->GetShape();
  auto* image_data = image_embeds_cache_->GetTensorData<float>();

  int64_t hidden_size = input_shape[2];
  int64_t total_image_tokens = image_shape[0];

  
  // Calculate end index for this chunk
  int num_image_positions = static_cast<int>(window_image_positions.size());
  img_emd_end_idx_ = img_emd_start_idx_ + num_image_positions;
  
  // Removed broken debug stream output

  // Inject image embeddings using sliding window approach (matching baseline)
  for (size_t i = 0; i < window_image_positions.size(); ++i) {
    auto [window_pos, global_image_idx] = window_image_positions[i];
    
    // Use sliding window index: img_emd_start_idx_ + i
    int image_embed_idx = img_emd_start_idx_ + static_cast<int>(i);
    
    if (image_embed_idx >= total_image_tokens) {
      break;  // Safety check
    }
    
    // Copy image embedding to input embedding at window-relative position
    std::memcpy(input_data + window_pos * hidden_size,
                image_data + image_embed_idx * hidden_size,
                hidden_size * sizeof(float));
  }

  // Update start index to track sliding window progress
  img_emd_start_idx_ = img_emd_end_idx_;
}

}  // namespace Generators
