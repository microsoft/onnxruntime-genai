#include "generator/generators.h"
#include "decoder_only.h"
#include "engram.h"

namespace Generators {
DecoderOnly_Model::DecoderOnly_Model(std::unique_ptr<Config> config, OrtEnv& ort_env)
    : Model{std::move(config)} {
  session_decoder_ = CreateSession(ort_env, config_->model.decoder.filename, session_options_.get());
  session_info_.Add(*session_decoder_);
  if (!config_->model.engram.filename.empty()) {
    engram_session_options_ = OrtSessionOptions::Create();
    CreateSessionOptionsFromConfig(
        config_->model.engram.session_options.value_or(config_->model.decoder.session_options),
        *engram_session_options_, true, /*disable_graph_capture=*/true);
    session_engram_ = CreateSession(ort_env, config_->model.engram.filename, engram_session_options_.get());
    session_info_.Add(*session_engram_);
  }
  if (!config_->model.embedding.filename.empty()) {
    if (!config_->engine.dynamic_batching) {
      throw std::runtime_error("Decoder-only CPU embedding requires the dynamic-batching Engine.");
    }
    cpu_embedding_ = std::make_shared<CpuEmbedding>(*this, ort_env);
    cpu_embedding_->ValidateConsumer(session_info_, config_->model.decoder.inputs.embeddings);
  }
}

std::unique_ptr<State> DecoderOnly_Model::CreateState(DeviceSpan<int32_t> sequence_lengths_unk, const GeneratorParams& params) const {
  if (cpu_embedding_) {
    throw std::runtime_error("Decoder-only CPU embedding requires Engine rather than Generator.");
  }
  return std::make_unique<DecoderOnly_State>(*this, sequence_lengths_unk, params);
}

void DecoderOnly_Model::InitializeIndexShare(const Config::Model::Mtp::IndexShare& config, OrtEnv&) {
  if (config.base_capacity == 0) return;
  SessionInfo extend_info;
  extend_info.Add(*session_decoder_);
  SessionInfo decode_info;
  decode_info.Add(*session_decoder_);
  const auto indices_shape = extend_info.GetOutputShape(config.indices_output);
  const auto counts_shape = extend_info.GetOutputShape(config.counts_output);
  const int64_t output_capacity = indices_shape.size() == 2 ? indices_shape[1] : 0;
  if (indices_shape.size() != 2 || output_capacity < config.base_capacity ||
      output_capacity > static_cast<int64_t>(config.base_capacity) + config.max_draft_tokens - 1 || counts_shape.size() != 1 ||
      extend_info.GetOutputDataType(config.indices_output) != ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32 ||
      extend_info.GetOutputDataType(config.counts_output) != ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32 ||
      decode_info.GetOutputDataType("indexshare.status") != ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32 ||
      decode_info.GetOutputShape("indexshare.status").size() != 1) {
    throw std::runtime_error("IndexShare graph outputs do not match their configured int32 selection contract.");
  }
  const std::vector<const char*> selection_inputs{
      config.indices_input.c_str(), config.counts_input.c_str(), "indexshare.base_row_indices",
      "indexshare.range_starts", "indexshare.range_ends"};
  for (const char* name : selection_inputs) {
    if (!decode_info.HasInput(name) || decode_info.GetInputDataType(name) != ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32) {
      throw std::runtime_error("IndexShare decode graph is missing a configured int32 input.");
    }
    const auto shape = decode_info.GetInputShape(name);
    const bool indices = std::string_view{name} == config.indices_input;
    if (shape.size() != (indices ? 2u : 1u) || (indices && shape[1] != config.base_capacity)) {
      throw std::runtime_error("IndexShare decode input shape does not match the configured selection contract.");
    }
  }
  {
    if (!decode_info.HasInput("indexshare.mode") || !decode_info.HasInput("indexshare.projection_rows") ||
        decode_info.GetInputDataType("indexshare.mode") != ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32 ||
        decode_info.GetInputShape("indexshare.mode") != std::vector<int64_t>{1} ||
        decode_info.GetInputDataType("indexshare.projection_rows") != ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64 ||
        decode_info.GetInputShape("indexshare.projection_rows").size() != 1) {
      throw std::runtime_error("IndexShare requires an int32[1] mode and int64 projection rows.");
    }
  }
  index_share_config_ = config;
  index_share_config_.max_draft_tokens = static_cast<int>(output_capacity - config.base_capacity + 1);
}

DecoderOnly_State::DecoderOnly_State(const DecoderOnly_Model& model, DeviceSpan<int32_t> sequence_lengths_unk, const GeneratorParams& params)
    : State{params, model},
      model_{model},
      kv_cache_(model_.p_device_kvcache_->CreateKeyValueCache(*this)),
      recurrent_state_(CreateRecurrentState(*this, /*graph_capture_variants_supported=*/true)),
      ple_state_(CreatePleState(*this)),
      indexer_cache_(CreateIndexerCache(*this)),
      position_inputs_{model_.p_device_inputs_->CreatePositionInputs(*this, sequence_lengths_unk, model_.config_->model.decoder.inputs.attention_mask)} {
  input_ids_.Add();
  position_inputs_->Add();
  logits_.Add();
  if (kv_cache_)
    kv_cache_->Add();
  if (recurrent_state_)
    recurrent_state_->Add();
  if (ple_state_)
    ple_state_->Add();
  if (indexer_cache_)
    indexer_cache_->Add();
  if (model_.session_engram_) {
    engram_state_ = std::make_unique<EngramState>(model_, *model_.session_engram_, params);
    const auto& name = model_.config_->model.decoder.inputs.engram_embeddings;
    engram_input_index_ = inputs_.size();
    inputs_.push_back(nullptr);
    input_names_.push_back(name.c_str());
  }
  // Models with a hidden_states input (e.g. the MTP self-speculative head) feed the main
  // model's last hidden state. Only created when the config declares the input.
  if (!model_.config_->model.decoder.inputs.hidden_states.empty()) {
    hidden_states_ = std::make_unique<HiddenStatesInputs>(*this);
    hidden_states_->Add();
  }
  // Models that emit a hidden_states output (exported with include_hidden_states, e.g. to feed
  // the MTP head) register it as a managed output so it survives CUDA-graph capture.
  if (!model_.config_->model.decoder.outputs.hidden_states.empty()) {
    hidden_states_output_ = std::make_unique<HiddenStatesOutputs>(*this);
    hidden_states_output_->Add();
  }
}

DecoderOnly_State::~DecoderOnly_State() = default;

void DecoderOnly_State::SetExtraInputs(const std::vector<ExtraInput>& extra_inputs) {
  extra_inputs_.Add(extra_inputs, model_.session_decoder_->GetInputNames());
}

void DecoderOnly_State::SetHiddenStates(OrtValue* hidden_states) {
  if (hidden_states_)
    hidden_states_->SetValue(hidden_states);
}

DeviceSpan<float> DecoderOnly_State::Run(int total_length, DeviceSpan<int32_t>& next_tokens, DeviceSpan<int32_t> next_indices) {
  size_t num_tokens = next_tokens.size();
  const auto& chunk_size_opt = params_->search.chunk_size;

  if (first_run_ && chunk_size_opt.has_value() && chunk_size_opt.value() > 0 && num_tokens > chunk_size_opt.value()) {
    return RunWithChunking(total_length, next_tokens, next_indices, chunk_size_opt.value());
  }

  UpdateInputsOutputs(next_tokens, next_indices, total_length);
  if (engram_state_) {
    engram_state_->UpdateInputsOutputs(next_tokens);
    engram_state_->Run(total_length, next_tokens, next_indices);
    auto shape = model_.session_info_.GetInputShape(model_.config_->model.decoder.inputs.engram_embeddings);
    if (shape.size() == 2)
      shape[0] = static_cast<int64_t>(num_tokens);
    else if (shape.size() == 3) {
      shape[0] = params_->BatchBeamSize();
      shape[1] = static_cast<int64_t>(num_tokens / params_->BatchBeamSize());
    } else
      throw std::runtime_error("Decoder Engram embeddings must be rank 2 or 3");
    engram_embeddings_ = OrtValue::CreateTensor(model_.p_device_inputs_->GetAllocator(), shape,
                                                model_.session_info_.GetInputDataType(model_.config_->model.decoder.inputs.engram_embeddings));
    inputs_[engram_input_index_] = engram_embeddings_.get();
    engram_state_->CopyEmbeddingsTo(*engram_embeddings_);
  }
  if (model_.config_->model.decoder.run_options.has_value()) {
    State::SetRunOptions(model_.config_->model.decoder.run_options.value());
  }

  // Graph capture enabled for token generation case, allowing it to repeat the same graph for each token.
  // MTP speculative decoding also captures the 2-token verify shape (max_graph_capture_length == 2),
  // each captured length getting its own annotation id / static buffers.
  const int seq_len = static_cast<int>(input_ids_.GetShape()[1]);
  const bool graph_capture_this_run =
      params_->use_graph_capture && seq_len >= 1 && seq_len <= params_->max_graph_capture_length;
  int graph_capture_variant = recurrent_state_ ? recurrent_state_->GraphCaptureVariant() : 0;
  if (ple_state_) {
    const int ple_graph_capture_variant = ple_state_->GraphCaptureVariant();
    if (recurrent_state_ && recurrent_state_->UsesGraphCaptureDoubleBuffer() &&
        graph_capture_variant != ple_graph_capture_variant)
      throw std::runtime_error("PLE and recurrent state graph-capture buffer variants are out of sync");
    graph_capture_variant = ple_graph_capture_variant;
  }

  // ORT captures by re-running the model inside this one Run(), which over-applies an
  // in-place recurrent state. Let the capture happen, then undo it and replay.
  const int graph_id = seq_len * 2 + graph_capture_variant;
  if (graph_capture_this_run && recurrent_state_ && recurrent_state_->ShouldFixUpGraphCapture(graph_id)) {
    recurrent_state_->SaveForGraphCapture();
    State::Run(*model_.session_decoder_, true, seq_len, graph_capture_variant);
    recurrent_state_->RestoreAfterGraphCapture(graph_id);
  }
  State::Run(*model_.session_decoder_, graph_capture_this_run, seq_len, graph_capture_variant);
  if (recurrent_state_) recurrent_state_->SetForwardLength(seq_len);
  if (ple_state_) ple_state_->SetForwardLength(seq_len);

  return logits_.Get();
}

DeviceSpan<float> DecoderOnly_State::RunWithChunking(int total_length, DeviceSpan<int32_t>& next_tokens,
                                                     DeviceSpan<int32_t> next_indices, size_t chunk_size) {
  // Chunking logic for context phase - process in chunks based on configured chunk_size
  size_t num_tokens = next_tokens.size();
  size_t processed_tokens = 0;
  int length = total_length - static_cast<int>(num_tokens);

  if (model_.config_->model.decoder.run_options.has_value()) {
    State::SetRunOptions(model_.config_->model.decoder.run_options.value());
  }
  while (processed_tokens < num_tokens) {
    size_t current_chunk_size = std::min(chunk_size, num_tokens - processed_tokens);

    // Create subspans for current chunk
    auto chunk_tokens = next_tokens.subspan(processed_tokens, current_chunk_size);
    length = length + static_cast<int>(current_chunk_size);

    // Process this chunk - fills KV cache progressively
    UpdateInputsOutputs(chunk_tokens, next_indices, length);
    if (engram_state_) {
      engram_state_->UpdateInputsOutputs(chunk_tokens);
      engram_state_->Run(length, chunk_tokens, next_indices);
      auto shape = model_.session_info_.GetInputShape(model_.config_->model.decoder.inputs.engram_embeddings);
      if (shape.size() == 2)
        shape[0] = static_cast<int64_t>(current_chunk_size);
      else if (shape.size() == 3) {
        shape[0] = params_->BatchBeamSize();
        shape[1] = static_cast<int64_t>(current_chunk_size / params_->BatchBeamSize());
      } else
        throw std::runtime_error("Decoder Engram embeddings must be rank 2 or 3");
      engram_embeddings_ = OrtValue::CreateTensor(model_.p_device_inputs_->GetAllocator(), shape,
                                                  model_.session_info_.GetInputDataType(model_.config_->model.decoder.inputs.engram_embeddings));
      inputs_[engram_input_index_] = engram_embeddings_.get();
      engram_state_->CopyEmbeddingsTo(*engram_embeddings_);
    }

    // Graph capture is typically disabled during context phase chunking
    bool graph_capture_this_run = false;  // Disable graph capture during chunking
    State::Run(*model_.session_decoder_, graph_capture_this_run);

    processed_tokens += current_chunk_size;
  }

  // Return logits from the last chunk for potential sampling
  return logits_.Get();
}

void DecoderOnly_State::RewindTo(size_t index) {
  if (engram_state_) engram_state_->RewindTo(index);
  position_inputs_->RewindTo(index);
  if (kv_cache_)
    kv_cache_->RewindTo(index);
  if (recurrent_state_)
    recurrent_state_->RewindTo(index);
  if (ple_state_)
    ple_state_->RewindTo(index);
  if (indexer_cache_)
    indexer_cache_->RewindTo(index);
}

void DecoderOnly_State::SnapshotState(size_t position) {
  if (engram_state_) engram_state_->SnapshotState(position);
  if (recurrent_state_)
    recurrent_state_->Snapshot(position);
  if (ple_state_)
    ple_state_->Snapshot(position);
  if (indexer_cache_)
    indexer_cache_->Snapshot(position);
}

bool DecoderOnly_State::HasCroppableRecurrentState() const {
  return recurrent_state_ && recurrent_state_->IsWindowed() && !ple_state_ && !engram_state_ &&
         (!indexer_cache_ || indexer_cache_->HasStateUpdates());
}

int64_t DecoderOnly_State::RecurrentStateWindow() const {
  return recurrent_state_ ? recurrent_state_->StateWindow() : 1;
}

void DecoderOnly_State::CropToAccepted(size_t new_length, size_t recurrent_position) {
  // Roll the attention KV cache + position back to the committed length, and crop the recurrent
  // state to the state AFTER verify token `recurrent_position` (no replay forward). Used by
  // lossless multi-token MTP partial accept.
  position_inputs_->RewindTo(new_length);
  if (kv_cache_)
    kv_cache_->RewindTo(new_length);
  if (recurrent_state_)
    recurrent_state_->CropToPosition(recurrent_position);
  if (ple_state_)
    ple_state_->CropToPosition(recurrent_position);
  if (indexer_cache_)
    indexer_cache_->CommitAcceptedPrefix(recurrent_position + 1);
}

void DecoderOnly_State::UpdateInputsOutputs(DeviceSpan<int32_t>& next_tokens, DeviceSpan<int32_t> beam_indices, int total_length) {
  input_ids_.Update(next_tokens);
  size_t new_length = static_cast<size_t>(input_ids_.GetShape()[1]);

  // Determine effective lengths for position_ids and KV cache based on sliding window config
  int position_length = total_length;
  int kv_cache_length = total_length;

  if (model_.config_->model.decoder.sliding_window.has_value() &&
      model_.config_->model.decoder.sliding_window->window_size > 0) {
    const int window_size = model_.config_->model.decoder.sliding_window->window_size;

    // Position IDs are clamped when slide_inputs is true
    if (model_.config_->model.decoder.sliding_window->slide_inputs) {
      position_length = std::min(total_length, window_size);
    }

    // KV cache is clamped when slide_key_value_cache is true
    if (model_.config_->model.decoder.sliding_window->slide_key_value_cache) {
      kv_cache_length = std::min(total_length, window_size);
    }
  }

  position_inputs_->Update(next_tokens, position_length, static_cast<int>(new_length));
  if (kv_cache_)
    kv_cache_->Update(beam_indices, kv_cache_length);
  if (recurrent_state_)
    recurrent_state_->Update();
  if (ple_state_)
    ple_state_->Update();
  if (indexer_cache_)
    indexer_cache_->Update(beam_indices, total_length, static_cast<int>(new_length));
  if (hidden_states_)
    hidden_states_->Update(static_cast<int>(new_length));
  if (hidden_states_output_)
    hidden_states_output_->Update(static_cast<int>(new_length));
  if (recurrent_state_)
    recurrent_state_->SetForwardLength(static_cast<int>(new_length));
  logits_.Update(next_tokens, new_length);
}

}  // namespace Generators
