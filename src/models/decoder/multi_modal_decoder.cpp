#include "generator/generators.h"
#include "models/decoder/gemma4_decoder_state.h"
#include "models/decoder/multi_modal_decoder.h"
#include "models/multi_modal.h"

#include <algorithm>

namespace Generators {

DecoderModel::DecoderModel(std::unique_ptr<Config> config, OrtEnv& ort_env)
    : Model{std::move(config)} {
  session_decoder_ = CreateSession(ort_env, config_->model.decoder.filename, session_options_.get());
  session_info_.Add(*session_decoder_);
  if (!config_->model.embedding.filename.empty()) {
    if (!config_->engine.dynamic_batching) {
      throw std::runtime_error("Decoder-only CPU embedding requires the dynamic-batching Engine.");
    }
    cpu_embedding_ = std::make_shared<CpuEmbedding>(*this, ort_env);
    cpu_embedding_->ValidateConsumer(session_info_, config_->model.decoder.inputs.embeddings);
  }
}

std::unique_ptr<State> DecoderModel::CreateState(DeviceSpan<int32_t> sequence_lengths,
                                                     const GeneratorParams& params) const {
  if (cpu_embedding_) {
    throw std::runtime_error("Decoder-only CPU embedding requires Engine rather than Generator.");
  }
  return std::make_unique<DecoderState>(*this, sequence_lengths, params);
}

DecoderState::DecoderState(const GeneratorParams& params, const Model& model, OrtSession& session,
                           DeviceSpan<int32_t> sequence_lengths)
    : State{params, model},
      decoder_session_{session},
      position_inputs_{model.p_device_inputs_->CreatePositionInputs(
          *this, sequence_lengths, model.config_->model.decoder.inputs.attention_mask)},
      kv_cache_{model.p_device_kvcache_->CreateKeyValueCache(*this)},
      recurrent_state_{CreateRecurrentState(*this, /*graph_capture_variants_supported=*/true)} {}

DecoderState::DecoderState(const DecoderModel& model, DeviceSpan<int32_t> sequence_lengths,
                           const GeneratorParams& params)
    : DecoderState{params, model, *model.session_decoder_, sequence_lengths} {
  input_ids_ = std::make_unique<DefaultInputIDs>(*this);
  input_ids_->Add();
  Initialize();

  if (!model.config_->model.decoder.inputs.hidden_states.empty()) {
    hidden_states_ = std::make_unique<HiddenStatesInputs>(*this);
    hidden_states_->Add();
  }
  if (!model.config_->model.decoder.outputs.hidden_states.empty()) {
    hidden_states_output_ = std::make_unique<HiddenStatesOutputs>(*this);
    hidden_states_output_->Add();
  }
}

DecoderState::DecoderState(const MultiModalLanguageModel& model,
                           DeviceSpan<int32_t> sequence_lengths,
                           const GeneratorParams& params)
    : DecoderState{params, model, *model.decoder_session_, sequence_lengths} {
  inputs_embeds_ = std::make_unique<Embeddings>(
      *this, Embeddings::Mode::Input, model.config_->model.decoder.inputs.embeddings);
  inputs_embeds_->Add();

  SessionInfo decoder_only_info;
  decoder_only_info.Add(*model.decoder_session_);
  if (decoder_only_info.HasInput(model.config_->model.decoder.inputs.input_ids)) {
    input_ids_ = std::make_unique<DefaultInputIDs>(*this);
    input_ids_->Add();
  }
  Initialize();
}

void DecoderState::Initialize() {
  position_inputs_->Add();
  logits_.Add();
  if (kv_cache_)
    kv_cache_->Add();
  if (recurrent_state_)
    recurrent_state_->Add();
}

void DecoderState::SetExtraInputs(const std::vector<ExtraInput>& extra_inputs) {
  extra_inputs_.Add(extra_inputs, decoder_session_.GetInputNames());
}

void DecoderState::SetHiddenStates(OrtValue* hidden_states) {
  if (hidden_states_)
    hidden_states_->SetValue(hidden_states);
}

void DecoderState::ApplyRunOptions() {
  if (model_.config_->model.decoder.run_options.has_value()) {
    SetRunOptions(model_.config_->model.decoder.run_options.value());
  }
}

DeviceSpan<float> DecoderState::RunDecoder(int sequence_length, bool graph_capture_this_run) {
  const int graph_capture_variant = recurrent_state_ ? recurrent_state_->GraphCaptureVariant() : 0;
  const int graph_id = sequence_length * 2 + graph_capture_variant;
  if (graph_capture_this_run && recurrent_state_ &&
      recurrent_state_->ShouldFixUpGraphCapture(graph_id)) {
    recurrent_state_->SaveForGraphCapture();
    State::Run(decoder_session_, true, sequence_length, graph_capture_variant);
    recurrent_state_->RestoreAfterGraphCapture(graph_id);
  }
  State::Run(decoder_session_, graph_capture_this_run, sequence_length, graph_capture_variant);
  return logits_.Get();
}

DeviceSpan<float> DecoderState::Run(int current_length, DeviceSpan<int32_t>& next_tokens,
                                    DeviceSpan<int32_t> next_indices) {
  if (!UsesEmbeddings()) {
    const auto& chunk_size = params_->search.chunk_size;
    if (first_run_ && chunk_size.has_value() && chunk_size.value() > 0 &&
        next_tokens.size() > chunk_size.value()) {
      return RunWithChunking(current_length, next_tokens, next_indices, chunk_size.value());
    }
    UpdateInputsOutputs(next_tokens, current_length, next_indices);
  }

  ApplyRunOptions();
  const int sequence_length = static_cast<int>(
      UsesEmbeddings() ? inputs_embeds_->GetShape()[1] : input_ids_->GetShape()[1]);
  const bool graph_capture_this_run =
      params_->use_graph_capture &&
      (UsesEmbeddings()
           ? sequence_length == 1
           : sequence_length >= 1 && sequence_length <= params_->max_graph_capture_length);
  return RunDecoder(sequence_length, graph_capture_this_run);
}

DeviceSpan<float> DecoderState::RunWithChunking(
    int current_length, DeviceSpan<int32_t>& next_tokens,
    DeviceSpan<int32_t> next_indices, size_t chunk_size) {
  ApplyRunOptions();

  const size_t num_tokens = next_tokens.size();
  size_t processed_tokens = 0;
  int length = current_length - static_cast<int>(num_tokens);

  while (processed_tokens < num_tokens) {
    const size_t current_chunk_size = std::min(chunk_size, num_tokens - processed_tokens);
    auto chunk_tokens = next_tokens.subspan(processed_tokens, current_chunk_size);
    length += static_cast<int>(current_chunk_size);

    if (UsesEmbeddings()) {
      if (input_ids_)
        input_ids_->Update(chunk_tokens);
      position_inputs_->Update(chunk_tokens, length, static_cast<int>(current_chunk_size));
      kv_cache_->Update(next_indices, length);
      if (recurrent_state_)
        recurrent_state_->Update();
      logits_.Update(chunk_tokens, current_chunk_size);
      inputs_embeds_->UseChunkView(processed_tokens, current_chunk_size);
      UseExtraChunkView(processed_tokens, current_chunk_size);
    } else {
      UpdateInputsOutputs(chunk_tokens, length, next_indices);
    }

    State::Run(decoder_session_, /*graph_capture_this_run=*/false);
    processed_tokens += current_chunk_size;
  }

  if (UsesEmbeddings()) {
    inputs_embeds_->RestoreFullView();
    RestoreExtraFullView();
  }
  return logits_.Get();
}

bool DecoderState::SupportsPrefillChunking(bool has_multimodal_content) const {
  if (params_->BatchBeamSize() != 1)
    return false;
  return position_inputs_->SupportsSequentialPrefillChunking(has_multimodal_content);
}

void DecoderState::PrepareEmbeddingsForPrefill(size_t new_length) {
  inputs_embeds_->UpdateSequenceLength(new_length);
  UpdateExtraSequenceLength(new_length);
}

DeviceSpan<float> DecoderState::RunPrefillWithChunking(
    int current_length, DeviceSpan<int32_t>& next_tokens,
    DeviceSpan<int32_t> next_indices, size_t chunk_size) {
  return RunWithChunking(current_length, next_tokens, next_indices, chunk_size);
}

void DecoderState::UpdateInputsOutputs(DeviceSpan<int32_t>& next_tokens, int total_length,
                                       DeviceSpan<int32_t> beam_indices) {
  if (input_ids_)
    input_ids_->Update(next_tokens);

  size_t new_length;
  int position_length = total_length;
  int kv_cache_length = total_length;

  if (UsesEmbeddings()) {
    const int batch_size = static_cast<int>(inputs_embeds_->GetShape()[0]);
    new_length = next_tokens.size() / batch_size;
  } else {
    new_length = static_cast<size_t>(input_ids_->GetShape()[1]);
    const auto& sliding_window = model_.config_->model.decoder.sliding_window;
    if (sliding_window.has_value() && sliding_window->window_size > 0) {
      if (sliding_window->slide_inputs)
        position_length = std::min(total_length, sliding_window->window_size);
      if (sliding_window->slide_key_value_cache)
        kv_cache_length = std::min(total_length, sliding_window->window_size);
    }
  }

  position_inputs_->Update(next_tokens, position_length, static_cast<int>(new_length));
  if (kv_cache_)
    kv_cache_->Update(beam_indices, kv_cache_length);
  if (recurrent_state_)
    recurrent_state_->Update();
  if (hidden_states_)
    hidden_states_->Update(static_cast<int>(new_length));
  if (hidden_states_output_)
    hidden_states_output_->Update(static_cast<int>(new_length));
  if (recurrent_state_ && !UsesEmbeddings())
    recurrent_state_->SetForwardLength(static_cast<int>(new_length));
  logits_.Update(next_tokens, new_length);

  if (UsesEmbeddings()) {
    inputs_embeds_->UpdateSequenceLength(new_length);
    UpdateExtraSequenceLength(new_length);
  }
}

void DecoderState::RewindTo(size_t index) {
  position_inputs_->RewindTo(index);
  if (kv_cache_)
    kv_cache_->RewindTo(index);
  if (recurrent_state_)
    recurrent_state_->RewindTo(index);
}

void DecoderState::SnapshotState(size_t position) {
  if (recurrent_state_)
    recurrent_state_->Snapshot(position);
}

bool DecoderState::HasCroppableRecurrentState() const {
  return recurrent_state_ && recurrent_state_->IsWindowed();
}

int64_t DecoderState::RecurrentStateWindow() const {
  return recurrent_state_ ? recurrent_state_->StateWindow() : 1;
}

void DecoderState::CropToAccepted(size_t new_length, size_t recurrent_position) {
  position_inputs_->RewindTo(new_length);
  if (kv_cache_)
    kv_cache_->RewindTo(new_length);
  if (recurrent_state_)
    recurrent_state_->CropToPosition(recurrent_position);
}

std::unique_ptr<DecoderState> CreateDecoderState(
    const MultiModalLanguageModel& model, DeviceSpan<int32_t> sequence_lengths,
    const GeneratorParams& params) {
  if (!model.config_->model.decoder.inputs.per_layer_inputs.empty()) {
    return std::make_unique<Gemma4DecoderState>(model, sequence_lengths, params);
  }
  return std::make_unique<DecoderState>(model, sequence_lengths, params);
}

}  // namespace Generators
