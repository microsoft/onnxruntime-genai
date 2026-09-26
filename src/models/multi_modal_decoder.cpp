// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "generator/generators.h"
#include "multi_modal.h"
#include "multi_modal_decoder.h"
#include "gemma4_decoder_state.h"

#include <algorithm>

namespace Generators {

DecoderState::DecoderState(const MultiModalLanguageModel& model, DeviceSpan<int32_t> sequence_lengths, const GeneratorParams& params)
    : State{params, model},
      model_{model},
      inputs_embeds_{*this, Embeddings::Mode::Input, model.config_->model.decoder.inputs.embeddings},
      position_inputs_{model_.p_device_inputs_->CreatePositionInputs(*this, sequence_lengths, model_.config_->model.decoder.inputs.attention_mask)},
      kv_cache_{model_.p_device_kvcache_->CreateKeyValueCache(*this)},
      recurrent_state_{CreateRecurrentState(*this, /*graph_capture_variants_supported=*/true)} {
  inputs_embeds_.Add();

  // Some multimodal decoders (e.g., Gemma4) require input_ids alongside inputs_embeds.
  // Use a decoder-only SessionInfo to avoid false positives: the combined session_info_
  // includes embedding session inputs (which always has input_ids), causing this check
  // to incorrectly fire for models like mistral3 whose decoder has no input_ids input.
  {
    SessionInfo decoder_only_info;
    decoder_only_info.Add(*model_.decoder_session_);
    if (decoder_only_info.HasInput(model_.config_->model.decoder.inputs.input_ids)) {
      decoder_input_ids_ = std::make_unique<DefaultInputIDs>(*this);
      decoder_input_ids_->Add();
    }
  }

  position_inputs_->Add();
  logits_.Add();
  if (kv_cache_)
    kv_cache_->Add();
  if (recurrent_state_)
    recurrent_state_->Add();
}

DeviceSpan<float> DecoderState::Run(int current_length, DeviceSpan<int32_t>& next_tokens, DeviceSpan<int32_t> next_indices) {
  if (model_.config_->model.decoder.run_options.has_value()) {
    State::SetRunOptions(model_.config_->model.decoder.run_options.value());
  }

  const int seq_len = static_cast<int>(inputs_embeds_.GetShape()[1]);
  const bool graph_capture_this_run = params_->use_graph_capture && seq_len == 1;
  const int graph_capture_variant = recurrent_state_ ? recurrent_state_->GraphCaptureVariant() : 0;

  const int graph_id = seq_len * 2 + graph_capture_variant;
  if (graph_capture_this_run && recurrent_state_ && recurrent_state_->ShouldFixUpGraphCapture(graph_id)) {
    recurrent_state_->SaveForGraphCapture();
    State::Run(*model_.decoder_session_, true, seq_len, graph_capture_variant);
    recurrent_state_->RestoreAfterGraphCapture(graph_id);
  }
  State::Run(*model_.decoder_session_, graph_capture_this_run, seq_len, graph_capture_variant);
  return logits_.Get();
}

bool DecoderState::SupportsPrefillChunking(bool has_multimodal_content) const {
  // Chunking slices the pre-computed embeddings along the sequence dimension, which is only
  // contiguous for a single sequence. Continuous decoding of position ids/attention mask in
  // DefaultPositionInputs is likewise restricted to a batch-beam size of one.
  if (params_->BatchBeamSize() != 1)
    return false;

  // Delegate to the position-input implementation: DefaultPositionInputs always produces
  // sequential positions (chunking is always safe); Qwen-VL's 3D mRoPE position ids only reduce to
  // sequential positions for text-only prompts; other implementations keep the conservative
  // single-pass prefill behavior. See PositionInputs::SupportsSequentialPrefillChunking.
  return position_inputs_->SupportsSequentialPrefillChunking(has_multimodal_content);
}

void DecoderState::PrepareEmbeddingsForPrefill(size_t new_length) {
  // Allocate the embeddings buffers for the whole prompt. The embedding model writes into these
  // buffers in one run; the decoder then consumes them chunk by chunk.
  inputs_embeds_.UpdateSequenceLength(new_length);
  UpdateExtraSequenceLength(new_length);
}

DeviceSpan<float> DecoderState::RunPrefillWithChunking(int current_length, DeviceSpan<int32_t>& next_tokens,
                                                       DeviceSpan<int32_t> next_indices, size_t chunk_size) {
  if (model_.config_->model.decoder.run_options.has_value()) {
    State::SetRunOptions(model_.config_->model.decoder.run_options.value());
  }

  const size_t num_tokens = next_tokens.size();
  size_t processed_tokens = 0;
  int length = current_length - static_cast<int>(num_tokens);

  while (processed_tokens < num_tokens) {
    const size_t current_chunk_size = std::min(chunk_size, num_tokens - processed_tokens);
    auto chunk_tokens = next_tokens.subspan(processed_tokens, current_chunk_size);
    length += static_cast<int>(current_chunk_size);

    if (decoder_input_ids_) decoder_input_ids_->Update(chunk_tokens);
    position_inputs_->Update(chunk_tokens, length, static_cast<int>(current_chunk_size));
    kv_cache_->Update(next_indices, length);
    if (recurrent_state_)
      recurrent_state_->Update();
    logits_.Update(chunk_tokens, current_chunk_size);

    // Feed only this chunk's slice of the pre-computed embeddings to the decoder.
    inputs_embeds_.UseChunkView(processed_tokens, current_chunk_size);
    UseExtraChunkView(processed_tokens, current_chunk_size);

    // Graph capture is disabled during prefill chunking.
    State::Run(*model_.decoder_session_, /*graph_capture_this_run=*/false);

    processed_tokens += current_chunk_size;
  }

  inputs_embeds_.RestoreFullView();
  RestoreExtraFullView();

  // Logits of the last chunk contain the logits for the last prompt token.
  return logits_.Get();
}

void DecoderState::UpdateInputsOutputs(DeviceSpan<int32_t>& next_tokens, int total_length, DeviceSpan<int32_t> beam_indices) {
  int batch_size = static_cast<int>(inputs_embeds_.GetShape()[0]);
  size_t new_length = next_tokens.size() / batch_size;
  if (decoder_input_ids_) decoder_input_ids_->Update(next_tokens);
  position_inputs_->Update(next_tokens, total_length, static_cast<int>(new_length));
  if (kv_cache_)
    kv_cache_->Update(beam_indices, total_length);
  if (recurrent_state_)
    recurrent_state_->Update();
  logits_.Update(next_tokens, new_length);
  inputs_embeds_.UpdateSequenceLength(new_length);
  UpdateExtraSequenceLength(new_length);
}

// Overload for pipeline to call
void DecoderState::UpdateInputsOutputs(DeviceSpan<int32_t>& next_tokens, int total_length, DeviceSpan<int32_t> beam_indices, size_t new_length) {
  if (decoder_input_ids_) decoder_input_ids_->Update(next_tokens);
  if (kv_cache_)
    kv_cache_->Update(beam_indices, total_length);
  if (recurrent_state_)
    recurrent_state_->Update();
  logits_.Update(next_tokens, new_length);
  inputs_embeds_.UpdateSequenceLength(new_length);
  UpdateExtraSequenceLength(new_length);
}

std::unique_ptr<DecoderState> CreateDecoderState(const MultiModalLanguageModel& model, DeviceSpan<int32_t> sequence_lengths,
                                                 const GeneratorParams& params) {
  // Gemma4: the decoder accepts per_layer_inputs from the embedding model as additional per-layer
  // conditioning. Dispatch on the config field itself (rather than a model-name literal) so any
  // model whose decoder graph declares this input gets the subclass.
  if (!model.config_->model.decoder.inputs.per_layer_inputs.empty()) {
    return std::make_unique<Gemma4DecoderState>(model, sequence_lengths, params);
  }
  return std::make_unique<DecoderState>(model, sequence_lengths, params);
}

}  // namespace Generators
