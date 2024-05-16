// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "../generators.h"
#include "multi_modal_vision_model.h"

namespace Generators {

namespace {

RoamingArray<float> MakeDummy() {
  return RoamingArray<float>();
}

std::unordered_set<std::string> GetInputNames(const OrtSession& session) {
  const auto input_names = session.GetInputNames();
  return {input_names.begin(), input_names.end()};
}

}  // namespace

MultiModalVisionModel::MultiModalVisionModel(std::unique_ptr<Config> config, OrtEnv& ort_env)
    : Model{std::move(config)} {
  embedding_session_ = OrtSession::Create(
      ort_env, (config_->config_path / config_->model.embedding.filename).c_str(), session_options_.get());
  vision_session_ = OrtSession::Create(
      ort_env, (config_->config_path / config_->model.vision.filename).c_str(), session_options_.get());
  decoder_session_ = OrtSession::Create(
      ort_env, (config_->config_path / config_->model.decoder.filename).c_str(), session_options_.get());

  InitDeviceAllocator(*decoder_session_);
  session_info_->Add(*embedding_session_);
  session_info_->Add(*vision_session_);
}

std::unique_ptr<State> MultiModalVisionModel::CreateState(RoamingArray<int32_t> sequence_lengths, const GeneratorParams& params) const {
  return std::make_unique<MultiModalPipelineState>(*this, sequence_lengths, params);
}

EmbeddingState::EmbeddingState(const MultiModalVisionModel& model, const GeneratorParams& params,
                               InputIDs&& input_ids, Embeddings&& embeddings)
    : State{params, GetInputNames(*model.embedding_session_)},
      model_{model},
      input_ids_{std::move(input_ids), *this},
      input_embeds_{std::move(embeddings), *this} {}

void EmbeddingState::UpdateInputsAndOutputs(RoamingArray<int32_t> next_tokens) {
  input_ids_.Update(next_tokens);
  input_embeds_.UpdateSequenceLength();
}

RoamingArray<float> EmbeddingState::Run(int current_length, RoamingArray<int32_t> next_tokens, RoamingArray<int32_t> next_indices) {
  State::Run(*model_.embedding_session_, *model_.run_options_);

  return MakeDummy();
}

VisionState::VisionState(const MultiModalVisionModel& model, const GeneratorParams& params)
    : State{params},
      model_{model} {
  input_ids_.Add();
  input_embeds_.Add();
}

RoamingArray<float> VisionState::Run(int current_length, RoamingArray<int32_t> next_tokens, RoamingArray<int32_t> next_indices) {
  State::Run(*model_.vision_session_, *model_.run_options_);

  return MakeDummy();
}

DecoderState::DecoderState(const MultiModalVisionModel& model, RoamingArray<int32_t> sequence_lengths, const GeneratorParams& params)
    : State{params, GetInputNames(*model.decoder_session_)},
      model_{model},
      position_inputs_{model, *this, sequence_lengths} {
  input_embeds_.Add();
  position_inputs_.Add();
  logits_.Add();
  kv_cache_.Add();
}

RoamingArray<float> DecoderState::Run(int current_length, RoamingArray<int32_t> next_tokens, RoamingArray<int32_t> next_indices) {
  State::Run(*model_.decoder_session_, *model_.run_options_);

  return logits_.Get();
}

void DecoderState::UpdateInputs(int current_length, RoamingArray<int32_t> beam_indices) {
  position_inputs_.Update(current_length);
  kv_cache_.Update(beam_indices.GetCPU(), current_length);
}

MultiModalPipelineState::MultiModalPipelineState(const MultiModalVisionModel& model,
                                                 RoamingArray<int32_t> sequence_lengths_unk,
                                                 const GeneratorParams& params)
    : State{params, {}},
      model_{model},
      vision_state_{std::make_unique<VisionState>(model_, params)},
      decoder_state_{std::make_unique<DecoderState>(model_, sequence_lengths_unk, params)} {
}

RoamingArray<float> MultiModalPipelineState::Run(int current_length, RoamingArray<int32_t> next_tokens,
                                                 RoamingArray<int32_t> next_indices) {
  // Pipeline state defines the pipeline of the execution of the models
  // Prompt stage:
  //   - input_ids, pixel_values, img_sizes -> |vision_model| -> |input_embeds|
  //   - input_embeds -> |decoder_model| -> |logits|
  // Generation stage:
  //   - input_ids -> |embeddings_model| -> |input_embeds|
  //   - input_embeds -> |decoder_model| -> |logits|
  if (is_prompt_) {
    vision_state_->Run(current_length, next_tokens, next_indices);

    decoder_state_->input_embeds_ = vision_state_->input_embeds_;
    auto logits = decoder_state_->Run(current_length, next_tokens, next_indices);

    is_prompt_ = false;
    // Reuse the vision state input_ids and input_embeds buffers going forward
    // in the embeddings state to avoid recreating them by moving them.
    embedding_state_ = std::make_unique<EmbeddingState>(model_, *params_,
                                                        std::move(vision_state_->input_ids_),
                                                        std::move(vision_state_->input_embeds_));
    vision_state_.reset();  // The vision state is no longer needed in generation stage

    return logits;
  }

  embedding_state_->UpdateInputsAndOutputs(next_tokens);
  decoder_state_->UpdateInputs(current_length, next_indices);

  embedding_state_->Run(current_length, next_tokens, next_indices);
  decoder_state_->input_embeds_ = embedding_state_->input_embeds_;
  return decoder_state_->Run(current_length, next_tokens, next_indices);
}

}  // namespace Generators
