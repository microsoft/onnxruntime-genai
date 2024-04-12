#include "../generators.h"
#include "decoder_only.h"

namespace Generators {

DecoderOnly_Model::DecoderOnly_Model(std::unique_ptr<Config> config, OrtEnv& ort_env)
    : Model{std::move(config)} {
  session_decoder_ = OrtSession::Create(ort_env, (config_->config_path / config_->model.decoder.filename).c_str(), session_options_.get());

  InitDeviceAllocator(*session_decoder_);
}

std::unique_ptr<State> DecoderOnly_Model::CreateState(RoamingArray<int32_t> sequence_lengths, const GeneratorParams& params) const {
  return std::make_unique<DecoderOnly_State>(*this, sequence_lengths, params);
}

DecoderOnly_State::DecoderOnly_State(const DecoderOnly_Model& model, RoamingArray<int32_t> sequence_lengths_unk, const GeneratorParams& params)
    : State{params},
      model_{model},
      position_inputs_{model, *this, sequence_lengths_unk} {
  input_ids_.Add();
  position_inputs_.Add();
  logits_.Add();
  kv_cache_.Add();
}

RoamingArray<float> DecoderOnly_State::Run(int current_length, RoamingArray<int32_t> next_tokens, RoamingArray<int32_t> next_indices) {
  if (first_run_) {
    if (model_.use_cuda_graphs_) {
      model_.run_options_->AddConfigEntry("gpu_graph_id", "-1");
    }
    first_run_ = false;
  } else {
    UpdateInputs(next_tokens, next_indices, current_length);
  }

  State::Run(*model_.session_decoder_, *model_.run_options_);

  // Set the graph id for the following runs.
  if (model_.use_cuda_graphs_) {
    int new_graph_annotation_id = GetGraphAnnotationId();
    if (new_graph_annotation_id != graph_annotation_id_) {
      graph_annotation_id_ = new_graph_annotation_id;
      model_.run_options_->AddConfigEntry("gpu_graph_id", std::to_string(graph_annotation_id_).c_str());
    }
  }
  return logits_.Get();
}

void DecoderOnly_State::UpdateInputs(const RoamingArray<int32_t>& next_tokens_unk, RoamingArray<int32_t> beam_indices, int current_length) {
  input_ids_.Update(next_tokens_unk);
  position_inputs_.Update(current_length);
  kv_cache_.Update(beam_indices.GetCPU(), current_length);
}

int DecoderOnly_State::GetGraphAnnotationId() const {
  // Here we use the batch size as the graph annotation id.
  return static_cast<int>(input_ids_.GetShape()[0]);
}

}  // namespace Generators
