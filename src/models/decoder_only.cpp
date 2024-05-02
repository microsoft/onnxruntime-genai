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
      captured_graph_info_(model.GetCapturedGraphPool()->ReserveCapturedGraph(model, params)),
      position_inputs_{model, *this, sequence_lengths_unk} {
  input_ids_.Add();
  position_inputs_.Add();
  logits_.Add();
  kv_cache_.Add();
}

RoamingArray<float> DecoderOnly_State::Run(int current_length, RoamingArray<int32_t> next_tokens, RoamingArray<int32_t> next_indices) {
  if (first_run_) {
    if (params_->use_cuda_graph) {
      model_.run_options_->AddConfigEntry("gpu_graph_id", "-1");
    }
    first_run_ = false;
  } else {
    UpdateInputs(next_tokens, next_indices, current_length);
  }

  State::Run(*model_.session_decoder_, *model_.run_options_);

  // Set the graph id for the following runs.
  if (params_->use_cuda_graph) {
    int new_batch_size = static_cast<int>(input_ids_.GetShape()[0]);
    if (new_batch_size != current_batch_size_) {
      current_batch_size_ = new_batch_size;
      auto annotation_id = std::to_string(captured_graph_info_->GenerateUniqueAnnotationID(new_batch_size));
      model_.run_options_->AddConfigEntry("gpu_graph_id", annotation_id.c_str());
    }
  }
  return logits_.Get();
}

void DecoderOnly_State::UpdateInputs(const RoamingArray<int32_t>& next_tokens_unk, RoamingArray<int32_t> beam_indices, int current_length) {
  input_ids_.Update(next_tokens_unk);
  position_inputs_.Update(current_length);
  kv_cache_.Update(beam_indices.GetCPU(), current_length);
}

}  // namespace Generators
