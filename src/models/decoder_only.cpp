#include "../generators.h"
#include "decoder_only.h"

namespace Generators {
DecoderOnly_Model::DecoderOnly_Model(std::unique_ptr<Config> config, OrtEnv& ort_env)
    : Model{std::move(config)} {
  session_decoder_ = OrtSession::Create(ort_env, (config_->config_path / fs::path(config_->model.decoder.filename)).c_str(), session_options_.get());

  InitDeviceAllocator(*session_decoder_);
}

std::unique_ptr<State> DecoderOnly_Model::CreateState(RoamingArray<int32_t> sequence_lengths, const GeneratorParams& params) const {
  return std::make_unique<DecoderOnly_State>(*this, sequence_lengths, params);
}

DecoderOnly_State::DecoderOnly_State(const DecoderOnly_Model& model, RoamingArray<int32_t> sequence_lengths_unk, const GeneratorParams& params)
    : State{params, model},
      model_{model},
      captured_graph_info_(model.GetCapturedGraphPool()->ReserveCapturedGraph(model, params)),
      position_inputs_{model, *this, sequence_lengths_unk} {
  input_ids_.Add();
  position_inputs_.Add();
  logits_.Add();
  kv_cache_.Add();
  extra_inputs_.Add();
}

RoamingArray<float> DecoderOnly_State::Run(int current_length, RoamingArray<int32_t> next_tokens, RoamingArray<int32_t> next_indices) {
  // TODO(aciddelgado): remove first_run
  if (!first_run_) {
    UpdateInputsOutputs(next_tokens, next_indices, current_length);
  }

  int batch_size = static_cast<int>(input_ids_.GetShape()[0]);
  State::Run(*model_.session_decoder_, *model_.run_options_, batch_size);

  return logits_.Get();
}

void DecoderOnly_State::UpdateInputsOutputs(const RoamingArray<int32_t>& next_tokens_unk, RoamingArray<int32_t> beam_indices, int current_length) {
  input_ids_.Update(next_tokens_unk);
  position_inputs_.Update(current_length);
  kv_cache_.Update(beam_indices.GetCPU(), current_length);
  logits_.Update();
}

// TODO(aciddelgado): make general
RoamingArray<float> DecoderOnly_State::Run(RoamingArray<int32_t> sequence, int next_token_length, int past_length, int return_last_logit_count) {
  int batch_size = static_cast<int>(input_ids_.GetShape()[0]);
  if (batch_size != 1)
    throw std::runtime_error("Speculative decoding only supports batch size 1, got " + std::to_string(batch_size));

  auto total_length = past_length + next_token_length;
  auto total_logits = first_run_ ? total_length : next_token_length; // TODO(aciddelgado): remove first_run
  // NB(bowenbao): workaround gqa limitation on token phase.
  // if (next_token_length > 1) {
  //   total_logits = total_length;
  // }
  UpdateInputsOutputsFromSequence(sequence, next_token_length, past_length);
  State::Run(*model_.session_decoder_, *model_.run_options_, batch_size);

  return logits_.Get(total_logits - return_last_logit_count, return_last_logit_count);
}

void DecoderOnly_State::UpdateInputsOutputsFromSequence(const RoamingArray<int32_t>& sequence, size_t next_token_length, int past_length) {
  auto total_length = past_length + next_token_length;
  if (g_log.enabled && g_log.continuous_decoding) {
    auto& stream = Log("continuous_decoding");
    stream << "UpdateInputsOutputsFromSequence: past_length=" << past_length << ", next_token_length=" << next_token_length << ", total_length=" << total_length << std::endl;
  }
  // TODO(aciddelgado): remove first_run
  if (first_run_) {
    // First run input ids includes prompt tokens.
    input_ids_.Update(sequence, 0, total_length);
    position_inputs_.Update(total_length, 0);
    kv_cache_.UpdatePresent(total_length);
    logits_.Update(total_length);
  } else {
    // Subsequent runs input ids only include candidate tokens.
    input_ids_.Update(sequence, past_length, next_token_length);
    position_inputs_.Update(total_length, past_length);
    kv_cache_.UpdateAndResize(total_length, past_length);
    logits_.Update(next_token_length);
  }
}

}  // namespace Generators
