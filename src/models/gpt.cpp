#include "../generators.h"
#include "gpt.h"

namespace Generators {

Gpt_Model::Gpt_Model(std::unique_ptr<Config> config, OrtEnv& ort_env)
    : Model{std::move(config)} {
  session_decoder_ = OrtSession::Create(ort_env, (config_->config_path / fs::path(config_->model.decoder.filename)).c_str(), session_options_.get());
  InitDeviceAllocator(*session_decoder_);
}

std::unique_ptr<State> Gpt_Model::CreateState(DeviceSpan<int32_t> sequence_lengths, const GeneratorParams& params) const {
  return std::make_unique<Gpt_State>(*this, sequence_lengths, params);
}

Gpt_State::Gpt_State(const Gpt_Model& model, DeviceSpan<int32_t> sequence_lengths_unk, const GeneratorParams& params)
    : State{params, model},
      model_{model},
      position_inputs_{model, *this, sequence_lengths_unk} {
  input_ids_.Add();
  position_inputs_.Add();
  logits_.Add();
  kv_cache_.Add();
  extra_inputs_.Add();
}

DeviceSpan<float> Gpt_State::Run(int total_length, DeviceSpan<int32_t>& next_tokens, DeviceSpan<int32_t> next_indices) {
  UpdateInputsOutputs(next_tokens, next_indices, total_length);

  int batch_size = static_cast<int>(input_ids_.GetShape()[0]);
  State::Run(*model_.session_decoder_, batch_size);

  return logits_.Get();
}

void Gpt_State::RewindTo(size_t index) {
  position_inputs_.RewindTo(index);
  kv_cache_.RewindTo(index);
}

void Gpt_State::UpdateInputsOutputs(DeviceSpan<int32_t>& next_tokens, DeviceSpan<int32_t> beam_indices, int total_length) {
  input_ids_.Update(next_tokens);
  size_t new_length = static_cast<size_t>(input_ids_.GetShape()[1]);
  position_inputs_.Update(next_tokens, total_length, static_cast<int>(new_length));
  kv_cache_.Update(beam_indices, total_length);
  logits_.Update(next_tokens, new_length);
}

}  // namespace Generators
