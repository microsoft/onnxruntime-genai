#include "model.h"
#include "input_ids.h"
#include "logits.h"
#include "kv_cache.h"

namespace Generators {

struct Whisper_Model : Model {
  Whisper_Model(std::unique_ptr<Config> config, OrtEnv& ort_env, const ProviderOptions* provider_options);

  std::unique_ptr<State> CreateState(RoamingArray<int32_t> sequence_lengths, const SearchParams& params) override;

  std::unique_ptr<OrtSession> session_decoder_;  // decoder.onnx
  std::unique_ptr<OrtSession> session_encoder_;  // encoder_decoder_init.onnx

  std::array<const char*, 2> past_names_{"past_key_self_%d", "past_value_self_%d"};
  std::array<const char*, 2> present_names_{"present_key_self_%d", "present_value_self_%d"};
  std::array<const char*, 2> past_cross_names_{"past_key_cross_%d", "past_value_cross_%d"};
  std::array<const char*, 2> present_cross_names_{"present_key_cross_%d", "present_value_cross_%d"};

 private:
  void InitModelParams();
};

struct Whisper_State : State {
  Whisper_State(Whisper_Model& model, RoamingArray<int32_t> sequence_lengths, const SearchParams& params);
  RoamingArray<float> Run(int current_length, RoamingArray<int32_t> next_tokens, RoamingArray<int32_t> next_indices) override;

 private:
  void UpdateInputs(RoamingArray<int32_t> next_tokens, RoamingArray<int32_t> next_indices, int current_length);

  Whisper_Model& model_;
  bool first_run_{true};

  InputIDs<int32_t> decoder_input_ids_{model_, *this};
  Logits logits_{model_, *this};
  KV_Cache kv_cache_{model_, *this, model_.past_names_, model_.present_names_};
  Cross_Cache cross_cache_{model_, *this, model_.past_cross_names_, model_.present_cross_names_};
  std::unique_ptr<OrtValue> encoder_hidden_states_;
};
}  // namespace Generators
