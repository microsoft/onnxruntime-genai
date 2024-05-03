#include "model.h"
#include "input_ids.h"
#include "logits.h"
#include "kv_cache.h"

namespace Generators {

struct Whisper_Model : Model {
  Whisper_Model(std::unique_ptr<Config> config, OrtEnv& ort_env);

  std::unique_ptr<State> CreateState(RoamingArray<int32_t> sequence_lengths, const GeneratorParams& params) const override;

  std::unique_ptr<OrtSession> session_encoder_;  // encoder_decoder_init.onnx
  std::unique_ptr<OrtSession> session_decoder_;  // decoder.onnx

  std::unique_ptr<SessionInfo> session_encoder_info_;
};

struct Whisper_State : State {
  Whisper_State(const Whisper_Model& model, RoamingArray<int32_t> sequence_lengths, const GeneratorParams& params);
  RoamingArray<float> Run(int current_length, RoamingArray<int32_t> next_tokens, RoamingArray<int32_t> next_indices) override;

 private:
  void UpdateInputs(const RoamingArray<int32_t>& next_tokens, RoamingArray<int32_t> next_indices, int current_length);

  const Whisper_Model& model_;
  enum struct RunState {
    Encoder_Decoder_Init, // Next Run() will run this
    Decoder_First,
    Decoder,
  };

  RunState run_state_ {RunState::Encoder_Decoder_Init};

  std::unique_ptr<OrtValue> encoder_input_ids_;

  InputIDs decoder_input_ids_{model_, *this};
  Logits logits_{model_, *this};
  KV_Cache kv_cache_{model_, *this};
  Cross_Cache cross_cache_{model_, *this};
  std::unique_ptr<OrtValue> encoder_hidden_states_;

  std::unique_ptr<OrtValue> past_sequence_length_;
  std::unique_ptr<OrtValue> beam_width_;
  std::unique_ptr<OrtValue> cache_indirection_;

  // Temporary hack to have different sized outputs from the encoder that we then expand into the decoder buffers
  std::vector<std::unique_ptr<OrtValue>> init_presents_; // Hacked sized encoder_decoder_init presents
  std::vector<OrtValue*> presents_;  // The original present buffers we must resize init_presents_ into after the first run
};
}  // namespace Generators
