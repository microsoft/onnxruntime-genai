#pragma once

namespace Generators {

struct Gpt_Model {
  Gpt_Model(Model& model, OrtEnv& ort_env, OrtSessionOptions& session_options);

  std::unique_ptr<OrtSession> session_decoder_;

  // Model parameters:
  Model& model_;
  int vocab_size_{};
  int head_count_{};
  int hidden_size_{};
  int layer_count_{};
  bool logits_uses_seq_len_{};  // Logits shape is [... seq_len, vocab_size ] vs [... 1, vocab_size ]
  ONNXTensorElementDataType score_type_;

 private:
  void InitModelParams();
};

}
