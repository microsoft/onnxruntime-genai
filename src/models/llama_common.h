#pragma once

namespace Generators {

struct Llama_Model {
  Llama_Model(OrtEnv& ort_env, Config& config, OrtSessionOptions& session_options);
  cudaStream_t cuda_stream_{};

  DeviceType GetDeviceType() const { return device_type_; }

  std::unique_ptr<OrtSession> session_decoder_;

  // Model parameters:
  Config& config_;
  int vocab_size_{};
  int head_count_{};
  int hidden_size_{};
  int layer_count_{};
  bool logits_uses_seq_len_{};  // Logits shape is [... seq_len, vocab_size ] vs [... 1, vocab_size ]
  ONNXTensorElementDataType score_type_;

  std::array<const char*, 2> past_names_{"past_key_values.%d.key", "past_key_values.%d.value"};
  std::array<const char*, 2> present_names_{"present.%d.key", "present.%d.value"};

 private:
  void InitModelParams();

  DeviceType device_type_;
};

}
