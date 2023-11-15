#pragma once

namespace Generators {

struct Gpt_Model {
  Gpt_Model(OrtEnv& ort_env, const ORTCHAR_T* decoder_path);
#ifdef USE_CUDA
  Gpt_Model(OrtEnv& ort_env, const ORTCHAR_T* decoder_path, cudaStream_t cuda_stream);
  cudaStream_t cuda_stream_;
#endif

  DeviceType GetDeviceType() const { return device_type_; }
  int GetVocabSize() const { return vocab_size_; }

  std::unique_ptr<OrtSession> session_decoder_;

  // Model parameters:
  int vocab_size_{};
  int head_count_{};
  int hidden_size_{};
  int layer_count_{};
  bool logits_uses_seq_len_{};  // Logits shape is [... seq_len, vocab_size ] vs [... 1, vocab_size ]

 private:
  void InitModelParams();

  DeviceType device_type_;
};

}
