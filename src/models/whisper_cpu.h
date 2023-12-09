namespace Generators {
#if 0
struct Whisper_Model {
  Whisper_Model(OrtEnv& ort_env, Config& config, OrtSessionOptions& session_options);
#ifdef USE_CUDA
  cudaStream_t cuda_stream_{};
#endif

  DeviceType GetDeviceType() const { return device_type_; }

  std::unique_ptr<OrtSession> session_decoder_; // decoder.onnx
  std::unique_ptr<OrtSession> session_encoder_; // encoder_decoder_init.onnx

  // Model parameters:
  Config& config_;
  int vocab_size_{};
  int head_count_{};
  int hidden_size_{};
  int layer_count_{};
  bool logits_uses_seq_len_{};  // Logits shape is [... seq_len, vocab_size ] vs [... 1, vocab_size ]
  ONNXTensorElementDataType score_type_;

 private:
  void InitModelParams();

  DeviceType device_type_;
};


struct Whisper_State : State {
  Whisper_State(Whisper_Model& model, RoamingArray<int32_t> sequence_lengths, const SearchParams& params);
  RoamingArray<float> Run(int current_length, RoamingArray<int32_t> next_tokens, RoamingArray<int32_t> next_indices) override;

 private:
  void UpdateInputs(std::span<const int32_t> next_tokens, int current_length);

  SearchParams search_params_;
  bool first_run_{true};

  std::span<int64_t> next_positions_;  // shape (batch_size, num_beams). Next position value for position_ids.
  Ort::IAllocatorUniquePtr<int64_t> next_positions_buffer_;
  std::unique_ptr<OrtValue> next_positions_tensor_;  // Tensor of the 'next_position_' buffer

  // Model
  Whisper_Model* model_;

  // Inputs
  std::unique_ptr<OrtValue> encoder_input_ids_, expanded_encoder_input_ids_;
  std::unique_ptr<OrtValue> decoder_input_ids_, expanded_decoder_input_ids_;
  std::unique_ptr<OrtValue> empty_past_;
  std::vector<std::unique_ptr<OrtValue>> pasts_;

  std::vector<std::string> input_name_strings_;
  std::vector<const char*> input_names_;
  std::vector<OrtValue*> inputs_;

  // Outputs
  std::unique_ptr<OrtValue> logits_;
  std::vector<std::unique_ptr<OrtValue>> presents_;
  std::vector<std::string> output_name_strings_;
  std::vector<const char*> output_names_;
  std::vector<OrtValue*> outputs_;
};
#endif
}
