struct Gpt {
  static constexpr size_t c_counts=5;

  Gpt(OrtEnv &ort_env, const ORTCHAR_T* init_path, const ORTCHAR_T* decode_path,
      std::unique_ptr<OrtValue> &&input_ids, SearchParams params);

  void CreateInputs(gsl::span<int32_t> sequence_lengths);
  void CreateInputsInternal(gsl::span<int32_t> sequence_lengths);

  void Run();
  void UpdateInputs(gsl::span<const int32_t> next_tokens, OrtValue* position_ids, int num_beams, int current_length);

  SearchParams params_;

  bool past_present_share_buffer_ {};

  std::unique_ptr<OrtValue> initial_input_ids_;

  // Sessions
  std::unique_ptr<OrtSession> session_init_;
  std::unique_ptr<OrtSession> session_decode_;

  // Inputs
  std::unique_ptr<OrtValue> input_ids_, expanded_input_ids_;
  std::unique_ptr<OrtValue> position_ids_, expanded_position_ids_;
  std::unique_ptr<OrtValue> attention_mask_, expanded_attention_mask_;
  std::unique_ptr<OrtValue> empty_past_;
  std::unique_ptr<OrtValue> pasts_[c_counts];

  std::vector<std::string> input_name_strings_;
  std::vector<const char *> input_names_;
  std::vector<OrtValue*> inputs_;

  // Outputs
  std::unique_ptr<OrtValue> logits_;
  std::unique_ptr<OrtValue> presents_[c_counts];
  std::vector<std::string> output_name_strings_;
  std::vector<const char*> output_names_;
  std::vector<OrtValue*> outputs_;
};
