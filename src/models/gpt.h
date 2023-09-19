#include "onnxruntime_cxx_api_2.h"

namespace Generators {

struct Gpt {

  static constexpr size_t c_vocab_size=1000;
  static constexpr size_t c_num_heads=4;
  static constexpr size_t c_head_size=8;
  static constexpr size_t c_counts=5;

  Gpt(OrtEnv &ort_env, const ORTCHAR_T* decode_path);

  void CreateInputs(std::span<int32_t> sequence_lengths, const SearchParams& params);
  std::span<const ScoreType> GetLogits();
  int GetVocabSize() { return c_vocab_size; }
  void Run(std::span<const int32_t> next_tokens, std::span<const int32_t> next_indices, int current_length);

private:
  void UpdateInputs(std::span<const int32_t> next_tokens, std::span<const int32_t> beam_indices, int current_length);
  void PickPastState(OrtAllocator& allocator, size_t index, std::span<const int32_t> beam_indices);

  SearchParams params_;
  bool first_run_{true};

  bool past_present_share_buffer_ {}; // NYI

  std::span<int32_t> next_positions_;  // shape (batch_size, num_beams). Next position value for position_ids.
  Ort::IAllocatorUniquePtr<int32_t> next_positions_buffer_;
  std::unique_ptr<OrtValue> next_positions_tensor_; // Tensor of the 'next_position_' buffer

  // Sessions
  std::unique_ptr<OrtSession> session_decode_;

  // Inputs
  std::unique_ptr<OrtValue> input_ids_, expanded_input_ids_;
  std::unique_ptr<OrtValue> position_ids_, expanded_position_ids_;
  std::unique_ptr<OrtValue> attention_mask_, expanded_attention_mask_;
  std::unique_ptr<OrtValue> empty_past_;
  std::unique_ptr<OrtValue> pasts_[c_counts];
  std::unique_ptr<OrtIoBinding> io_binding_decode_;

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

}