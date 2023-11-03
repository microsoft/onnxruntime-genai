#include "onnxruntime_cxx_api_2.h"
#include "llama_common.h"

namespace Generators {

struct Llama_State {

  Llama_State(Llama_Model& model, std::span<int32_t> sequence_lengths, const SearchParams& params);
  std::span<ScoreType> Run(int current_length, std::span<const int32_t> next_tokens);

private:
  void UpdateInputs(std::span<const int32_t> next_tokens, int current_length);

  SearchParams search_params_;
  bool first_run_{true};

  std::span<int64_t> next_positions_;  // shape (batch_size, num_beams). Next position value for position_ids.
  Ort::IAllocatorUniquePtr<int64_t> next_positions_buffer_;
  std::unique_ptr<OrtValue> next_positions_tensor_; // Tensor of the 'next_position_' buffer

  // Model
  Llama_Model* model_;

  // Inputs
  std::unique_ptr<OrtValue> input_ids_, expanded_input_ids_;
  std::unique_ptr<OrtValue> position_ids_, expanded_position_ids_;
  std::unique_ptr<OrtValue> attention_mask_, expanded_attention_mask_;
  std::unique_ptr<OrtValue> empty_past_;
  std::vector<std::unique_ptr<OrtValue>> pasts_;

  std::vector<std::string> input_name_strings_;
  std::vector<const char *> input_names_;
  std::vector<OrtValue*> inputs_;

  // Outputs
  std::unique_ptr<OrtValue> logits_;
  std::vector<std::unique_ptr<OrtValue>> presents_;
  std::vector<std::string> output_name_strings_;
  std::vector<const char*> output_names_;
  std::vector<OrtValue*> outputs_;
};

}