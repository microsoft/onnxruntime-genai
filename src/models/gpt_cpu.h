#include "gpt_common.h"
#include "model.h"
#include "kv_cache.h"

namespace Generators {

struct Gpt_State : State {

  Gpt_State(Gpt_Model& model, RoamingArray<int32_t> sequence_lengths, const SearchParams& search_params);
  RoamingArray<float> Run(int current_length, RoamingArray<int32_t> next_tokens, RoamingArray<int32_t> next_indices) override;

 private:
  void UpdateInputs(cpu_span<const int32_t> next_tokens, cpu_span<const int32_t> beam_indices, int current_length);

  const SearchParams& search_params_;
  bool first_run_{true};

  Gpt_Model* model_;
  Ort::Allocator& allocator_cpu_{Ort::Allocator::GetWithDefaultOptions()};
  KV_Cache_Combined kv_cache_;

  std::span<int32_t> next_positions_;  // shape (batch_size, num_beams). Next position value for position_ids.
  Ort::IAllocatorUniquePtr<int32_t> next_positions_buffer_;
  std::unique_ptr<OrtValue> next_positions_tensor_; // Tensor of the 'next_position_' buffer

  // Inputs
  std::unique_ptr<OrtValue> input_ids_, expanded_input_ids_;
  std::unique_ptr<OrtValue> position_ids_, expanded_position_ids_;
  std::unique_ptr<OrtValue> attention_mask_, expanded_attention_mask_;

  std::vector<const char *> input_names_;
  std::vector<OrtValue*> inputs_;

  // Outputs
  std::unique_ptr<OrtValue> logits_;
  std::vector<const char*> output_names_;
  std::vector<OrtValue*> outputs_;
};

}