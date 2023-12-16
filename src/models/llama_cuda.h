#include "llama_common.h"
#include "model.h"
#include "kv_cache.h"

namespace Generators {

struct Llama_Cuda : State {

  Llama_Cuda(Llama_Model& model, RoamingArray<int32_t> sequence_lengths, const SearchParams& params);
  RoamingArray<float> Run(int current_length, RoamingArray<int32_t> next_tokens, RoamingArray<int32_t> next_indices) override;

private:
  void UpdateInputs(std::span<const int32_t> next_tokens, std::span<const int32_t> beam_indices, int current_length);

  Llama_Model* model_;
  const SearchParams& search_params_;
  bool first_run_{true};

  Ort::Allocator& allocator_cpu_;
  std::unique_ptr<OrtMemoryInfo> memory_info_cuda_;
  std::unique_ptr<Ort::Allocator> allocator_cuda_;
  KV_Cache kv_cache_;

  std::span<int64_t> next_positions_;  // shape (batch_size, num_beams). Next position value for position_ids.
  Ort::IAllocatorUniquePtr<int64_t> next_positions_buffer_;
  std::unique_ptr<OrtValue> next_positions_tensor_; // Tensor of the 'next_position_' buffer

  // Inputs
  std::unique_ptr<OrtValue> input_ids_, expanded_input_ids_;
  std::unique_ptr<OrtValue> position_ids_, expanded_position_ids_;
  std::unique_ptr<OrtValue> attention_mask_, expanded_attention_mask_;

  std::vector<const char *> input_names_;
  std::vector<OrtValue*> inputs_;

  // Outputs
  std::unique_ptr<OrtValue> logits_;
  std::unique_ptr<OrtValue> logits32_; // When model output is fp16, this holds the fp32 conversion of them
  std::vector<const char*> output_names_;
  std::vector<OrtValue*> outputs_;
};

namespace cuda {

void LaunchGpt_InitAttentionMask(int64_t* mask_data, int64_t* position_data, int32_t* sequence_lengths, const int64_t* input_ids,
                                 int batch_size, int num_beams, int sequence_length, int pad_token_id, cudaStream_t stream);
void LaunchGpt_UpdatePositionIds(int64_t* positions, int batch_beam_size, int current_length, cudaStream_t stream);
void LaunchGpt_UpdateMask(int64_t* mask_data, const int64_t* old_mask_data, int batch_beam_size, int current_length, cudaStream_t stream);

void LaunchFp16ToFp32(const uint16_t *fp16, float* fp32, int count, cudaStream_t stream);

}
}