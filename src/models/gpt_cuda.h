#include "onnxruntime_cxx_api_2.h"

namespace Generators {

struct Gpt_Cuda {

  struct ModelParams {
    int vocab_size{};
    int head_count{};
    int hidden_size{};
    int layer_count{};
    bool logits_uses_seq_len{};  // Logits shape is [... seq_len, vocab_size ] vs [... 1, vocab_size ]
  };

  Gpt_Cuda(OrtEnv& ort_env, const ORTCHAR_T* decode_path, cudaStream_t cuda_stream);

  void CreateInputs(std::span<int32_t> sequence_lengths, const SearchParams& params);
  std::span<const ScoreType> GetLogits();
  int GetVocabSize() { return model_params_.vocab_size; }
  void Run(std::span<const int32_t> next_tokens, std::span<const int32_t> next_indices, int current_length);

 private:
  void UpdateInputs(std::span<const int32_t> next_tokens, std::span<const int32_t> beam_indices, int current_length);
  void PickPastState(size_t index, std::span<const int32_t> beam_indices);

  ModelParams model_params_;
  SearchParams search_params_;
  bool first_run_{true};

  Ort::Allocator& allocator_cpu_;
  std::unique_ptr<OrtMemoryInfo> memory_info_cuda_;
  std::unique_ptr<Ort::Allocator> allocator_cuda_;

  cudaStream_t cuda_stream_;

  bool past_present_share_buffer_{};  // NYI

  std::span<int32_t> next_positions_;  // shape (batch_size, num_beams). Next position value for position_ids.
  Ort::IAllocatorUniquePtr<int32_t> next_positions_buffer_;
  std::unique_ptr<OrtValue> next_positions_tensor_;  // Tensor of the 'next_position_' buffer

  // Sessions
  std::unique_ptr<OrtSession> session_decode_;

  // Inputs
  std::unique_ptr<OrtValue> input_ids_, expanded_input_ids_;
  std::unique_ptr<OrtValue> position_ids_, expanded_position_ids_;
  std::unique_ptr<OrtValue> attention_mask_, expanded_attention_mask_;
  std::unique_ptr<OrtValue> empty_past_;
  std::vector<std::unique_ptr<OrtValue>> pasts_;
  std::unique_ptr<OrtIoBinding> io_binding_decode_;

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

namespace cuda {

void LaunchGpt_InitAttentionMask(int32_t* mask_data, int32_t* position_data, int32_t* sequence_lengths, const int32_t* input_ids,
                                 int batch_size, int num_beams, int sequence_length, int pad_token_id, cudaStream_t stream);
void LaunchGpt_UpdatePositionIds(int32_t* positions, int batch_beam_size, int current_length, cudaStream_t stream);
void LaunchGpt_UpdateMask(int32_t* mask_data, const int32_t* old_mask_data, int batch_beam_size, int current_length, cudaStream_t stream);
}
}
