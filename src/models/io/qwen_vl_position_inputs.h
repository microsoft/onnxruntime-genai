#pragma once

#include "position_inputs.h"

namespace Generators {

inline constexpr int64_t kMaxQwen2VLGridDim = 16384;

inline void ValidateQwen2VLGridTensorValues(const int64_t* grid_data, size_t elem_count, const char* tensor_name) {
  if (elem_count % 3 != 0) {
    throw std::runtime_error(std::string(tensor_name) + " element count must be divisible by 3");
  }

  for (size_t i = 0; i < elem_count; ++i) {
    if (grid_data[i] < 0) {
      throw std::runtime_error(std::string(tensor_name) + " values must be non-negative");
    }
    if (grid_data[i] > kMaxQwen2VLGridDim) {
      throw std::runtime_error(std::string(tensor_name) + " values must be <= " + std::to_string(kMaxQwen2VLGridDim));
    }
  }
}

inline void ValidateQwen2VLVisionLengthFitsSequence(int64_t llm_grid_t, int64_t llm_grid_h, int64_t llm_grid_w,
                                                    int64_t ed, int64_t seq_len) {
  const int64_t vision_len = llm_grid_t * llm_grid_h * llm_grid_w;
  if (ed + vision_len > seq_len) {
    throw std::runtime_error("Image/video grid dimensions (t=" + std::to_string(llm_grid_t) +
                             " h=" + std::to_string(llm_grid_h) +
                             " w=" + std::to_string(llm_grid_w) +
                             ") result in " + std::to_string(vision_len) +
                             " tokens but only " + std::to_string(seq_len - ed) +
                             " positions available in sequence");
  }
}

struct Qwen2VLPositionInputs;
struct InitPositionIdsFunctor;
struct InitAttentionMaskFunctor;

// Qwen2-VL uses 3D rotary position embeddings (mrope) for multimodal (vision + text) content.
struct Qwen2VLPositionInputs : PositionInputs {
  Qwen2VLPositionInputs(const Model& model, State& state, DeviceSpan<int32_t> sequence_lengths_unk);
  Qwen2VLPositionInputs(const Qwen2VLPositionInputs&) = delete;
  Qwen2VLPositionInputs& operator=(const Qwen2VLPositionInputs&) = delete;

  void Add() override;
  void Update(DeviceSpan<int32_t> next_tokens, int total_length, int new_length) override;
  void RewindTo(size_t index) override;

  void SetGridTensors(const std::shared_ptr<Tensor>& image_grid_thw,
                      const std::shared_ptr<Tensor>& video_grid_thw,
                      const std::shared_ptr<Tensor>& second_per_grid_ts);

  friend struct InitPositionIdsFunctor;
  friend struct InitAttentionMaskFunctor;

 private:
  void AddPositionIDs();
  void AddAttentionMask();
  template <typename T>
  void CreateAndInitialize3DPositionIDs(DeviceSpan<int32_t> next_tokens, std::array<int64_t, 3> shape);
  void Update3DPositionIDs(int base_pos);
  template <typename T>
  void CreateAndInitializeAttentionMask(DeviceSpan<int32_t> next_tokens, std::array<int64_t, 2> shape);
  void UpdateAttentionMask();

  const Model& model_;
  State& state_;
  size_t mask_input_index_{~0U};
  size_t posid_input_index_{~0U};
  ONNXTensorElementDataType type_;
  bool has_mask_input_{false};
  bool has_posid_input_{false};
  std::array<int64_t, 3> position_ids_shape_{};
  std::unique_ptr<Tensor> position_ids_;
  std::array<int64_t, 2> attention_mask_shape_{};
  std::unique_ptr<Tensor> attention_mask_;
  bool is_first_update_{true};
  std::shared_ptr<Tensor> image_grid_thw_;
  std::shared_ptr<Tensor> video_grid_thw_;
  std::shared_ptr<Tensor> second_per_grid_ts_;
  std::vector<int64_t> rope_deltas_;
  const int32_t image_token_id_;
  const int32_t video_token_id_;
  const int32_t vision_start_token_id_;
  const float tokens_per_second_;
  const int32_t spatial_merge_size_;
};

}  // namespace Generators
