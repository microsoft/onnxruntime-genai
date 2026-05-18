#pragma once

#include "position_inputs.h"

namespace Generators {

// Qwen-VL/Qwen3.5 uses 3D rotary position embeddings (mRoPE) for multimodal
// and text-only decoder inputs. Position IDs have shape [3, batch_size, seq_len]
// where the 3 dimensions represent temporal, height, and width positions. For
// text, all 3 dimensions are identical. For vision, they are distinct.
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
  void Update3DPositionIDsInPlace(int base_pos);
  bool ShouldUseStaticPositionIDHandling() const;

  template <typename T>
  void CreateAndInitializeAttentionMask(DeviceSpan<int32_t> next_tokens, std::array<int64_t, 2> shape);
  template <typename T>
  void InitializeStaticMask(OrtValue& cpu_attention_mask);
  void UpdateAttentionMask(int total_length, int new_length);
  bool ShouldUseStaticMaskHandling() const;

  bool ShouldUseStaticInputsForGraphReplay() const;

  const Model& model_;
  State& state_;

  size_t mask_input_index_{~0U};
  size_t posid_input_index_{~0U};

  ONNXTensorElementDataType type_;

  bool has_mask_input_{false};
  bool has_posid_input_{false};

  std::array<int64_t, 3> position_ids_shape_{};  // {3, batch_size, sequence_length} for 3D positions
  std::unique_ptr<Tensor> position_ids_;

  std::array<int64_t, 2> attention_mask_shape_{};  // {batch_size, sequence_length}
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

std::unique_ptr<PositionInputs> TryCreateQwenPositionInputs(State& state, DeviceSpan<int32_t> sequence_lengths);

}  // namespace Generators
