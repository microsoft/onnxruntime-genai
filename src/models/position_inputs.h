#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "../generators.h"
#include "model.h"

namespace Generators {

struct PositionInputs {
  virtual ~PositionInputs() = default;
  virtual void Add() = 0;
  virtual void Update(DeviceSpan<int32_t> next_tokens, int total_length, int new_length) = 0;
  virtual void RewindTo(size_t index) = 0;
};

enum class AttentionMaskMode {
  // Select static handling for graph capture or TRT-RTX shared KV buffers.
  Automatic,
  // Resize the mask to the active sequence length as generation advances.
  Dynamic,
  // Keep one fixed-shape mask and update its active prefix in place.
  Static,
};

struct AttentionMaskOptions {
  AttentionMaskMode mode{AttentionMaskMode::Automatic};
  // Overrides the static mask's second dimension; 0 uses generation max_length.
  int static_mask_length_override{};
};

struct DefaultPositionInputs : PositionInputs {
  DefaultPositionInputs(const Model& model, State& state, DeviceSpan<int32_t> sequence_lengths_unk,
                        const std::string& attention_mask_name,
                        AttentionMaskOptions attention_mask_options = {});

  void Add() override;
  void Update(DeviceSpan<int32_t> next_tokens, int total_length, int new_length) override;

  void RewindTo(size_t index) override;

 private:
  void AddAttentionMask();
  void AddPositionIDs();

  void CreateNextPositionIDsTensor();
  void CreateNextAttentionMaskTensor(int total_length);

  void UpdatePositionIDs(int total_length, int new_length);
  void UpdateAttentionMask(int total_length, int new_length);

  template <typename T>
  void InitializeSequenceLengths(std::array<int64_t, 2> shape, cpu_span<int32_t> sequence_lengths_unk);
  template <typename T>
  void CreateAndInitializePositionIDs(DeviceSpan<int32_t> next_tokens, std::array<int64_t, 2> shape);
  template <typename T>
  void CreateAndInitializeAttentionMask(DeviceSpan<int32_t> next_tokens, std::array<int64_t, 2> shape);
  template <typename T>
  void InitializeStaticMask(OrtValue& cpu_attention_mask);

  void RewindMask(size_t index);

  // This returns true when either:
  // 1. Graph capture is enabled, OR
  // 2. Past-present buffer sharing is enabled AND the device is NvTensorRtRtx
  // Both scenarios require static mask allocation and special shape handling for optimization
  bool ShouldUseStaticMaskHandling() const;
  int GetAttentionMaskCapacity() const;

  const Model& model_;
  State& state_;
  std::string attention_mask_name_;
  AttentionMaskOptions attention_mask_options_;

  size_t mask_input_index_{~0U};
  size_t posid_input_index_{~0U};

  ONNXTensorElementDataType type_;  // Common type for position_ids and attention_mask

  bool has_mask_input_{};
  bool has_posid_input_{};

  std::array<int64_t, 2> position_ids_shape_{};  // {params.batch_size*params.beam_size, params.sequence_length}
  std::unique_ptr<Tensor> position_ids_;
  std::unique_ptr<Tensor> position_ids_next_;      // Replaces position_ids_ after the first Run() call
  std::array<int64_t, 2> attention_mask_shape_{};  // {params.batch_size*params.beam_size, params.sequence_length}
  std::unique_ptr<Tensor> attention_mask_;
  std::unique_ptr<Tensor> attention_mask_next_;  // Replaces attention_mask_ after each run

  bool is_first_update_{true};
};

// Certain models can only process a fixed number of tokens at a time.
// For example, given a prompt with 120 tokens, and a model that can only process 20 tokens at a time,
// this class will split the position ids into 6 windows of 20 tokens each.
// At each update step, the next window of position ids is prepared.
// This is done until all windows have been processed before switching to the model-generation phase
// where position ids are prepared one id at a time.
// This class will also prepare the attention mask for each iteration. The attention mask buffer is allocated just
// once and reused for each iteration by setting the mask to 1 for current window tokens and previously active window tokens
// In contrast, DefaultPositionInputs processes all position ids at once.
struct WindowedPositionInputs : PositionInputs {
  WindowedPositionInputs(State& state);
  WindowedPositionInputs(const WindowedPositionInputs&) = delete;
  WindowedPositionInputs& operator=(const WindowedPositionInputs&) = delete;

  void Add() override;
  void Update(DeviceSpan<int32_t> next_tokens, int total_length, int new_length) override;
  void RewindTo(size_t index) override {
    throw std::runtime_error("WindowedPositionInputs does not support RewindTo.");
  };

 private:
  State& state_;
  const Model& model_{state_.model_};

  bool has_mask_input_{};
  bool has_posid_input_{};

  std::array<int64_t, 2> position_ids_shape_{};
  ONNXTensorElementDataType position_ids_type_{};
  std::unique_ptr<OrtValue> position_ids_;
  std::array<int64_t, 2> attention_mask_shape_{};
  ONNXTensorElementDataType attention_mask_type_{};
  std::unique_ptr<OrtValue> attention_mask_;
  size_t attention_mask_backward_offset_{~0U};

  size_t attention_mask_index_{~0U};
  size_t position_ids_index_{~0U};

  size_t window_size_{};
  size_t num_windows_{};
  size_t window_index_{};
};

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

// Qwen2-VL uses 3D rotary position embeddings (mrope) for multimodal (vision + text) content.
// Position IDs have shape [3, batch_size, seq_len] where the 3 dimensions represent:
//   - Dimensions 0: Temporal position
//   - Dimensions 1: Height position
//   - Dimensions 2: Width position
// For text, all 3 dimensions are identical. For vision, they are distinct.
// This class implements the logic from `get_rope_index` to build these 3D IDs.
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

  // Friend declarations for functors that need access to private methods
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

  std::array<int64_t, 3> position_ids_shape_{};  // {3, batch_size, sequence_length} for 3D positions
  std::unique_ptr<Tensor> position_ids_;

  std::array<int64_t, 2> attention_mask_shape_{};  // {batch_size, sequence_length}
  std::unique_ptr<Tensor> attention_mask_;

  bool is_first_update_{true};

  // Cached data from processor
  std::shared_ptr<Tensor> image_grid_thw_;
  std::shared_ptr<Tensor> video_grid_thw_;
  std::shared_ptr<Tensor> second_per_grid_ts_;
  std::vector<int64_t> rope_deltas_;

  // Config values initialized from model.config_ in constructor
  const int32_t image_token_id_;
  const int32_t video_token_id_;
  const int32_t vision_start_token_id_;
  const float tokens_per_second_;
  const int32_t spatial_merge_size_;
};

std::unique_ptr<PositionInputs> CreatePositionInputs(State& state, DeviceSpan<int32_t> sequence_lengths, const std::string& attention_mask_name);

}  // namespace Generators
