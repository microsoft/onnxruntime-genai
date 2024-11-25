#pragma once

#include "static_buffer.h"

#if USE_DML
#include "../dml/dml_update_mask_kernel.h"
#include "../dml/dml_increment_values_kernel.h"
#endif

namespace Generators {

struct PositionInputs {
  PositionInputs(const Model& model, State& state, DeviceSpan<int32_t> sequence_lengths_unk);

  void Add();
  void Update(const DeviceSpan<int32_t>& next_tokens, int total_length, int new_length);

  void RewindTo(size_t index);

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
  void CreateAndInitializePositionIDs(const DeviceSpan<int32_t>& next_tokens, std::array<int64_t, 2> shape);
  template <typename T>
  void CreateAndInitializeAttentionMask(const DeviceSpan<int32_t>& next_tokens, std::array<int64_t, 2> shape);

  template <typename T>
  void UpdatePositionIDsImpl(int total_length, int new_kv_length);
  template <typename T>
  void UpdateAttentionMaskImpl(int total_length);

#if USE_CUDA || USE_DML
  void CopyNextPositionIDsToCurrent();
#endif

#if USE_DML
  void UpdatePositionIDsImplDML();
  void UpdateAttentionMaskImplDML(int total_length);
#endif

#if USE_CUDA
  void RewindMask(size_t index);
#endif

  const Model& model_;
  State& state_;

  size_t mask_input_index_{~0U};
  size_t posid_input_index_{~0U};

  ONNXTensorElementDataType type_;  // Common type for position_ids and attention_mask

  bool has_mask_input_{};
  bool has_posid_input_{};

  std::array<int64_t, 2> position_ids_shape_{};  // {params.batch_size*params.beam_size, params.sequence_length}
  std::unique_ptr<OrtValue> position_ids_;
  std::array<int64_t, 2> attention_mask_shape_{};  // {params.batch_size*params.beam_size, params.sequence_length}
  std::unique_ptr<OrtValue> attention_mask_;

  std::unique_ptr<OrtValue> position_ids_next_;    // Replaces position_ids_ after the first Run() call
  std::unique_ptr<OrtValue> attention_mask_next_;  // Replaces attention_mask_ after the first Run() call

  // Used for decoding runs with cuda graphs.
  StaticBuffer* sb_position_ids_{};
  StaticBuffer* sb_attention_mask_{};

  bool is_first_mask_update_{true};
  bool is_first_update_{true};

#if USE_DML
  std::optional<DmlUpdateMaskKernel> dml_update_mask_kernel_;
  StaticBuffer* sb_attention_mask_next_{};
  std::optional<DmlIncrementValuesKernel> dml_update_position_ids_kernel_;
  bool is_second_mask_update_{};
#endif
};

}  // namespace Generators
