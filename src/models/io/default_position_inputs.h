#pragma once

#include "position_inputs.h"

namespace Generators {

struct DefaultPositionInputs : PositionInputs {
  DefaultPositionInputs(const Model& model, State& state, DeviceSpan<int32_t> sequence_lengths_unk, const std::string& attention_mask_name);

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
  bool ShouldUseStaticMaskHandling() const;

  const Model& model_;
  State& state_;
  std::string attention_mask_name_;
  size_t mask_input_index_{~0U};
  size_t posid_input_index_{~0U};
  ONNXTensorElementDataType type_;
  bool has_mask_input_{};
  bool has_posid_input_{};
  std::array<int64_t, 2> position_ids_shape_{};
  std::unique_ptr<Tensor> position_ids_;
  std::unique_ptr<Tensor> position_ids_next_;
  std::array<int64_t, 2> attention_mask_shape_{};
  std::unique_ptr<Tensor> attention_mask_;
  std::unique_ptr<Tensor> attention_mask_next_;
  bool is_first_update_{true};
};

}  // namespace Generators
