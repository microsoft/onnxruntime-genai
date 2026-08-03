#pragma once
#include "model.h"
#include "input_ids.h"
#include "logits.h"
#include "kv_cache.h"
#include "position_inputs.h"
#include "extra_inputs.h"
#include "recurrent_state.h"

namespace Generators {

struct DecoderOnly_Model : Model {
  DecoderOnly_Model(std::unique_ptr<Config> config, OrtEnv& ort_env);

  std::unique_ptr<State> CreateState(DeviceSpan<int32_t> sequence_lengths_unk, const GeneratorParams& params) const override;

  std::unique_ptr<OrtSession> session_decoder_;
};

struct DecoderOnly_State : State {
  DecoderOnly_State(const DecoderOnly_Model& model, DeviceSpan<int32_t> sequence_lengths_unk, const GeneratorParams& params);

  void SetExtraInputs(const std::vector<ExtraInput>& extra_inputs) override;

  DeviceSpan<float> Run(int total_length, DeviceSpan<int32_t>& next_tokens, DeviceSpan<int32_t> next_indices) override;
  DeviceSpan<float> RunTree(int stable_length, DeviceSpan<int32_t>& tree_tokens,
                            std::span<const int64_t> position_ids,
                            std::span<const uint8_t> tree_mask);

  void RewindTo(size_t index) override;
  void CompactTreeCache(size_t stable_length, std::span<const size_t> tree_indices);

 private:
  DeviceSpan<float> RunWithChunking(int total_length, DeviceSpan<int32_t>& next_tokens,
                                    DeviceSpan<int32_t> next_indices, size_t chunk_size);

  void UpdateInputsOutputs(DeviceSpan<int32_t>& next_tokens, DeviceSpan<int32_t> beam_indices, int total_length);
  void AddAttentionBias();
  void UpdateAttentionBias(int total_length, int new_length);
  void SetTreeAttentionBias(size_t stable_length, std::span<const uint8_t> tree_mask);

  const DecoderOnly_Model& model_;

  DefaultInputIDs input_ids_{*this};
  Logits logits_{*this};
  std::unique_ptr<KeyValueCache> kv_cache_;
  std::unique_ptr<RecurrentState> recurrent_state_;
  std::unique_ptr<PositionInputs> position_inputs_;
  ExtraInputs extra_inputs_{*this};
  std::unique_ptr<Tensor> attention_bias_;
  ONNXTensorElementDataType attention_bias_type_{ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED};
  std::array<int64_t, 4> attention_bias_shape_{};
  size_t attention_bias_input_index_{~0U};
};

}  // namespace Generators
