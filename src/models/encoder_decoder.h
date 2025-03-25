// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.


#pragma once
#include "model.h"
#include "input_ids.h"
#include "logits.h"
#include "kv_cache.h"
#include "position_inputs.h"
#include "extra_inputs.h"

namespace Generators {

struct EncoderDecoderModel : Model {
  EncoderDecoderModel(std::unique_ptr<Config> config, OrtEnv& ort_env);
  EncoderDecoderModel(const EncoderDecoderModel&) = delete;
  EncoderDecoderModel& operator=(const EncoderDecoderModel&) = delete;

  std::unique_ptr<State> CreateState(RoamingArray<int32_t> sequence_lengths, const GeneratorParams& params) const override;

  std::unique_ptr<OrtSession> session_decoder_; // input_ids, attention_mask, self_kv_cache, cross_kv_cache -> logits
  std::unique_ptr<OrtSession> session_encoder_; // input_ids, attention_mask -> cross_kv_cache
};

struct EncoderState : State {
    EncoderState(const EncoderDecoderModel& model, RoamingArray<int32_t> sequence_lengths, const GeneratorParams& params);
    EncoderState(const EncoderState&) = delete;
    EncoderState& operator=(const EncoderState&) = delete;
    
    void AddCrossCache(std::unique_ptr<Cross_Cache>& cross_cache) { cross_cache->AddOutputs(*this); }
    RoamingArray<float> Run(int current_length, RoamingArray<int32_t> next_tokens, RoamingArray<int32_t> next_indices) override;
  
   private:
    friend struct EncoderDecoderState;
  
    const EncoderDecoderModel& model_;
    InputIDs input_ids_{*this};
    PositionInputs position_inputs_;
  };

struct DecoderState : State {
  DecoderState(const EncoderDecoderModel& model, RoamingArray<int32_t> sequence_lengths, const GeneratorParams& params);
  DecoderState(const DecoderState&) = delete;
  DecoderState& operator=(const DecoderState&) = delete;
  
  void AddCrossCache(std::unique_ptr<Cross_Cache>& cross_cache) { cross_cache->AddInputs(*this); }
  RoamingArray<float> Run(int current_length, RoamingArray<int32_t> next_tokens, RoamingArray<int32_t> next_indices) override;

 private:
  friend struct EncoderDecoderState;

  // Update inputs and outputs for decoder
  void UpdateInputsOutputs(const RoamingArray<int32_t>& next_tokens, RoamingArray<int32_t> next_indices, int current_length, bool first_update);

  const EncoderDecoderModel& model_;

  InputIDs input_ids_{*this};
  KV_Cache kv_cache_{*this};
  PositionInputs position_inputs_;   

  std::unique_ptr<OrtValue> past_sequence_length_;          // Model input

  Logits logits_{*this};
  std::vector<std::unique_ptr<OrtValue>> output_cross_qk_;

  // Properties about cross attention's QK outputs
  std::vector<std::string> output_cross_qk_names_;          // Formatted names to check if cross attention's QK outputs exist in model
  std::array<int64_t, 4> output_cross_qk_shape_;            // Shape of cross attention's QK outputs
  ONNXTensorElementDataType output_cross_qk_type_;          // Type of cross attention's QK outputs

  size_t output_cross_qk_index_{~0U};
};

struct EncoderDecoderState : State {
    EncoderDecoderState(const EncoderDecoderModel& model, const GeneratorParams& params, RoamingArray<int32_t> sequence_lengths);
    EncoderDecoderState(const EncoderDecoderState&) = delete;
    EncoderDecoderState& operator=(const EncoderDecoderState&) = delete;
  
    RoamingArray<float> Run(int current_length, RoamingArray<int32_t> next_tokens, RoamingArray<int32_t> next_indices) override;
    OrtValue* GetInput(const char* name) override;
    OrtValue* GetOutput(const char* name) override;
  
  private:
    void TransposeKCaches(std::vector<std::unique_ptr<OrtValue>>& kv_caches);
    void UpdateInputsOutputs(const RoamingArray<int32_t>& next_tokens, RoamingArray<int32_t> next_indices, int current_length);
  
    const EncoderDecoderModel& model_;
  
    std::unique_ptr<EncoderState> encoder_state_;
    std::unique_ptr<Cross_Cache> cross_cache_;            // Model output for encoder, constant input for decoder
    std::unique_ptr<DecoderState> decoder_state_;

    std::unique_ptr<OrtValue> transpose_k_cache_buffer_;
  };

}  // namespace Generators