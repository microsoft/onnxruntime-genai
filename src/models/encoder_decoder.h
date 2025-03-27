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

  std::unique_ptr<State> CreateState(DeviceSpan<int32_t> sequence_lengths_unk, const GeneratorParams& params) const override;

  std::unique_ptr<OrtSession> session_decoder_; // input_ids, attention_mask, self_kv_cache, cross_kv_cache -> logits
  std::unique_ptr<OrtSession> session_encoder_; // input_ids, attention_mask -> cross_kv_cache
};

struct EncoderState : State {
    EncoderState(const EncoderDecoderModel& model, DeviceSpan<int32_t> sequence_lengths, const GeneratorParams& params);
    
    void AddCrossCache(std::unique_ptr<CrossCache>& cross_cache) { cross_cache->AddOutputs(); }
    DeviceSpan<float> Run(int current_length, DeviceSpan<int32_t>& next_tokens, DeviceSpan<int32_t> next_indices) override;
  
   private:
    friend struct EncoderDecoderState;
  
    const EncoderDecoderModel& model_;
    EncoderInputIDs encoder_input_ids_{*this};
    // DefaultPositionInputs position_inputs_;
    // std::unique_ptr<OrtValue> encoder_input_ids_;
    std::unique_ptr<OrtValue> encoder_attention_mask_;
  };

struct DecoderState : State {
  DecoderState(const EncoderDecoderModel& model, DeviceSpan<int32_t> sequence_lengths, const GeneratorParams& params);
  
  void AddCrossCache(std::unique_ptr<CrossCache>& cross_cache) { cross_cache->AddInputs(); }
  DeviceSpan<float> Run(int current_length, DeviceSpan<int32_t>& next_tokens, DeviceSpan<int32_t> next_indices) override;

 private:
  friend struct EncoderDecoderState;

  // Update inputs and outputs for decoder
  void UpdateInputsOutputs(DeviceSpan<int32_t>& next_tokens, DeviceSpan<int32_t> next_indices, int current_length, bool first_update);

  const EncoderDecoderModel& model_;

  DefaultInputIDs input_ids_{*this};
  DefaultKeyValueCache kv_cache_{*this};
  DefaultPositionInputs position_inputs_;   

  std::unique_ptr<OrtValue> past_sequence_length_;          // Model input

  Logits logits_{*this};
};

struct EncoderDecoderState : State {
    EncoderDecoderState(const EncoderDecoderModel& model, DeviceSpan<int32_t> sequence_lengths, const GeneratorParams& params);

    DeviceSpan<float> Run(int current_length, DeviceSpan<int32_t>& next_tokens, DeviceSpan<int32_t> next_indices) override;
    OrtValue* GetInput(const char* name) override;
    OrtValue* GetOutput(const char* name) override;
  
  private:
    void UpdateInputsOutputs(DeviceSpan<int32_t>& next_tokens, DeviceSpan<int32_t> next_indices, int current_length, bool first_update);
    void Initialize(DeviceSpan<int32_t>& next_tokens, int total_length, DeviceSpan<int32_t> beam_indices);
    enum struct RunState {
      Encoder_Decoder_Init,
      Decoder_First,
      Decoder,
    } run_state_{RunState::Encoder_Decoder_Init};
  
    const EncoderDecoderModel& model_;

    DefaultInputIDs decoder_input_ids_{*this};
    Logits logits_{*this};
    DefaultKeyValueCache kv_cache_{*this};
    std::unique_ptr<InputIDs> input_ids_;
    std::unique_ptr<OrtValue> encoder_attention_mask;
    std::unique_ptr<PositionInputs> position_inputs_;
    ExtraInputs extra_inputs_{*this};
  
    // Temporary hack to have different sized outputs from the encoder that we then expand into the decoder buffers
    std::vector<std::unique_ptr<OrtValue>> init_presents_;  // Hacked sized encoder_decoder_init presents
    std::vector<OrtValue*> presents_;                       // The original present buffers we must resize init_presents_ into after the first run
  
    std::unique_ptr<EncoderState> encoder_state_;
    CrossCache cross_cache_{*this};          // Model output for encoder, constant input for decoder
    std::unique_ptr<DecoderState> decoder_state_;
  };

}  // namespace Generators