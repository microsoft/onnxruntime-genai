#pragma once

#include <cstddef>
#include <memory>

#include "models/model.h"
#include "models/embedding/cpu_embedding.h"
#include "models/embedding/embeddings.h"
#include "models/io/input_ids.h"
#include "models/io/logits.h"
#include "models/io/kv_cache.h"
#include "models/io/position_inputs.h"
#include "models/io/extra_inputs.h"
#include "models/io/hidden_states.h"
#include "models/io/recurrent_state.h"

namespace Generators {

struct MultiModalLanguageModel;
struct MultiModalPipelineState;

struct DecoderState : State {
 protected:
  DecoderState(const GeneratorParams& params, const Model& model, OrtSession& session)
      : State{params, model}, decoder_session_{session} {}

  DeviceSpan<float> RunDecoder(Logits& logits, RecurrentState* recurrent_state,
                               int sequence_length, bool graph_capture_this_run);
  void ApplyRunOptions();

  OrtSession& decoder_session_;
};

struct DecoderOnlyModel : Model {
  DecoderOnlyModel(std::unique_ptr<Config> config, OrtEnv& ort_env);

  std::unique_ptr<State> CreateState(DeviceSpan<int32_t> sequence_lengths_unk, const GeneratorParams& params) const override;

  std::unique_ptr<OrtSession> session_decoder_;
  std::shared_ptr<CpuEmbedding> cpu_embedding_;
};

struct DecoderOnlyState : DecoderState {
  DecoderOnlyState(const DecoderOnlyModel& model, DeviceSpan<int32_t> sequence_lengths_unk, const GeneratorParams& params);

  void SetExtraInputs(const std::vector<ExtraInput>& extra_inputs) override;

  DeviceSpan<float> Run(int total_length, DeviceSpan<int32_t>& next_tokens, DeviceSpan<int32_t> next_indices) override;

  void RewindTo(size_t index) override;

  void SnapshotState(size_t position) override;

  bool HasCroppableRecurrentState() const override;
  int64_t RecurrentStateWindow() const override;
  void CropToAccepted(size_t new_length, size_t recurrent_position) override;

  // Stage the hidden_states values for the next Run (for models with a hidden_states input,
  // e.g. the MTP self-speculative head). No-op if the model has no hidden_states input.
  void SetHiddenStates(OrtValue* hidden_states) override;

 private:
  DeviceSpan<float> RunWithChunking(int total_length, DeviceSpan<int32_t>& next_tokens,
                                    DeviceSpan<int32_t> next_indices, size_t chunk_size);

  void UpdateInputsOutputs(DeviceSpan<int32_t>& next_tokens, DeviceSpan<int32_t> beam_indices, int total_length);

  const DecoderOnlyModel& model_;

  DefaultInputIDs input_ids_{*this};
  Logits logits_{*this};
  std::unique_ptr<KeyValueCache> kv_cache_;
  std::unique_ptr<RecurrentState> recurrent_state_;
  std::unique_ptr<PositionInputs> position_inputs_;
  std::unique_ptr<HiddenStatesInputs> hidden_states_;          // Only for models with a hidden_states input (MTP head).
  std::unique_ptr<HiddenStatesOutputs> hidden_states_output_;  // Only for models that emit a hidden_states output (CUDA-graph-safe).
  ExtraInputs extra_inputs_{*this};
};

struct MultiModalDecoderState : DecoderState {
  MultiModalDecoderState(const MultiModalLanguageModel& model, DeviceSpan<int32_t> sequence_lengths,
                         const GeneratorParams& params);
  MultiModalDecoderState(const MultiModalDecoderState&) = delete;
  MultiModalDecoderState& operator=(const MultiModalDecoderState&) = delete;

  DeviceSpan<float> Run(int current_length, DeviceSpan<int32_t>& next_tokens, DeviceSpan<int32_t> next_indices) override;
  void UpdateInputsOutputs(DeviceSpan<int32_t>& next_tokens, int current_length, DeviceSpan<int32_t> beam_indices);

  bool SupportsPrefillChunking(bool has_multimodal_content) const;
  void PrepareEmbeddingsForPrefill(size_t new_length);
  DeviceSpan<float> RunPrefillWithChunking(int current_length, DeviceSpan<int32_t>& next_tokens,
                                           DeviceSpan<int32_t> next_indices, size_t chunk_size);

  Embeddings& GetInputsEmbeds() { return inputs_embeds_; }
  PositionInputs& GetPositionInputs() { return *position_inputs_; }
  virtual Embeddings* GetPerLayerInputs() { return nullptr; }

 protected:
  friend struct MultiModalPipelineState;

  void UpdateInputsOutputs(DeviceSpan<int32_t>& next_tokens, int current_length,
                           DeviceSpan<int32_t> beam_indices, size_t new_length);
  virtual void UpdateExtraSequenceLength(size_t new_length) {}
  virtual void UseExtraChunkView(size_t offset, size_t count) {}
  virtual void RestoreExtraFullView() {}

  const MultiModalLanguageModel& model_;
  Embeddings inputs_embeds_;
  std::unique_ptr<DefaultInputIDs> decoder_input_ids_;
  std::unique_ptr<PositionInputs> position_inputs_;
  std::unique_ptr<KeyValueCache> kv_cache_;
  std::unique_ptr<RecurrentState> recurrent_state_;
  Logits logits_{*this};
};

std::unique_ptr<MultiModalDecoderState> CreateDecoderState(
    const MultiModalLanguageModel& model, DeviceSpan<int32_t> sequence_lengths,
    const GeneratorParams& params);

}  // namespace Generators
