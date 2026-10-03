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

struct DecoderOnlyModel : Model {
  DecoderOnlyModel(std::unique_ptr<Config> config, OrtEnv& ort_env);

  std::unique_ptr<State> CreateState(DeviceSpan<int32_t> sequence_lengths_unk, const GeneratorParams& params) const override;

  std::unique_ptr<OrtSession> session_decoder_;
  std::shared_ptr<CpuEmbedding> cpu_embedding_;
};

struct DecoderState : State {
  DecoderState(const DecoderOnlyModel& model, DeviceSpan<int32_t> sequence_lengths,
               const GeneratorParams& params);
  DecoderState(const MultiModalLanguageModel& model, DeviceSpan<int32_t> sequence_lengths,
               const GeneratorParams& params);
  DecoderState(const DecoderState&) = delete;
  DecoderState& operator=(const DecoderState&) = delete;

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

  bool SupportsPrefillChunking(bool has_multimodal_content) const;
  void PrepareEmbeddingsForPrefill(size_t new_length);
  DeviceSpan<float> RunPrefillWithChunking(int current_length, DeviceSpan<int32_t>& next_tokens,
                                           DeviceSpan<int32_t> next_indices, size_t chunk_size);

  void UpdateInputsOutputs(DeviceSpan<int32_t>& next_tokens, int current_length,
                           DeviceSpan<int32_t> beam_indices);

  Embeddings& GetInputsEmbeds() { return *inputs_embeds_; }
  PositionInputs& GetPositionInputs() { return *position_inputs_; }
  Embeddings* GetPerLayerInputs() { return per_layer_inputs_.get(); }

 private:
  DecoderState(const GeneratorParams& params, const Model& model, OrtSession& session,
               DeviceSpan<int32_t> sequence_lengths);

  DeviceSpan<float> RunWithChunking(int current_length, DeviceSpan<int32_t>& next_tokens,
                                    DeviceSpan<int32_t> next_indices, size_t chunk_size);
  DeviceSpan<float> RunDecoder(int sequence_length, bool graph_capture_this_run);
  void ApplyRunOptions();
  void Initialize();
  bool UsesEmbeddings() const { return inputs_embeds_ != nullptr; }

  OrtSession& decoder_session_;
  std::unique_ptr<DefaultInputIDs> input_ids_;
  std::unique_ptr<Embeddings> inputs_embeds_;
  std::unique_ptr<Embeddings> per_layer_inputs_;
  std::unique_ptr<PositionInputs> position_inputs_;
  std::unique_ptr<KeyValueCache> kv_cache_;
  std::unique_ptr<RecurrentState> recurrent_state_;
  Logits logits_{*this};
  std::unique_ptr<HiddenStatesInputs> hidden_states_;
  std::unique_ptr<HiddenStatesOutputs> hidden_states_output_;
  ExtraInputs extra_inputs_{*this};
};

}  // namespace Generators
