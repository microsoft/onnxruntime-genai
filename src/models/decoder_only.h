#pragma once
#include <mutex>
#include "model.h"
#include "cpu_embedding.h"
#include "models/io/input_ids.h"
#include "models/io/embeddings.h"
#include "models/io/logits.h"
#include "io/kv_cache.h"
#include "models/io/position_inputs.h"
#include "models/io/extra_inputs.h"
#include "models/io/hidden_states.h"
#include "models/io/recurrent_state.h"
#include "models/io/ple_state.h"
#include "models/io/indexer_cache.h"

namespace Generators {
struct EngramState;
struct CpuEngram;

struct DecoderOnly_Model : Model {
  DecoderOnly_Model(std::unique_ptr<Config> config, OrtEnv& ort_env);

  std::unique_ptr<State> CreateState(DeviceSpan<int32_t> sequence_lengths_unk, const GeneratorParams& params) const override;

  void InitializeIndexShare(const Config::Model::Mtp::IndexShare& config, OrtEnv& ort_env);

  std::unique_ptr<OrtSession> session_decoder_;
  Config::Model::Mtp::IndexShare index_share_config_;
  std::unique_ptr<OrtSession> session_engram_;
  std::unique_ptr<OrtSessionOptions> engram_session_options_;
  std::shared_ptr<CpuEmbedding> cpu_embedding_;
  std::shared_ptr<const CpuEngram> cpu_engram_;
  std::once_flag cpu_engram_init_;
};

struct DecoderOnly_State : State {
  DecoderOnly_State(const DecoderOnly_Model& model, DeviceSpan<int32_t> sequence_lengths_unk, const GeneratorParams& params);
  ~DecoderOnly_State() override;

  void SetExtraInputs(const std::vector<ExtraInput>& extra_inputs) override;

  DeviceSpan<float> Run(int total_length, DeviceSpan<int32_t>& next_tokens, DeviceSpan<int32_t> next_indices) override;

  void RewindTo(size_t index) override;
  void ValidateRewindTo(size_t index) const override;

  void SnapshotState(size_t position) override;

  bool HasCroppableRecurrentState() const override;
  bool CropRequiresSnapshot() const override {
    return (recurrent_state_ && recurrent_state_->CropRequiresSnapshot()) ||
           (indexer_cache_ && indexer_cache_->HasStateUpdates());
  }
  int64_t RecurrentStateWindow() const override;
  void CropToAccepted(size_t new_length, size_t recurrent_position) override;
  void ValidateCropToAccepted(size_t new_length, size_t recurrent_position) const override;

  // Stage the hidden_states values for the next Run (for models with a hidden_states input,
  // e.g. the MTP self-speculative head). No-op if the model has no hidden_states input.
  void SetHiddenStates(OrtValue* hidden_states) override;
  OrtValue* GetRotaryPositionIds() override { return position_inputs_->GetRotaryPositionIds(); }
  void SetRotaryPositionIds(OrtValue* positions) override { position_inputs_->SetRotaryPositionIds(positions); }

 private:
  DeviceSpan<float> RunWithChunking(int total_length, DeviceSpan<int32_t>& next_tokens,
                                    DeviceSpan<int32_t> next_indices, size_t chunk_size);

  void UpdateInputsOutputs(DeviceSpan<int32_t>& next_tokens, DeviceSpan<int32_t> beam_indices, int total_length);

  const DecoderOnly_Model& model_;

  DefaultInputIDs input_ids_{*this};
  Logits logits_{*this};
  std::unique_ptr<KeyValueCache> kv_cache_;
  std::unique_ptr<RecurrentState> recurrent_state_;
  std::unique_ptr<PleState> ple_state_;
  std::unique_ptr<IndexerCache> indexer_cache_;
  std::unique_ptr<PositionInputs> position_inputs_;
  std::unique_ptr<HiddenStatesInputs> hidden_states_;          // Only for models with a hidden_states input (MTP head).
  std::unique_ptr<HiddenStatesOutputs> hidden_states_output_;  // Only for models that emit a hidden_states output (CUDA-graph-safe).
  std::unique_ptr<EngramState> engram_state_;
  std::unique_ptr<Embeddings> engram_embeddings_;
  ExtraInputs extra_inputs_{*this};
};

}  // namespace Generators
