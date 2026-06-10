// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <future>
#include <optional>
#include <string>
#include <string_view>
#include <unordered_map>
#include <utility>
#include <vector>

#include "../worker_thread.h"
#include "model.h"
#include "input_ids.h"
#include "logits.h"
#include "kv_cache.h"
#include "windowed_kv_cache.h"
#include "position_inputs.h"
#include "extra_inputs.h"
#include "recurrent_state.h"

namespace Generators {

// PipelineFlow (issue #2114 PR3): resolves `config.pipeline.flow`/`dataflow` into the single source
// of truth for stage lifecycle gating and cross-stage wiring inside DecoderOnlyPipelineState.
//
// Each decoder.pipeline[] stage is classified into a lifecycle phase derived from the matching
// flow step's `when`:
//   * "init"  -> Phase::Init  (prompt-processing only; preserved via run_on_prompt/run_on_token_gen)
//   * "step"  -> Phase::Step  (every token; the default)
//   * "final" -> Phase::Final (NEW: executed once after the generation loop, via State::Finalize)
//
// Backward compatibility: for v1-derived configs the existing run_on_prompt/run_on_token_gen fields
// (populated by PR1 lowering) still drive init-vs-step gating, so behavior is byte-for-byte preserved.
// The flow only additionally identifies "final" stages (no in-tree model uses them yet) and supplies
// explicit `dataflow[]` wires that override the name-based auto-match.
struct PipelineFlow {
  enum class Phase { Init, Step, Final };

  PipelineFlow() = default;

  // Resolves the flow against config.model.decoder.pipeline[] order. Applies the issue §3.2
  // load-time guardrails: throws on a dataflow cycle or when the stage count exceeds 10.
  explicit PipelineFlow(const Config& config);

  Phase PhaseForStage(size_t stage_id) const;
  bool IsFinal(size_t stage_id) const;
  bool HasFinalStages() const { return has_final_; }

  // Explicit dataflow override (issue §3.3): returns the producing tensor (ortvalue_store_ key) that
  // should feed input `input` of session `session`, or nullptr when no explicit wire exists.
  const std::string* ExplicitSource(const std::string& session, const std::string& input) const;

 private:
  static std::string WireKey(std::string_view session, std::string_view input);
  static std::pair<std::string, std::string> SplitWire(const std::string& endpoint);

  std::vector<Phase> stage_phase_;  // indexed by decoder.pipeline[] stage id
  bool has_final_{false};
  std::unordered_map<std::string, std::string> explicit_wires_;  // WireKey(to_session,to_input) -> from_tensor
};

struct DecoderOnlyPipelineModel : Model {
  DecoderOnlyPipelineModel(std::unique_ptr<Config> config, OrtEnv& ort_env);

  DecoderOnlyPipelineModel(const DecoderOnlyPipelineModel&) = delete;
  DecoderOnlyPipelineModel& operator=(const DecoderOnlyPipelineModel&) = delete;

  std::unique_ptr<State> CreateState(DeviceSpan<int32_t> sequence_lengths,
                                     const GeneratorParams& params) const override;

  std::vector<std::unique_ptr<OrtSession>> sessions_;
  OrtEnv& ort_env_;
};

struct IntermediatePipelineState : State {
  IntermediatePipelineState(const DecoderOnlyPipelineModel& model, const GeneratorParams& params,
                            size_t pipeline_state_index);

  IntermediatePipelineState(const IntermediatePipelineState&) = delete;
  IntermediatePipelineState& operator=(const IntermediatePipelineState&) = delete;

  DeviceSpan<float> Run(int current_length, DeviceSpan<int32_t>& next_tokens,
                        DeviceSpan<int32_t> next_indices) override;

  bool HasInput(std::string_view name) const;

  bool HasOutput(std::string_view name) const;

  bool SupportsPrimaryDevice() const;

  size_t id_;

 private:
  const DecoderOnlyPipelineModel& model_;
};

struct DecoderOnlyPipelineState : State {
  DecoderOnlyPipelineState(const DecoderOnlyPipelineModel& model, DeviceSpan<int32_t> sequence_lengths,
                           const GeneratorParams& params);

  DecoderOnlyPipelineState(const DecoderOnlyPipelineState&) = delete;
  DecoderOnlyPipelineState& operator=(const DecoderOnlyPipelineState&) = delete;

  void SetExtraInputs(const std::vector<ExtraInput>& extra_inputs) override;

  DeviceSpan<float> Run(int total_length, DeviceSpan<int32_t>& next_tokens,
                        DeviceSpan<int32_t> next_indices) override;

  // Post-loop hook (issue #2114 PR3): executes flow stages whose `when=="final"` once generation
  // has completed. No-op when the resolved flow has no final stages (all in-tree models today).
  void Finalize(int current_length) override;

  OrtValue* GetOutput(const char* name) override;

  void RunPipeline(int total_length, DeviceSpan<int32_t>& next_tokens,
                   DeviceSpan<int32_t> next_indices, bool is_last_chunk);

  // Rewinds all per-step state owned by the pipeline (KV cache(s), position inputs, and
  // recurrent state) back to `index` tokens, so the next Run() produces correct results
  // starting from that length. Mirrors DecoderOnly_State::RewindTo. Throws (via the owned
  // cache) if a sub-cache cannot be rewound rather than silently corrupting state.
  void RewindTo(size_t index) override;

 protected:
  // Virtual hook called after each pipeline stage completes, before next stage starts.
  // Allows derived classes to modify stage outputs (e.g., inject vision embeddings).
  // stage_id: ID of the stage that just completed
  virtual void OnStageComplete(size_t stage_id) {}

  // Stores all the outputs from the previous pipeline state(s)
  std::unordered_map<std::string, std::unique_ptr<OrtValue>> ortvalue_store_;
  std::unique_ptr<InputIDs> input_ids_;  // Made protected for derived class access

 private:
  void UpdateKeyValueCache(DeviceSpan<int32_t> beam_indices, int total_length);

  // Binds managed/explicit/auto-matched IO for a single pipeline stage and runs it.
  // Shared by the per-token RunPipeline loop and the post-loop final-stage path.
  void RunStage(IntermediatePipelineState& pipeline_state, int total_length,
                DeviceSpan<int32_t>& next_tokens, DeviceSpan<int32_t> next_indices);

  void UpdateInputsOutputs(DeviceSpan<int32_t>& next_tokens, DeviceSpan<int32_t> next_indices,
                           int total_length);

  const DecoderOnlyPipelineModel& model_;
  std::vector<std::unique_ptr<IntermediatePipelineState>> pipeline_states_;

  PipelineFlow flow_;

  // Saved from the last Run() so Finalize() can replay managed indices into the final stages.
  DeviceSpan<int32_t> last_next_indices_{};
  int last_total_length_{};

  struct PartialKeyValueCacheUpdateRecord {
    std::vector<size_t> layer_indices{};     // indicates which layers of the KV cache are to be updated
    std::future<void> outstanding_update{};  // future for an outstanding update task
  };

  std::map<size_t, size_t> pipeline_state_id_to_partial_kv_cache_update_record_idx_;
  std::vector<PartialKeyValueCacheUpdateRecord> partial_kv_cache_update_records_;

  Logits logits_{*this};

  std::unique_ptr<KeyValueCache> key_value_cache_;
  const bool do_key_value_cache_partial_update_;
  std::optional<WorkerThread> key_value_cache_update_worker_thread_{};

  std::unique_ptr<RecurrentState> recurrent_state_;
  std::unique_ptr<PositionInputs> position_inputs_;
  ExtraInputs extra_inputs_{*this};
};

}  // namespace Generators
