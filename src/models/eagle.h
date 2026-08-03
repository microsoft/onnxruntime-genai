// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#pragma once

#include "decoder_only.h"
#include "speculative_decoding.h"

namespace Generators {

struct EagleModel;

struct EagleTree {
  std::vector<int32_t> tokens;
  std::vector<uint8_t> attention_mask;
  std::vector<int64_t> position_ids;
  std::vector<std::vector<size_t>> retrieve_indices;
  std::vector<size_t> selected_candidate_indices;
};

struct EagleTargetState : State {
  EagleTargetState(const EagleModel& model,
                   DeviceSpan<int32_t> sequence_lengths,
                   const GeneratorParams& params);

  DeviceSpan<float> Run(int total_length, DeviceSpan<int32_t>& next_tokens,
                        DeviceSpan<int32_t> next_indices) override;
  DeviceSpan<float> RunTree(DeviceSpan<int32_t>& tree_tokens,
                            std::span<const int64_t> position_ids,
                            std::span<const uint8_t> tree_mask);
  void CompactTree(std::span<const size_t> tree_indices);
  std::unique_ptr<OrtValue> CopyFeatures(size_t start, size_t count) const;
  void DiscardFeaturesBefore(size_t index);

  void RewindTo(size_t index) override;
  OrtValue* GetInput(const char* name) override;
  OrtValue* GetOutput(const char* name) override;
  void SetActiveAdapter(Adapters* adapters, const std::string& adapter_name) override;
  void SetRunOption(const char* key, const char* value) override;
  void SetExtraInputs(const std::vector<ExtraInput>& extra_inputs) override;

  size_t stable_length() const { return stable_length_; }

 private:
  struct FeatureSegment {
    size_t start{};
    size_t count{};
    size_t value_offset{};
    std::unique_ptr<OrtValue> value;
  };

  std::unique_ptr<OrtValue> CaptureTargetFeatures() const;
  void RecordFeatures(size_t start, std::unique_ptr<OrtValue> features);

  const EagleModel& eagle_model_;
  std::unique_ptr<DecoderOnly_State> inner_;
  size_t stable_length_{};
  std::vector<FeatureSegment> feature_segments_;
  std::unique_ptr<OrtValue> tree_features_;
};

struct EagleDraftState {
  explicit EagleDraftState(const EagleModel& model);

  void Prepare(std::unique_ptr<OrtValue> target_hidden_states,
               std::span<const int32_t> shifted_input_ids);
  EagleTree BuildTree() const;

  void RewindTo(size_t index);
  void Reset();
  void SetRunOption(const char* key, const char* value);

  bool initialized() const { return initialized_; }
  size_t cache_length() const { return cache_length_; }
  int32_t conditioning_token() const { return conditioning_token_; }

 private:
  struct RunResult {
    std::unique_ptr<OrtValue> hidden_states;
    std::unique_ptr<OrtValue> topk_log_scores;
    std::unique_ptr<OrtValue> mapped_topk_ids;
    std::unique_ptr<OrtValue> key;
    std::unique_ptr<OrtValue> value;
  };

  RunResult Run(std::span<const int32_t> input_ids,
                std::unique_ptr<OrtValue> target_hidden_states,
                std::unique_ptr<OrtValue> recurrent_hidden_states,
                bool use_target_hidden_states,
                const OrtValue& past_key,
                const OrtValue& past_value,
                std::span<const int64_t> position_ids,
                std::span<const uint8_t> tree_mask,
                size_t stable_length) const;

  const EagleModel& model_;
  ONNXTensorElementDataType data_type_{ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED};
  std::unique_ptr<OrtValue> key_;
  std::unique_ptr<OrtValue> value_;
  std::unique_ptr<OrtValue> last_hidden_state_;
  std::vector<float> initial_scores_;
  std::vector<int64_t> initial_mapped_ids_;
  size_t cache_length_{};
  int32_t conditioning_token_{};
  bool initialized_{};
  bool session_terminated_{};
  std::unique_ptr<OrtRunOptions> run_options_;
};

struct EagleModel : Model {
  EagleModel(std::unique_ptr<Config> config, OrtEnv& ort_env);

  std::unique_ptr<State> CreateState(DeviceSpan<int32_t> sequence_lengths,
                                     const GeneratorParams& params) const override;

  const DecoderOnly_Model& target_model() const { return *target_model_; }
  ONNXTensorElementDataType data_type() const { return data_type_; }

  std::shared_ptr<DecoderOnly_Model> target_model_;
  std::unique_ptr<OrtSessionOptions> eagle_session_options_;
  std::unique_ptr<OrtSession> eagle_session_;
  SessionInfo eagle_session_info_;

 private:
  ONNXTensorElementDataType data_type_{ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED};
};

struct EagleState : State {
  EagleState(const EagleModel& model,
             DeviceSpan<int32_t> sequence_lengths,
             const GeneratorParams& params);

  DeviceSpan<float> Run(int total_length, DeviceSpan<int32_t>& next_tokens,
                        DeviceSpan<int32_t> next_indices) override;
  void RewindTo(size_t index) override;
  OrtValue* GetInput(const char* name) override;
  OrtValue* GetOutput(const char* name) override;
  void SetActiveAdapter(Adapters* adapters, const std::string& adapter_name) override;
  void SetRunOption(const char* key, const char* value) override;
  void SetExtraInputs(const std::vector<ExtraInput>& extra_inputs) override;

  const EagleModel& eagle_model() const { return model_; }
  EagleTargetState& target_state() { return *target_state_; }
  EagleDraftState& draft_state() { return draft_state_; }

 private:
  const EagleModel& model_;
  std::unique_ptr<EagleTargetState> target_state_;
  EagleDraftState draft_state_;
};

}  // namespace Generators
