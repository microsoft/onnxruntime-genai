// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <future>
#include <optional>
#include <limits>

#include "../worker_thread.h"
#include "model.h"
#include "input_ids.h"
#include "logits.h"
#include "kv_cache.h"
#include "windowed_kv_cache.h"
#include "position_inputs.h"
#include "extra_inputs.h"
#include "vision_pipeline.h"

namespace Generators {

struct VisionPipelineModel;
struct VisionPipelineState;

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

  OrtValue* GetOutput(const char* name) override;

  void RunPipeline(int total_length, DeviceSpan<int32_t>& next_tokens,
                   DeviceSpan<int32_t> next_indices, bool is_last_chunk);

 private:
  void UpdateKeyValueCache(DeviceSpan<int32_t> beam_indices, int total_length);

  void UpdateInputsOutputs(DeviceSpan<int32_t>& next_tokens, DeviceSpan<int32_t> next_indices,
                           int total_length);

  const DecoderOnlyPipelineModel& model_;
  std::vector<std::unique_ptr<IntermediatePipelineState>> pipeline_states_;

  struct PartialKeyValueCacheUpdateRecord {
    std::vector<size_t> layer_indices{};     // indicates which layers of the KV cache are to be updated
    std::future<void> outstanding_update{};  // future for an outstanding update task
  };

  std::map<size_t, size_t> pipeline_state_id_to_partial_kv_cache_update_record_idx_;
  std::vector<PartialKeyValueCacheUpdateRecord> partial_kv_cache_update_records_;

  // Stores all the outputs from the previous pipeline state(s)
  std::unordered_map<std::string, std::unique_ptr<OrtValue>> ortvalue_store_;

  std::unique_ptr<InputIDs> input_ids_;
  Logits logits_{*this};

  std::unique_ptr<KeyValueCache> key_value_cache_;
  const bool do_key_value_cache_partial_update_;
  std::optional<WorkerThread> key_value_cache_update_worker_thread_{};

  std::unique_ptr<PositionInputs> position_inputs_;
  ExtraInputs extra_inputs_{*this};
};

struct VisionDecoderPipelineModel : DecoderOnlyPipelineModel {
  VisionDecoderPipelineModel(std::unique_ptr<Config> config, OrtEnv& ort_env);

  VisionDecoderPipelineModel(const VisionDecoderPipelineModel&) = delete;
  VisionDecoderPipelineModel& operator=(const VisionDecoderPipelineModel&) = delete;

  std::unique_ptr<State> CreateState(DeviceSpan<int32_t> sequence_lengths,
                                     const GeneratorParams& params) const override;

  bool DecoderConsumesImageFeatures() const { return decoder_expects_image_features_; }
  const std::string& ImageFeaturesName() const { return image_features_name_; }

  std::unique_ptr<VisionPipelineModel> vision_pipeline_model_;
  bool decoder_expects_image_features_{false};
  std::string image_features_name_;
};

struct VisionDecoderPipelineState : DecoderOnlyPipelineState {
  VisionDecoderPipelineState(const VisionDecoderPipelineModel& model, DeviceSpan<int32_t> sequence_lengths,
                             const GeneratorParams& params);

  VisionDecoderPipelineState(const VisionDecoderPipelineState&) = delete;
  VisionDecoderPipelineState& operator=(const VisionDecoderPipelineState&) = delete;

  void SetExtraInputs(const std::vector<ExtraInput>& extra_inputs) override;

  DeviceSpan<float> Run(int total_length, DeviceSpan<int32_t>& next_tokens,
                        DeviceSpan<int32_t> next_indices) override;

 private:
  void AttachVisionFeatures();

  const VisionDecoderPipelineModel& model_;
  std::unique_ptr<VisionPipelineState> vision_state_;
  int64_t num_image_tokens_{0};
  std::string image_features_input_name_{};
  size_t image_features_input_index_{std::numeric_limits<size_t>::max()};
  bool decoder_consumes_image_features_{false};
};

}  // namespace Generators
