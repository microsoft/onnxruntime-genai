// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include "model.h"
#include "input_ids.h"
#include "multi_modal_features.h"
#include "embeddings.h"
#include "extra_inputs.h"
#include "logits.h"
#include "kv_cache.h"
#include "position_inputs.h"
#include "decoder_only_pipeline.h"

namespace Generators {

struct MultiModalPipelineLanguageModel : Model {
  MultiModalPipelineLanguageModel(std::unique_ptr<Config> config, OrtEnv& ort_env, bool vision, bool speech);
  MultiModalPipelineLanguageModel(const MultiModalPipelineLanguageModel&) = delete;
  MultiModalPipelineLanguageModel& operator=(const MultiModalPipelineLanguageModel&) = delete;

  std::unique_ptr<State> CreateState(DeviceSpan<int32_t> sequence_lengths, const GeneratorParams& params) const;

  std::unique_ptr<OrtSession> vision_session_;     // pixel_values, [image_attention_mask], image_sizes -> image_features
  std::unique_ptr<OrtSession> speech_session_;     // audio_embeds, audio_sizes, audio_projection_mode -> audio_features
  std::unique_ptr<OrtSession> embedding_session_;  // input_ids, image_features, audio_features -> inputs_embeds

  std::vector<std::unique_ptr<OrtSession>> decoder_pipeline_sessions_;
  OrtEnv& ort_env_;
};

struct VisionPipelineState : State {
  VisionPipelineState(const MultiModalPipelineLanguageModel& model, const GeneratorParams& params);
  VisionPipelineState(const VisionPipelineState&) = delete;
  VisionPipelineState& operator=(const VisionPipelineState&) = delete;

  void SetExtraInputs(const std::vector<ExtraInput>& extra_inputs, const int64_t num_images, const int64_t num_image_tokens);
  DeviceSpan<float> Run(int current_length, DeviceSpan<int32_t>& next_tokens, DeviceSpan<int32_t> next_indices = {}) override;

 private:
  friend struct MultiModalDecoderPipelineState;

  const MultiModalPipelineLanguageModel& model_;
  int64_t num_image_tokens_;
  int64_t num_images_{};
  std::shared_ptr<Tensor> pixel_values_tensor_;
  ExtraInputs extra_inputs_{*this};  // Model inputs
  std::unique_ptr<MultiModalFeatures> image_features_;
};

struct SpeechPipelineState : State {
  SpeechPipelineState(const MultiModalPipelineLanguageModel& model, const GeneratorParams& params);
  SpeechPipelineState(const SpeechPipelineState&) = delete;
  SpeechPipelineState& operator=(const SpeechPipelineState&) = delete;

  void SetExtraInputs(const std::vector<ExtraInput>& extra_inputs, const int64_t num_audio_tokens);
  DeviceSpan<float> Run(int current_length, DeviceSpan<int32_t>& next_tokens, DeviceSpan<int32_t> next_indices = {}) override;

 private:
  friend struct MultiModalDecoderPipelineState;

  const MultiModalPipelineLanguageModel& model_;
  int64_t num_audio_tokens_;
  ExtraInputs extra_inputs_{*this};  // Model inputs
  std::unique_ptr<MultiModalFeatures> audio_features_;
};

struct EmbeddingPipelineState : State {
  EmbeddingPipelineState(const MultiModalPipelineLanguageModel& model, const GeneratorParams& params);
  EmbeddingPipelineState(const EmbeddingPipelineState&) = delete;
  EmbeddingPipelineState& operator=(const EmbeddingPipelineState&) = delete;

  void SetExtraInputs(const int64_t num_images_, const int64_t num_image_tokens_, const int64_t num_audio_tokens_);
  DeviceSpan<float> Run(int current_length, DeviceSpan<int32_t>& next_tokens, DeviceSpan<int32_t> next_indices = {});

 private:
  friend struct MultiModalDecoderPipelineState;

  void UpdateInputsOutputs(DeviceSpan<int32_t>& next_tokens, bool is_prompt);

  const MultiModalPipelineLanguageModel& model_;
  int64_t num_image_tokens_;
  int64_t num_audio_tokens_;

  DefaultInputIDs input_ids_{*this};                          // Model input
  std::unique_ptr<MultiModalFeatures> image_features_;        // Optional model input
  std::unique_ptr<MultiModalFeatures> audio_features_;        // Optional model input
  std::unique_ptr <Embeddings> inputs_embeds_;
};

struct IntermediateDecoderPipelineState : State {
  IntermediateDecoderPipelineState(const MultiModalPipelineLanguageModel& model, const GeneratorParams& params,
                            size_t pipeline_state_index);

  IntermediateDecoderPipelineState(const IntermediateDecoderPipelineState&) = delete;
  IntermediateDecoderPipelineState& operator=(const IntermediateDecoderPipelineState&) = delete;

  DeviceSpan<float> Run(int current_length, DeviceSpan<int32_t>& next_tokens,
                        DeviceSpan<int32_t> next_indices) override;

  bool HasInput(std::string_view name) const;

  bool HasOutput(std::string_view name) const;

  bool SupportsPrimaryDevice() const;

  size_t id_;

 private:
  const MultiModalPipelineLanguageModel& model_;
};

struct DecoderPipelineState : State {
  DecoderPipelineState(const MultiModalPipelineLanguageModel& model, DeviceSpan<int32_t> sequence_lengths,
                           const GeneratorParams& params);

  DecoderPipelineState(const DecoderPipelineState&) = delete;
  DecoderPipelineState& operator=(const DecoderPipelineState&) = delete;

  void SetExtraInputs(const int64_t num_images, const int64_t num_image_tokens,  const std::vector<ExtraInput>& extra_inputs);

  DeviceSpan<float> Run(int total_length, DeviceSpan<int32_t>& next_tokens,
                    DeviceSpan<int32_t> next_indices) override;

  OrtValue* GetOutput(const char* name) override;

  void RunPipeline(int total_length, DeviceSpan<int32_t>& next_tokens,
                   DeviceSpan<int32_t> next_indices);

  std::unique_ptr<MultiModalFeatures> image_features_;  // model input

  std::unique_ptr<Embeddings> full_inputs_embeds_;

  std::unique_ptr<Embeddings> inputs_embeds_;
 private:
  void UpdateInputsOutputs(DeviceSpan<int32_t>& next_tokens, DeviceSpan<int32_t> next_indices, Embeddings& embeddings,
                           int total_length);

  void UpdateKeyValueCache(DeviceSpan<int32_t> beam_indices, int total_length);

  
  const MultiModalPipelineLanguageModel& model_;
  std::vector<std::unique_ptr<IntermediateDecoderPipelineState>> pipeline_states_;

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

struct MultiModalDecoderPipelineState : State {
  MultiModalDecoderPipelineState(const MultiModalPipelineLanguageModel& model, DeviceSpan<int32_t> sequence_lengths,
                          const GeneratorParams& params);
  MultiModalDecoderPipelineState(const MultiModalDecoderPipelineState&) = delete;
  MultiModalDecoderPipelineState& operator=(const MultiModalDecoderPipelineState&) = delete;

  void SetExtraInputs(const std::vector<ExtraInput>& extra_inputs) override;

  DeviceSpan<float> Run(int current_length, DeviceSpan<int32_t>& next_tokens,
                        DeviceSpan<int32_t> next_indices) override;

  OrtValue* GetInput(const char* name) override;

  OrtValue* GetOutput(const char* name) override;

 private:
  void UpdateInputsOutputs(const DeviceSpan<int32_t>& next_tokens, DeviceSpan<int32_t> next_indices,
                           int current_length);

  const MultiModalPipelineLanguageModel& model_;
  int64_t num_image_tokens_{};
  int64_t num_audio_tokens_{};
  int64_t num_images_{};
  std::unique_ptr<VisionPipelineState> vision_state_;
  std::unique_ptr<SpeechPipelineState> speech_state_;
  std::unique_ptr<EmbeddingPipelineState> embedding_state_;
  std::unique_ptr<DecoderPipelineState> decoder_pipeline_state_;
  std::shared_ptr<Adapters> adapters_;
  bool is_prompt_{true};

  const std::string vision_adapter_name_{"vision"};
  const std::string speech_adapter_name_{"speech"};
};

}  // namespace Generators
