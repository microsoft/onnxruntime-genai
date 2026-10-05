#pragma once

#include "span.h"

#include "decoder_only_pipeline.h"
#include "qwen_vl_vision.h"

namespace Generators {

inline void ValidateVisionEmbeddingShapes(std::span<const int64_t> embeddings_shape,
                                          size_t embeddings_element_count,
                                          std::span<const int64_t> vision_shape,
                                          size_t input_token_count) {
  if (embeddings_shape.size() != 2 && embeddings_shape.size() != 3) {
    throw std::runtime_error("Vision embedding injection: expected embeddings rank 2 or 3, got " +
                             std::to_string(embeddings_shape.size()));
  }
  if (vision_shape.size() != 2) {
    throw std::runtime_error("Vision embedding injection: expected vision features rank 2, got " +
                             std::to_string(vision_shape.size()));
  }

  const int64_t embedding_dim = embeddings_shape.back();
  const int64_t vision_dim = vision_shape[1];
  if (embedding_dim <= 0 || vision_dim != embedding_dim) {
    throw std::runtime_error("Vision embedding injection: dimension mismatch - vision_dim=" + std::to_string(vision_dim) +
                             ", embedding_dim=" + std::to_string(embedding_dim));
  }
  const size_t token_capacity = embeddings_element_count / static_cast<size_t>(embedding_dim);
  if (input_token_count > token_capacity) {
    throw std::runtime_error(
        "Vision embedding injection: embeddings output cannot hold all input tokens "
        "(input_token_count=" +
        std::to_string(input_token_count) + ", capacity=" + std::to_string(token_capacity) + ")");
  }
}

// Qwen2.5-VL pipeline model integrating vision pipeline + decoder pipeline.
// Loads decoder pipeline sessions (handled by base) and constructs vision pipeline sessions.
// State runs vision once (on first SetExtraInputs when pixel_values arrives) to produce image_features
// which are injected into embeddings output via existing injection logic in DecoderOnlyPipelineState.
struct Qwen2_5_VL_PipelineModel : public DecoderOnlyPipelineModel {
  Qwen2_5_VL_PipelineModel(std::unique_ptr<Config> config, OrtEnv& ort_env);

  std::unique_ptr<State> CreateState(DeviceSpan<int32_t> sequence_lengths,
                                     const GeneratorParams& params) const override;

  // Vision pipeline shared across states (sessions reused).
  std::unique_ptr<QwenVisionPipeline> vision_pipeline_;

  // Gemma-4 vision uses either a single session or an encoder and projector pair.
  std::unique_ptr<OrtSessionOptions> vision_session_options_;
  std::unique_ptr<OrtSession> vision_session_;
  std::unique_ptr<OrtSessionOptions> vision_projector_session_options_;
  std::unique_ptr<OrtSession> vision_projector_session_;
};

struct Qwen2_5_VL_PipelineState : public DecoderOnlyPipelineState {
  Qwen2_5_VL_PipelineState(const Qwen2_5_VL_PipelineModel& model,
                           DeviceSpan<int32_t> sequence_lengths,
                           const GeneratorParams& params);

  void SetExtraInputs(const std::vector<ExtraInput>& extra_inputs) override;

  DeviceSpan<float> Run(int total_length, DeviceSpan<int32_t>& next_tokens,
                        DeviceSpan<int32_t> next_indices) override;

 protected:
  void OnStageStart(size_t stage_id) override;
  void OnStageComplete(size_t stage_id) override;

 private:
  void InjectVisionEmbeddings(const std::string& embeddings_output_name);

  // Runs the Gemma-4 vision session(s) and publishes image_features_value_.
  void RunSingleSessionVision(const std::vector<ExtraInput>& extra_inputs);

  // Runs whichever vision path this model was configured with, at most once.
  void RunVision(const std::vector<ExtraInput>& extra_inputs);

  // Registers image_features/audio_features as managed pipeline inputs.
  void InitializeFeatureInputs();

  // Selects the image feature rows belonging to the embedding stage's current token window.
  void UpdateImageFeatureInput();

  // Creates an empty feature tensor matching the embedding input's rank, width, and dtype.
  std::unique_ptr<OrtValue> CreateEmptyFeatureInput(const std::string& name) const;

  int32_t GetImageTokenId() const;

  const Qwen2_5_VL_PipelineModel& vl_model_;
  bool vision_ran_{false};
  std::unique_ptr<OrtValue> image_features_value_;
  std::vector<float> image_features_buffer_;       // backing storage for OrtValue
  std::unique_ptr<OrtValue> vision_output_owner_;  // keeps the encoder's output alive when
                                                   // image_features_value_ is a reshaped view of it
  size_t image_embed_consumed_{0};                 // Track how many vision embeddings we've injected
  bool embedding_merges_features_{false};          // embedding graph does the merge, so skip injection
  size_t image_feature_input_index_{SIZE_MAX};
  size_t audio_feature_input_index_{SIZE_MAX};
  size_t image_feature_offset_{0};
  std::unique_ptr<OrtValue> image_feature_input_;
  std::unique_ptr<OrtValue> image_feature_cast_;
  std::unique_ptr<OrtValue> audio_feature_input_;
};

}  // namespace Generators
