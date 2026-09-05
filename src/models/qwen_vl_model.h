#pragma once

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
  if (input_token_count > embeddings_element_count / static_cast<size_t>(embedding_dim)) {
    throw std::runtime_error("Vision embedding injection: embeddings output cannot hold all input tokens");
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
};

struct Qwen2_5_VL_PipelineState : public DecoderOnlyPipelineState {
  Qwen2_5_VL_PipelineState(const Qwen2_5_VL_PipelineModel& model,
                           DeviceSpan<int32_t> sequence_lengths,
                           const GeneratorParams& params);

  void SetExtraInputs(const std::vector<ExtraInput>& extra_inputs) override;

 protected:
  void OnStageComplete(size_t stage_id) override;

 private:
  void InjectVisionEmbeddings(const std::string& embeddings_output_name);

  const Qwen2_5_VL_PipelineModel& vl_model_;
  bool vision_ran_{false};
  std::unique_ptr<OrtValue> image_features_value_;
  std::vector<float> image_features_buffer_;  // backing storage for OrtValue
  size_t image_embed_consumed_{0};            // Track how many vision embeddings we've injected
};

}  // namespace Generators
