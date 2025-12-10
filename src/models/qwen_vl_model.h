#pragma once

#include "decoder_only_pipeline.h"
#include "qwen_vl_vision.h"

namespace Generators {

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
  void OnStageComplete(size_t stage_id, DeviceSpan<int32_t>& next_tokens) override;

 private:
  void InjectVisionEmbeddings(const std::string& embeddings_output_name,
                              DeviceSpan<int32_t>& input_token_ids);

  const Qwen2_5_VL_PipelineModel& vl_model_;
  bool vision_ran_{false};
  std::unique_ptr<OrtValue> image_features_value_;
  std::vector<float> image_features_buffer_;  // backing storage for OrtValue
  size_t image_embed_consumed_{0};            // Track how many vision embeddings we've injected
};

}  // namespace Generators
