// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

#include "model.h"
#include "models/io/input_ids.h"
#include "models/io/multi_modal_features.h"
#include "models/io/embeddings.h"
#include "models/io/extra_inputs.h"
#include "models/io/logits.h"
#include "io/kv_cache.h"
#include "models/io/position_inputs.h"
#include "model_type.h"
#include "models/io/recurrent_state.h"

namespace Generators {

struct MultiModalLanguageModel : Model {
  MultiModalLanguageModel(std::unique_ptr<Config> config, OrtEnv& ort_env, bool vision, bool speech);
  MultiModalLanguageModel(const MultiModalLanguageModel&) = delete;
  MultiModalLanguageModel& operator=(const MultiModalLanguageModel&) = delete;

  std::unique_ptr<State> CreateState(DeviceSpan<int32_t> sequence_lengths, const GeneratorParams& params) const override;

  std::unique_ptr<OrtSession> vision_session_;     // pixel_values, [image_attention_mask], image_sizes -> image_features
  std::unique_ptr<OrtSession> speech_session_;     // audio_embeds, audio_sizes, audio_projection_mode -> audio_features
  std::unique_ptr<OrtSession> embedding_session_;  // input_ids, image_features, audio_features -> inputs_embeds
  std::unique_ptr<OrtSession> decoder_session_;    // inputs_embeds, attention_mask, kv_cache -> logits

  std::unique_ptr<OrtSessionOptions> vision_session_options_;
  std::unique_ptr<OrtSessionOptions> speech_session_options_;
  std::unique_ptr<OrtSessionOptions> embedding_session_options_;
};

// Base VisionState: runs vision.onnx with a single State::Run() call.
// Works for models whose vision encoder accepts batched input (Phi, Gemma).
struct VisionState : State {
  VisionState(const MultiModalLanguageModel& model, const GeneratorParams& params);
  VisionState(const VisionState&) = delete;
  VisionState& operator=(const VisionState&) = delete;

  virtual void SetExtraInputs(const std::vector<ExtraInput>& extra_inputs, const int64_t num_images, const int64_t num_image_tokens);
  DeviceSpan<float> Run(int current_length, DeviceSpan<int32_t>& next_tokens, DeviceSpan<int32_t> next_indices = {}) override;

 protected:
  friend struct MultiModalPipelineState;

  const MultiModalLanguageModel& model_;
  int64_t num_image_tokens_;
  int64_t num_images_{};
  ExtraInputs extra_inputs_{*this};  // Model inputs
  std::unique_ptr<MultiModalFeatures> image_features_;
  // Qwen3-VL DeepStack: extra per-token vision outputs (one per deepstack_visual_index).
  // Same shape/type as image_features; consumed by the embedding model.
  std::vector<std::unique_ptr<MultiModalFeatures>> deepstack_features_;
};

// QwenVisionState: per-image slicing loop for Qwen2.5-VL / Qwen3-VL.
//
// vision.onnx is exported for exactly one image (Dynamo unrolls Python
// for-loops at trace time, so an N-image dummy produces a graph that only
// works for that exact N).  This subclass iterates over images in C++,
// creating zero-copy sub-tensor views of pixel_values / image_grid_thw and
// writing each result into the correct offset of the pre-allocated
// image_features output buffer.
struct QwenVisionState : VisionState {
  using VisionState::VisionState;  // inherit constructor

  DeviceSpan<float> Run(int current_length, DeviceSpan<int32_t>& next_tokens, DeviceSpan<int32_t> next_indices = {}) override;
};

// PixtralVisionState: per-image vision loop for Pixtral / Mistral3.
//
// Each image is independently smart_resize'd to a different resolution.
// The preprocessor zero-pads all images to max(H) × max(W) and provides
// image_sizes[N, 2] with per-image (H, W).  This subclass slices
// pixel_values[i, :, :H_i, :W_i] for each image, runs vision.onnx with
// [1, C, H_i, W_i], and concatenates the resulting features.
struct PixtralVisionState : VisionState {
  using VisionState::VisionState;  // inherit constructor

  void SetExtraInputs(const std::vector<ExtraInput>& extra_inputs, const int64_t num_images, const int64_t num_image_tokens) override;
  DeviceSpan<float> Run(int current_length, DeviceSpan<int32_t>& next_tokens, DeviceSpan<int32_t> next_indices = {}) override;

 private:
  std::vector<int64_t> image_heights_;
  std::vector<int64_t> image_widths_;
};

inline void ValidateImageGridThwLayoutAndCount(const std::vector<int64_t>& shape,
                                               size_t elem_count,
                                               int64_t num_images,
                                               const char* tensor_name) {
  if (num_images < 0) {
    throw std::runtime_error(std::string(tensor_name) + " num_images must be non-negative");
  }

  if (shape.size() != 2) {
    throw std::runtime_error(std::string(tensor_name) + " must have rank 2 [num_images, 3]");
  }

  if (shape[0] < 0 || shape[1] < 0) {
    throw std::runtime_error(std::string(tensor_name) + " dimensions must be non-negative");
  }

  if (shape[1] != 3) {
    throw std::runtime_error(std::string(tensor_name) + " second dimension must be 3");
  }

  const size_t shape_image_count = static_cast<size_t>(shape[0]);
  const size_t expected_image_count = static_cast<size_t>(num_images);
  if (shape_image_count < expected_image_count) {
    throw std::runtime_error(std::string(tensor_name) + " shape[0] (" + std::to_string(shape_image_count) +
                             ") is less than required image count (" + std::to_string(expected_image_count) + ")");
  }

  if (elem_count % 3 != 0 || elem_count / 3 < expected_image_count) {
    throw std::runtime_error(std::string(tensor_name) + " element count (" + std::to_string(elem_count) +
                             ") is less than required for " + std::to_string(num_images) +
                             " images (need at least 3 values per image)");
  }
}

// Factory: pick the right VisionState subclass based on model type.
std::unique_ptr<VisionState> CreateVisionState(const MultiModalLanguageModel& model, const GeneratorParams& params);

struct SpeechState : State {
  SpeechState(const MultiModalLanguageModel& model, const GeneratorParams& params);
  SpeechState(const SpeechState&) = delete;
  SpeechState& operator=(const SpeechState&) = delete;

  void SetExtraInputs(const std::vector<ExtraInput>& extra_inputs, const int64_t num_audio_tokens);
  DeviceSpan<float> Run(int current_length, DeviceSpan<int32_t>& next_tokens, DeviceSpan<int32_t> next_indices = {}) override;

 private:
  friend struct MultiModalPipelineState;

  const MultiModalLanguageModel& model_;
  int64_t num_audio_tokens_;
  ExtraInputs extra_inputs_{*this};  // Model inputs
  std::unique_ptr<MultiModalFeatures> audio_features_;
};

struct EmbeddingState : State {
  EmbeddingState(const MultiModalLanguageModel& model, const GeneratorParams& params);
  EmbeddingState(const EmbeddingState&) = delete;
  EmbeddingState& operator=(const EmbeddingState&) = delete;

  void SetExtraInputs(const int64_t num_images_, const int64_t num_image_tokens_, const int64_t num_audio_tokens_);
  DeviceSpan<float> Run(int current_length, DeviceSpan<int32_t>& next_tokens, DeviceSpan<int32_t> next_indices = {});

 private:
  friend struct MultiModalPipelineState;

  void UpdateInputsOutputs(DeviceSpan<int32_t>& next_tokens, bool is_prompt);

  const MultiModalLanguageModel& model_;
  int64_t num_image_tokens_;
  int64_t num_audio_tokens_;

  DefaultInputIDs input_ids_{*this};                          // Model input
  std::unique_ptr<MultiModalFeatures> image_features_;        // Optional model input
  std::unique_ptr<MultiModalFeatures> audio_features_;        // Optional model input
  Embeddings inputs_embeds_{*this, Embeddings::Mode::Output,  // Model output
                            model_.config_->model.embedding.outputs.embeddings};
  std::unique_ptr<Embeddings> per_layer_inputs_;  // Optional model output (Gemma4)
  // Qwen3-VL DeepStack: per-token features in (from vision), full-length scattered features out
  // (to decoder). The scatter (input_ids==image_token) is done inside embedding.onnx.
  std::vector<std::unique_ptr<MultiModalFeatures>> deepstack_features_in_;  // Optional model inputs
  std::vector<std::unique_ptr<Embeddings>> deepstack_out_;                  // Model outputs
};

struct DecoderState : State {
  DecoderState(const MultiModalLanguageModel& model, DeviceSpan<int32_t> sequence_lengths,
               const GeneratorParams& params);
  DecoderState(const DecoderState&) = delete;
  DecoderState& operator=(const DecoderState&) = delete;

  DeviceSpan<float> Run(int current_length, DeviceSpan<int32_t>& next_tokens, DeviceSpan<int32_t> next_indices) override;
  void UpdateInputsOutputs(DeviceSpan<int32_t>& next_tokens, int current_length, DeviceSpan<int32_t> beam_indices);

  // Prefill chunking (see search.chunk_size). The embedding model still runs once over the whole
  // prompt (it is a lookup/projection), while the decoder prefill is split into several runs so the
  // peak attention workspace scales with the chunk size instead of the full prompt length.
  bool SupportsPrefillChunking() const;
  void PrepareEmbeddingsForPrefill(size_t new_length);
  DeviceSpan<float> RunPrefillWithChunking(int current_length, DeviceSpan<int32_t>& next_tokens,
                                           DeviceSpan<int32_t> next_indices, size_t chunk_size);

 private:
  friend struct MultiModalPipelineState;

  void UpdateInputsOutputs(DeviceSpan<int32_t>& next_tokens, int current_length, DeviceSpan<int32_t> beam_indices, size_t new_length);

  const MultiModalLanguageModel& model_;
  Embeddings inputs_embeds_{*this, Embeddings::Mode::Input,  // Model input
                            model_.config_->model.decoder.inputs.embeddings};
  std::unique_ptr<Embeddings> per_layer_inputs_;        // Optional model input (Gemma4: per-layer conditioning)
  std::unique_ptr<DefaultInputIDs> decoder_input_ids_;  // Optional model input (e.g., Gemma4 decoder needs input_ids)
  // Qwen3-VL DeepStack: full-length scattered features injected after decoder layers 0/1/2.
  std::vector<std::unique_ptr<Embeddings>> deepstack_in_;  // Model inputs
  std::unique_ptr<PositionInputs> position_inputs_;     // Model input
  std::unique_ptr<KeyValueCache> kv_cache_;             // Model input
  std::unique_ptr<RecurrentState> recurrent_state_;     // Model input (for hybrid models)
  Logits logits_{*this};                                // Model output
};

struct MultiModalPipelineState : State {
  MultiModalPipelineState(const MultiModalLanguageModel& model, DeviceSpan<int32_t> sequence_lengths,
                          const GeneratorParams& params);
  MultiModalPipelineState(const MultiModalPipelineState&) = delete;
  MultiModalPipelineState& operator=(const MultiModalPipelineState&) = delete;

  void SetExtraInputs(const std::vector<ExtraInput>& extra_inputs) override;

  DeviceSpan<float> Run(int current_length, DeviceSpan<int32_t>& next_tokens,
                        DeviceSpan<int32_t> next_indices) override;

  OrtValue* GetInput(const char* name) override;

  OrtValue* GetOutput(const char* name) override;

 private:
  void UpdateInputsOutputs(const DeviceSpan<int32_t>& next_tokens, DeviceSpan<int32_t> next_indices,
                           int current_length);

  const MultiModalLanguageModel& model_;
  int64_t num_image_tokens_{};
  int64_t num_audio_tokens_{};
  int64_t num_images_{};
  std::unique_ptr<VisionState> vision_state_;
  std::unique_ptr<SpeechState> speech_state_;
  std::unique_ptr<EmbeddingState> embedding_state_;
  std::unique_ptr<DecoderState> decoder_state_;
  std::shared_ptr<Adapters> adapters_;
  bool is_prompt_{true};

  const std::string vision_adapter_name_{"vision"};
  const std::string speech_adapter_name_{"speech"};
};

}  // namespace Generators
