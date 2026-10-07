// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <cstddef>
#include <cstdint>
#include <optional>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

#include "model.h"
#include "models/io/input_ids.h"
#include "models/io/multi_modal_features.h"
#include "models/io/embeddings.h"
#include "models/io/extra_inputs.h"
#include "models/io/hidden_states.h"
#include "models/io/logits.h"
#include "models/io/indexer_cache.h"
#include "models/io/ple_state.h"
#include "io/kv_cache.h"
#include "models/io/position_inputs.h"
#include "model_type.h"
#include "models/io/recurrent_state.h"
#include "models/lfm2_audio_output.h"
#include "engram.h"

namespace Generators {

struct MultiModalLanguageModel : Model {
  MultiModalLanguageModel(std::unique_ptr<Config> config, OrtEnv& ort_env, bool vision, bool speech);
  MultiModalLanguageModel(const MultiModalLanguageModel&) = delete;
  MultiModalLanguageModel& operator=(const MultiModalLanguageModel&) = delete;

  std::unique_ptr<State> CreateState(DeviceSpan<int32_t> sequence_lengths, const GeneratorParams& params) const override;

  std::unique_ptr<OrtSession> vision_session_;     // pixel_values, [image_attention_mask], image_sizes -> image_features
  std::unique_ptr<OrtSession> speech_session_;     // audio_embeds, audio_sizes, audio_projection_mode -> audio_features
  std::unique_ptr<OrtSession> embedding_session_;  // input_ids, image_features, audio_features -> inputs_embeds
  std::unique_ptr<OrtSession> engram_session_;     // input_ids, token history -> engram_embeddings
  std::unique_ptr<OrtSession> decoder_session_;    // inputs_embeds, attention_mask, kv_cache -> logits

  std::unique_ptr<OrtSessionOptions> vision_session_options_;
  std::unique_ptr<OrtSessionOptions> speech_session_options_;
  std::unique_ptr<OrtSessionOptions> embedding_session_options_;
  std::unique_ptr<OrtSessionOptions> engram_session_options_;

  // LFM2-Audio speech output, present when model.audio_output names the two graphs.
  std::unique_ptr<OrtSession> depthformer_session_;      // hidden_states -> one audio code per run, a frame in num_codebooks runs
  std::unique_ptr<OrtSession> audio_embedding_session_;  // audio_codes -> audio_embeds, summed into the decoder's next input
  std::unique_ptr<OrtSessionOptions> depthformer_session_options_;
  std::unique_ptr<OrtSessionOptions> audio_embedding_session_options_;

  // The device each sub-model session actually runs on. A sub-model whose config block carries
  // its own `session_options` does not inherit the decoder's providers, so it can land on the CPU
  // EP while the decoder is on a GPU one. The states below allocate against these, not p_device_.
  DeviceInterface* vision_device_{};
  DeviceInterface* speech_device_{};
  DeviceInterface* embedding_device_{};
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
};

// Factory: pick the right VisionState subclass based on model type.
std::unique_ptr<VisionState> CreateVisionState(const MultiModalLanguageModel& model, const GeneratorParams& params);

struct SpeechState : State {
  SpeechState(const MultiModalLanguageModel& model, const GeneratorParams& params);
  SpeechState(const SpeechState&) = delete;
  SpeechState& operator=(const SpeechState&) = delete;

  virtual void SetExtraInputs(const std::vector<ExtraInput>& extra_inputs, const int64_t num_audio_tokens);
  DeviceSpan<float> Run(int current_length, DeviceSpan<int32_t>& next_tokens, DeviceSpan<int32_t> next_indices = {}) override;

 protected:
  friend struct MultiModalPipelineState;

  const MultiModalLanguageModel& model_;
  int64_t num_audio_tokens_;
  ExtraInputs extra_inputs_{*this};  // Model inputs
  std::unique_ptr<MultiModalFeatures> audio_features_;
};

// Lfm2AudioSpeechState: per-clip encoder loop for LFM2-Audio.
//
// The processor stacks the clips of one prompt into a zero-padded [N, T_max, num_mels] mel tensor
// with their real frame counts alongside. The published encoder export is traced for a single clip
// (its subsampling mask cannot broadcast over a batch), so with several clips this subclass slices
// each clip's own frames out of that tensor, runs the encoder on [1, T_i, num_mels], and writes the
// results one after another into the contiguous [1, total_tokens, hidden] feature buffer the
// embedding model expects, in clip order.
struct Lfm2AudioSpeechState : SpeechState {
  using SpeechState::SpeechState;  // inherit constructor

  void SetExtraInputs(const std::vector<ExtraInput>& extra_inputs, const int64_t num_audio_tokens) override;
  DeviceSpan<float> Run(int current_length, DeviceSpan<int32_t>& next_tokens, DeviceSpan<int32_t> next_indices = {}) override;

 private:
  // Where the processor's staged mel tensor and the encoder's feature buffer are bound.
  struct SpeechBindings {
    size_t mel_index{};       // the [num_clips, longest, num_mels] staging tensor
    size_t lengths_index{};   // its per-clip frame counts
    size_t features_index{};  // the [1, total_tokens, hidden] buffer the embedding model reads
    int64_t num_clips{};
    int64_t longest_clip{};
    int64_t num_mels{};
    int64_t hidden_size{};
    ONNXTensorElementDataType mel_type{};
    ONNXTensorElementDataType features_type{};
  };

  // Reads the bound shapes and checks them against the per-clip token counts.
  SpeechBindings ResolveBindings() const;

  // Runs the encoder on clip `index`'s own frames and returns its features.
  std::unique_ptr<OrtValue> RunClip(const SpeechBindings& bindings, int64_t index, int64_t num_frames);

  size_t FindInput(const std::string& name) const;
  size_t FindOutput(const std::string& name) const;

  std::vector<int64_t> tokens_per_clip_;
};

// Factory: pick the right SpeechState subclass based on model type.
std::unique_ptr<SpeechState> CreateSpeechState(const MultiModalLanguageModel& model, const GeneratorParams& params);

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
};

struct DecoderState : State {
  DecoderState(const MultiModalLanguageModel& model, DeviceSpan<int32_t> sequence_lengths,
               const GeneratorParams& params);
  DecoderState(const DecoderState&) = delete;
  DecoderState& operator=(const DecoderState&) = delete;

  DeviceSpan<float> Run(int current_length, DeviceSpan<int32_t>& next_tokens, DeviceSpan<int32_t> next_indices) override;
  void RewindTo(size_t index) override;
  void SnapshotState(size_t position) override;
  bool HasCroppableRecurrentState() const override;
  int64_t RecurrentStateWindow() const override;
  void CropToAccepted(size_t new_length, size_t recurrent_position) override;
  void UpdateInputsOutputs(DeviceSpan<int32_t>& next_tokens, int current_length, DeviceSpan<int32_t> beam_indices);

  // Prefill chunking (see search.chunk_size). The embedding model still runs once over the whole
  // prompt (it is a lookup/projection), while the decoder prefill is split into several runs so the
  // peak attention workspace scales with the chunk size instead of the full prompt length.
  bool SupportsPrefillChunking(bool has_multimodal_content) const;
  void PrepareEmbeddingsForPrefill(size_t new_length);
  DeviceSpan<float> RunPrefillWithChunking(int current_length, DeviceSpan<int32_t>& next_tokens,
                                           DeviceSpan<int32_t> next_indices, size_t chunk_size,
                                           EngramState* engram_state);

 private:
  friend struct MultiModalPipelineState;

  void UpdateInputsOutputs(DeviceSpan<int32_t>& next_tokens, int current_length, DeviceSpan<int32_t> beam_indices, size_t new_length);

  const MultiModalLanguageModel& model_;
  Embeddings inputs_embeds_{*this, Embeddings::Mode::Input,  // Model input
                            model_.config_->model.decoder.inputs.embeddings};
  std::unique_ptr<Embeddings> per_layer_inputs_;        // Optional model input (Gemma4: per-layer conditioning)
  std::unique_ptr<Embeddings> engram_embeddings_;       // Optional CPU Engram output staged on CUDA
  std::unique_ptr<DefaultInputIDs> decoder_input_ids_;  // Optional model input (e.g., Gemma4 decoder needs input_ids)
  std::unique_ptr<PositionInputs> position_inputs_;     // Model input
  std::unique_ptr<KeyValueCache> kv_cache_;             // Model input
  std::unique_ptr<RecurrentState> recurrent_state_;     // Model input (for hybrid models)
  std::unique_ptr<PleState> ple_state_;                  // Model input (Qwen4-Exp PLE)
  std::unique_ptr<IndexerCache> indexer_cache_;          // Model input (Qwen4-Exp QSA)
  std::unique_ptr<HiddenStatesOutputs> hidden_states_output_;
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
  void RewindTo(size_t index) override;
  void SnapshotState(size_t position) override;
  bool HasCroppableRecurrentState() const override;
  int64_t RecurrentStateWindow() const override;
  void CropToAccepted(size_t new_length, size_t recurrent_position) override;

  OrtValue* GetInput(const char* name) override;

  OrtValue* GetOutput(const char* name) override;

 private:
  void UpdateInputsOutputs(const DeviceSpan<int32_t>& next_tokens, DeviceSpan<int32_t> next_indices,
                           int current_length);
  // The decoder's logits while the answer is text, or the next audio frame's placeholder while it is speech.
  DeviceSpan<float> SampleAudioOrText(DeviceSpan<float> logits);

  const MultiModalLanguageModel& model_;
  int64_t num_image_tokens_{};
  int64_t num_audio_tokens_{};
  int64_t num_images_{};
  std::unique_ptr<VisionState> vision_state_;
  std::unique_ptr<SpeechState> speech_state_;
  std::unique_ptr<EmbeddingState> embedding_state_;
  std::unique_ptr<EngramState> engram_state_;
  std::unique_ptr<DecoderState> decoder_state_;
  std::unique_ptr<Lfm2AudioOutput> audio_output_;  // LFM2-Audio speech output, when the model has it
  std::shared_ptr<Adapters> adapters_;
  bool is_prompt_{true};

  const std::string vision_adapter_name_{"vision"};
  const std::string speech_adapter_name_{"speech"};
};

}  // namespace Generators
