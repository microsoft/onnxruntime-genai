// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
//
// VibeVoice-Realtime text to speech (https://github.com/microsoft/VibeVoice).
//
// Device support: the five float16 graphs run on the configured execution provider; the loop keeps
// its tensors in CPU memory (validated on CPU).
//
// Graphs (all tensors float16 except token ids and masks, which are int64):
//   text_lm             input_ids [1, n] + past_key/value_<i>                   -> hidden_states [1, n, H] + present_*
//   tts_lm              inputs_embeds [1, n, H] + tts_text_mask [1, n] + past  -> hidden_states, eos_logits [1, 1] + present_*
//   diffusion_head      noisy_latent [1, L] + timestep [1] + condition [1, H]   -> predicted_noise [1, L]
//   acoustic_connector  latent [1, 1, L]                                         -> embedding [1, 1, H]
//   vocoder             latents [1, F, L]                                        -> audio [1, S] at 24 kHz
//
// A voice preset is an ONNX graph without inputs whose outputs are the KV caches of the text LM, the
// TTS LM and the negative (unconditioned) TTS LM after the voice prompt, and the last hidden state of
// the negative TTS LM. It is run once when the model loads.
//
// Driven by the Generator through the StepState interface. SetInputs passes OgaInputName_TextIds
// ([1, n] int64, from the VibeVoice processor). Each GenerateNextToken generates one speech frame:
// at the start of every speech window the next text window goes through the text LM into the TTS
// LM, then the diffusion head samples a latent with DPM-Solver++ and classifier-free guidance
// against the negative TTS LM, and the acoustic connector feeds the latent back to both TTS LMs.
// The utterance ends when the EOS classifier fires after all text is fed (or at the frame budget);
// the last step then decodes all latents with the vocoder into the OgaOutputName_Audio output.
#pragma once

#include <functional>
#include <map>
#include <random>
#include <span>

#include "model.h"
#include "step_state.h"

namespace Generators {

// The v-prediction of the diffusion model for `sample` at solver step `step`, with guidance applied.
using VelocityPredictor = std::function<std::vector<float>(size_t step, const std::vector<float>& sample)>;

// Denoises `noise` with multistep DPM-Solver++ for a v-prediction model: second order except for the
// first and the last step, as diffusers' DPMSolverMultistepScheduler that VibeVoice uses. `sigmas`
// has one entry more than steps; a final sigma of 0 returns the predicted clean sample.
std::vector<float> SampleDpmSolver(std::span<const double> sigmas, const std::vector<float>& noise,
                                   const VelocityPredictor& predict);

// Checks the model.vibevoice section and the voices of a VibeVoice config. Throws std::runtime_error.
void ValidateVibeVoiceConfig(const Config& config);

// One ONNX graph with its input and output names in the order of the values passed to Run. A
// language model also takes past_key_<i>/past_value_<i> after its inputs and returns
// present_key_<i>/present_value_<i> after its outputs.
struct VibeVoiceGraph {
  std::unique_ptr<OrtSession> session;
  std::vector<std::string> input_names, output_names;
  std::vector<const char*> inputs, outputs;
  size_t layers{};
  size_t head_outputs{};
};

struct VibeVoicePreset {
  std::vector<std::unique_ptr<OrtValue>> text_lm_kv;  // key_0, value_0, key_1, ...: [1, kv_heads, length, head_size]
  std::vector<std::unique_ptr<OrtValue>> tts_lm_kv;
  std::vector<std::unique_ptr<OrtValue>> negative_tts_lm_kv;
  std::vector<uint16_t> negative_condition;  // [hidden_size]
  int64_t text_lm_length{};
  int64_t tts_lm_length{};
  int64_t negative_tts_lm_length{};
};

struct VibeVoiceModel : Model {
  VibeVoiceModel(std::unique_ptr<Config> config, OrtEnv& ort_env);

  std::unique_ptr<State> CreateState(DeviceSpan<int32_t> sequence_lengths, const GeneratorParams& params) const override;

  // The named voice, or the default voice for an empty name. Throws for an unknown name.
  const VibeVoicePreset& GetVoice(std::string_view name) const;

  // Sessions and voices are immutable after construction, so generators can share the model.
  VibeVoiceGraph text_lm_, tts_lm_, diffusion_head_, acoustic_connector_, vocoder_;
  int64_t hidden_size_{};
  int64_t latent_size_{};
  std::unique_ptr<OrtMemoryInfo> memory_info_;

 private:
  void LoadVoices(OrtEnv& ort_env);

  std::map<std::string, VibeVoicePreset, std::less<>> voices_;
};

struct VibeVoiceState : StepState {
  VibeVoiceState(const VibeVoiceModel& model, const GeneratorParams& params);

  DeviceSpan<float> Run(int total_length, DeviceSpan<int32_t>& next_tokens, DeviceSpan<int32_t> next_indices = {}) override;
  // Checks the text of each SetInputs call; SetExtraInputs takes the latest text.
  void ValidateExtraInputs(std::span<const ExtraInput> extra_inputs) override;
  void SetExtraInputs(const std::vector<ExtraInput>& extra_inputs) override;
  void StepToken() override;

  // OgaRuntimeOption_Voice: the voice of the next utterance.
  void SetRunOption(const char* key, const char* value) override;
  bool SupportsRewind() const override { return false; }

  // OgaOutputName_Audio: float32 [1, num_samples] at 24 kHz, once the utterance is done.
  OrtValue* GetOutput(const char* name) override;

 private:
  struct KvCache {
    std::vector<const OrtValue*> past;
    std::vector<std::unique_ptr<OrtValue>> owned;
  };

  std::vector<std::unique_ptr<OrtValue>> RunGraph(const VibeVoiceGraph& graph, std::vector<const OrtValue*> inputs,
                                                  KvCache* cache = nullptr);
  std::unique_ptr<OrtValue> HalfView(std::span<const uint16_t> data, std::span<const int64_t> shape) const;
  std::unique_ptr<OrtValue> Int64View(std::span<int64_t> data) const;

  // The most speech frames an utterance of `tokens` text tokens may take. Throws when the prompt of
  // `voice`, the text and these frames do not fit the context of the TTS LM.
  size_t FrameBudget(size_t tokens, const VibeVoicePreset& voice) const;
  void StartUtterance();
  void FeedText();
  std::vector<float> SampleLatent();
  void FinishUtterance();

  const VibeVoiceModel& model_;
  const Config::Model::VibeVoice& config_;
  const VibeVoicePreset* voice_;
  std::mt19937 rng_;

  std::shared_ptr<Tensor> pending_text_;
  std::vector<int64_t> text_ids_;
  size_t text_fed_{};
  int window_frames_{};
  size_t frames_{};
  size_t max_frames_{};
  KvCache text_cache_, tts_cache_, negative_cache_;
  std::vector<uint16_t> condition_, negative_condition_;
  std::vector<float> latents_;
  std::unique_ptr<OrtValue> audio_;
};

}  // namespace Generators
