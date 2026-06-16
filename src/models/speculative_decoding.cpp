// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#include <algorithm>
#include "../generators.h"
#include "../speculative_sampling.h"
#include "speculative_decoding.h"

namespace Generators {

namespace {

std::unique_ptr<Config> CloneConfigForDraft(const Config& source) {
  auto cfg = std::make_unique<Config>(source);
  cfg->model.decoder = source.model.draft;
  return cfg;
}

std::unique_ptr<Config> CloneConfigForTarget(const Config& source) {
  return std::make_unique<Config>(source);
}

bool ProviderConfigurationMatches(const Config::SessionOptions& a,
                                  const Config::SessionOptions& b) {
  if (a.providers.size() != b.providers.size())
    return false;
  for (size_t i = 0; i < a.providers.size(); ++i)
    if (a.providers[i] != b.providers[i])
      return false;
  if (a.provider_options.size() != b.provider_options.size())
    return false;
  for (size_t i = 0; i < a.provider_options.size(); ++i)
    if (a.provider_options[i].name != b.provider_options[i].name)
      return false;
  return true;
}

void ValidateLogitsDimensionsMatch(const DecoderOnly_Model& target,
                                   const DecoderOnly_Model& draft) {
  const auto& tn = target.config_->model.decoder.outputs.logits;
  const auto& dn = draft.config_->model.decoder.outputs.logits;
  if (!target.session_info_.HasOutput(tn) || !draft.session_info_.HasOutput(dn))
    return;
  const auto ts = target.session_info_.GetOutputShape(tn);
  const auto ds = draft.session_info_.GetOutputShape(dn);
  if (ts.empty() || ds.empty())
    return;
  int64_t tv = ts.back(), dv = ds.back();
  if (tv <= 0 || dv <= 0)
    return;
  if (tv != dv)
    throw std::runtime_error(
        "Target and draft logit dimensions don't match. Target vocab: " +
        std::to_string(tv) + ", Draft vocab: " + std::to_string(dv) +
        ". Target and draft must share the same vocabulary.");
}

}  // namespace

// SpeculativeDecodingModel
SpeculativeDecodingModel::SpeculativeDecodingModel(std::unique_ptr<Config> config, OrtEnv& ort_env)
    : Model{std::move(config)} {
  if (config_->model.draft.filename.empty())
    throw std::runtime_error(
        "model.type is \"speculative\" but model.draft.filename is not set in genai_config.json.");

  if (!ProviderConfigurationMatches(config_->model.decoder.session_options,
                                    config_->model.draft.session_options))
    throw std::runtime_error(
        "Target and draft must use the same execution provider. "
        "Cross-EP speculative decoding is not supported in this release.");

  // v0 scope: target and draft must be plain decoder-only LLMs. No Pipeline/multimodal (reject upfront).
  if (!config_->model.decoder.pipeline.empty() || !config_->model.draft.pipeline.empty())
    throw std::runtime_error(
        "Speculative decoding does not support pipeline models in this release; "
        "target and draft must be plain decoder-only LLMs.");
  if (!config_->model.vision.filename.empty() || !config_->model.speech.filename.empty())
    throw std::runtime_error(
        "Speculative decoding does not support multimodal (vision/audio) models in this release; "
        "target and draft must be plain decoder-only LLMs.");

  // Sliding-window and LFM2 caches discard old K/V by design, so RewindTo throws.
  // Speculative decoding rewinds on every rejection, so reject these up front (checked on both target and draft).
  auto uses_sliding_kv = [](const Config::Model::Decoder& d) {
    return d.sliding_window.has_value() && d.sliding_window->slide_key_value_cache;
  };
  if (uses_sliding_kv(config_->model.decoder) || uses_sliding_kv(config_->model.draft))
    throw std::runtime_error(
        "Speculative decoding does not support sliding-window KV cache models in this release; "
        "rewind is impossible once the window slides.");
  if (!config_->model.decoder.layer_types.empty() || !config_->model.draft.layer_types.empty())
    throw std::runtime_error(
        "Speculative decoding does not support LFM2 (hybrid SSM/attention) models in this release; "
        "their rolling convolution state cannot be rewound.");

  target_model_ = std::make_shared<DecoderOnly_Model>(CloneConfigForTarget(*config_), ort_env);
  draft_model_ = std::make_shared<DecoderOnly_Model>(CloneConfigForDraft(*config_), ort_env);
  ValidateLogitsDimensionsMatch(*target_model_, *draft_model_);
  session_info_.Add(*target_model_->session_decoder_);
}

std::unique_ptr<State> SpeculativeDecodingModel::CreateState(DeviceSpan<int32_t> sequence_lengths,
                                                             const GeneratorParams& params) const {
  return std::make_unique<SpeculativeDecodingState>(*this, sequence_lengths, params);
}

// SpeculativeDecodingState — construction
SpeculativeDecodingState::SpeculativeDecodingState(const SpeculativeDecodingModel& model,
                                                   DeviceSpan<int32_t> sequence_lengths,
                                                   const GeneratorParams& params)
    : State{params, model},
      model_{model},
      target_state_{model.target_model().CreateState(sequence_lengths, params)},
      draft_state_{model.draft_model().CreateState(sequence_lengths, params)} {
  // No support for batch_size > 1 in this release.
  if (params.search.batch_size != 1)
    throw std::runtime_error(
        "Speculative decoding does not support batch_size > 1 in this release. Got batch_size=" +
        std::to_string(params.search.batch_size));

  // No support for num_beams > 1 (beam search) in the speculative loop in this release.
  if (params.search.num_beams != 1)
    throw std::runtime_error(
        "Speculative decoding does not support num_beams > 1 (beam search). Got num_beams=" +
        std::to_string(params.search.num_beams) + ".");

  // No support for repetition_penalty and min_length; 
  // needs cross-position bookkeeping that isn't implemented yet.
  if (params.search.repetition_penalty != 1.0f)
    throw std::runtime_error(
        "Speculative decoding does not support repetition_penalty != 1.0 in this release. Got " +
        std::to_string(params.search.repetition_penalty) + ".");
  if (params.search.min_length > 0)
    throw std::runtime_error(
        "Speculative decoding does not support min_length > 0 in this release. Got min_length=" +
        std::to_string(params.search.min_length) + ".");

  // No support for guidance in this release; applies a grammar mask and commits tokens one at a time. The speculative
  // loop bypasses Generator::ComputeLogits, so guidance would be silently ignored -> reject it.
  if (!params.guidance_type.empty() && !params.guidance_data.empty())
    throw std::runtime_error(
        "Speculative decoding does not support constrained decoding (guidance) in this release.");
}

// Run() - prefill path (called via Generator::AppendTokens -> ComputeLogits).
// Runs both inner states on the prompt, saves draft's pending distribution for
// the next position, and returns target's logits.
DeviceSpan<float> SpeculativeDecodingState::Run(int total_length,
                                                DeviceSpan<int32_t>& next_tokens,
                                                DeviceSpan<int32_t> next_indices) {
  const int vocab_size = params_->config.model.vocab_size;
  auto draft_logits = draft_state_->Run(total_length, next_tokens, next_indices);
  auto cpu_draft = draft_logits.CopyDeviceToCpu();
  draft_pending_probs_ = Softmax({cpu_draft.data(), static_cast<size_t>(vocab_size)});
  draft_pending_valid_ = true;
  return target_state_->Run(total_length, next_tokens, next_indices);
}

void SpeculativeDecodingState::RewindTo(size_t index) {
  target_state_->RewindTo(index);
  draft_state_->RewindTo(index);
}

}  // namespace Generators
