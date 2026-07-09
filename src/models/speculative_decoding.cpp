// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#include <algorithm>
#include "../generators.h"
#include "../softmax.h"
#include "../speculative_sampling.h"
#include "speculative_decoding.h"
#include "model_type.h"

namespace Generators {

namespace {

std::unique_ptr<Config> CloneConfigForDraft(const Config& source) {
  auto config = std::make_unique<Config>(source);
  config->model.decoder = source.model.draft;
  return config;
}

std::unique_ptr<Config> CloneConfigForTarget(const Config& source) {
  return std::make_unique<Config>(source);
}

bool ProviderConfigurationMatches(const Config::SessionOptions& target_options,
                                  const Config::SessionOptions& draft_options) {
  if (target_options.providers != draft_options.providers) return false;

  if (target_options.provider_options.size() != draft_options.provider_options.size()) return false;
  for (size_t i = 0; i < target_options.provider_options.size(); ++i) {
    const auto& target_provider = target_options.provider_options[i];
    const auto& draft_provider = draft_options.provider_options[i];
    if (target_provider.name != draft_provider.name) return false;
    if (target_provider.options != draft_provider.options) return false;
    if (target_provider.device_filtering_options.has_value() != draft_provider.device_filtering_options.has_value()) return false;
    if (target_provider.device_filtering_options) {
      const auto& target_device_filter = *target_provider.device_filtering_options;
      const auto& draft_device_filter = *draft_provider.device_filtering_options;
      if (target_device_filter.hardware_device_type != draft_device_filter.hardware_device_type ||
          target_device_filter.hardware_device_id != draft_device_filter.hardware_device_id ||
          target_device_filter.hardware_vendor_id != draft_device_filter.hardware_vendor_id)
        return false;
    }
  }

  return true;
}

void ValidateLogitsDimensionsMatch(const DecoderOnly_Model& target,
                                   const DecoderOnly_Model& draft) {
  const auto& target_logits_name = target.config_->model.decoder.outputs.logits;
  const auto& draft_logits_name = draft.config_->model.decoder.outputs.logits;
  if (!target.session_info_.HasOutput(target_logits_name) ||
      !draft.session_info_.HasOutput(draft_logits_name))
    return;
  const auto target_logits_shape = target.session_info_.GetOutputShape(target_logits_name);
  const auto draft_logits_shape = draft.session_info_.GetOutputShape(draft_logits_name);
  if (target_logits_shape.empty() || draft_logits_shape.empty())
    return;
  int64_t target_vocab_size = target_logits_shape.back();
  int64_t draft_vocab_size = draft_logits_shape.back();
  if (target_vocab_size <= 0 || draft_vocab_size <= 0)
    return;
  if (target_vocab_size != draft_vocab_size)
    throw std::runtime_error(
        "Target and draft logit dimensions don't match. Target vocab: " +
        std::to_string(target_vocab_size) + ", Draft vocab: " + std::to_string(draft_vocab_size) +
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

  // The inner states are constructed directly as DecoderOnly_Model, which requires the modern
  // separate key/value KV-cache format (past_key_names/past_value_names). 
  auto uses_combined_kv = [](const Config::Model::Decoder& d) {
    return !d.inputs.past_names.empty() || !d.outputs.present_names.empty();
  };
  if (uses_combined_kv(config_->model.decoder) || uses_combined_kv(config_->model.draft))
    throw std::runtime_error(
        "Speculative decoding requires decoder-only target and draft models that use the separate "
        "key/value KV-cache format (past_key_names/past_value_names). Combined-KV / legacy formats "
        "such as the original gpt2 graph (past_%d/present_%d) are not supported in this release.");

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
  draft_pending_logits_.assign(cpu_draft.data(), cpu_draft.data() + vocab_size);
  draft_pending_valid_ = true;
  return target_state_->Run(total_length, next_tokens, next_indices);
}

void SpeculativeDecodingState::RewindTo(size_t index) {
  target_state_->RewindTo(index);
  draft_state_->RewindTo(index);
  // draft_pending_logits_ is stale after a rewind and must not seed the next proposal. 
  // Invalidate it -> refresh it. Check draft_pending_valid_ and throws if it is 
  // ever consumed while stale.
  draft_pending_valid_ = false;
}

}  // namespace Generators
