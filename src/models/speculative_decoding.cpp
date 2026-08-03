// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#include <algorithm>
#include "../generators.h"
#include "../softmax.h"
#include "../speculative_sampling.h"
#include "speculative_decoding.h"
#include "kv_cache.h"
#include "model_type.h"

namespace Generators {

namespace {

std::unique_ptr<Config> CloneConfigForDraft(const Config& source,
                                            const Config::Model::Decoder& draft) {
  auto config = std::make_unique<Config>(source);
  config->model.decoder = draft;
  return config;
}

std::unique_ptr<Config> CloneConfigForTarget(const Config& source) {
  return std::make_unique<Config>(source);
}

}  // namespace

bool ProviderConfigurationMatches(const Config::SessionOptions& target_options,
                                  const Config::SessionOptions& proposer_options) {
  if (target_options.providers != proposer_options.providers) return false;

  if (target_options.provider_options.size() != proposer_options.provider_options.size()) return false;
  for (size_t i = 0; i < target_options.provider_options.size(); ++i) {
    const auto& target_provider = target_options.provider_options[i];
    const auto& proposer_provider = proposer_options.provider_options[i];
    if (target_provider.name != proposer_provider.name) return false;
    if (target_provider.options != proposer_provider.options) return false;
    if (target_provider.device_filtering_options.has_value() != proposer_provider.device_filtering_options.has_value()) return false;
    if (target_provider.device_filtering_options) {
      const auto& target_device_filter = *target_provider.device_filtering_options;
      const auto& proposer_device_filter = *proposer_provider.device_filtering_options;
      if (target_device_filter.hardware_device_type != proposer_device_filter.hardware_device_type ||
          target_device_filter.hardware_device_id != proposer_device_filter.hardware_device_id ||
          target_device_filter.hardware_vendor_id != proposer_device_filter.hardware_vendor_id)
        return false;
    }
  }

  return true;
}

void ValidateSpeculativeModelCompatibility(const Model& model,
                                          const Config::Model::Decoder* proposer) {
  const auto& config = *model.config_;
  if (!config.model.decoder.pipeline.empty() ||
      (proposer && !proposer->pipeline.empty()))
    throw std::runtime_error(
        "Speculative decoding does not support pipeline models in this release; "
        "target and proposer must be plain decoder-only LLMs.");
  if (!config.model.vision.filename.empty() || !config.model.speech.filename.empty())
    throw std::runtime_error(
        "Speculative decoding does not support multimodal (vision/audio) models in this release; "
        "target and proposer must be plain decoder-only LLMs.");

  if (UsesNonRewindableWindowedKeyValueCache(model, config.model.decoder) ||
      (proposer && UsesNonRewindableWindowedKeyValueCache(model, *proposer)))
    throw std::runtime_error(
        "Speculative decoding does not support physically sliding KV caches in this release; "
        "WindowedKeyValueCache cannot rewind after discarding KV history.");
  if (!config.model.decoder.layer_types.empty() ||
      (proposer && !proposer->layer_types.empty()))
    throw std::runtime_error(
        "Speculative decoding does not support LFM2 (hybrid SSM/attention) models in this release; "
        "their rolling convolution state cannot be rewound.");
}

void ValidateSpeculativeGeneratorParams(const GeneratorParams& params) {
  if (params.search.batch_size != 1)
    throw std::runtime_error(
        "Speculative decoding does not support batch_size > 1 in this release. Got batch_size=" +
        std::to_string(params.search.batch_size));
  if (params.search.num_beams != 1)
    throw std::runtime_error(
        "Speculative decoding does not support num_beams > 1 (beam search). Got num_beams=" +
        std::to_string(params.search.num_beams) + ".");
  if (params.search.no_repeat_ngram_size != 0)
    throw std::runtime_error(
        "Speculative decoding does not support no_repeat_ngram_size != 0 in this release. Got "
        "no_repeat_ngram_size=" +
        std::to_string(params.search.no_repeat_ngram_size) + ".");
}

namespace {

int64_t GetLogitsVocabSize(const DecoderOnly_Model& model, const char* model_role) {
  const auto& logits_name = model.config_->model.decoder.outputs.logits;
  if (!model.session_info_.HasOutput(logits_name))
    throw std::runtime_error(
        std::string(model_role) + " logits output '" + logits_name +
        "' was not found in the ONNX model.");

  const auto logits_shape = model.session_info_.GetOutputShape(logits_name);
  if (logits_shape.empty())
    throw std::runtime_error(
        std::string(model_role) + " logits output '" + logits_name +
        "' must have at least one dimension.");

  const int64_t vocab_size = logits_shape.back();
  if (vocab_size <= 0)
    throw std::runtime_error(
        std::string(model_role) + " logits output '" + logits_name +
        "' must have a static positive vocabulary dimension.");

  return vocab_size;
}

void ValidateLogitsDimensionsMatch(const DecoderOnly_Model& target,
                                   const DecoderOnly_Model& draft) {
  const int64_t target_vocab_size = GetLogitsVocabSize(target, "Target");
  const int64_t draft_vocab_size = GetLogitsVocabSize(draft, "Draft");
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
  if (!config_->model.draft || config_->model.draft->filename.empty())
    throw std::runtime_error(
        "model.draft.filename is not set in genai_config.json.");

  const auto& draft_config = *config_->model.draft;
  if (!ProviderConfigurationMatches(config_->model.decoder.session_options,
                                    draft_config.session_options))
    throw std::runtime_error(
        "Target and draft must use the same execution provider. "
        "Cross-EP speculative decoding is not supported in this release.");

  ValidateSpeculativeModelCompatibility(*this, &draft_config);

  target_model_ = std::make_shared<DecoderOnly_Model>(CloneConfigForTarget(*config_), ort_env);
  draft_model_ = std::make_shared<DecoderOnly_Model>(
      CloneConfigForDraft(*config_, draft_config), ort_env);
  ValidateLogitsDimensionsMatch(*target_model_, *draft_model_);
  session_info_.Add(*target_model_->session_decoder_);
}

std::unique_ptr<State> SpeculativeDecodingModel::CreateState(DeviceSpan<int32_t> sequence_lengths,
                                                             const GeneratorParams& params) const {
  return std::make_unique<SpeculativeDecodingState>(*this, sequence_lengths, params);
}

SpeculativeDecodingState::SpeculativeDecodingState(const SpeculativeDecodingModel& model,
                                                   DeviceSpan<int32_t> sequence_lengths,
                                                   const GeneratorParams& params)
    : State{params, model},
      model_{model},
      target_state_{model.target_model().CreateState(sequence_lengths, params)},
      draft_state_{model.draft_model().CreateState(sequence_lengths, params)} {
  ValidateSpeculativeGeneratorParams(params);
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

OrtValue* SpeculativeDecodingState::GetInput(const char* name) {
  if (auto* input = target_state_->GetInput(name))
    return input;
  return draft_state_->GetInput(name);
}

OrtValue* SpeculativeDecodingState::GetOutput(const char* name) {
  if (auto* output = target_state_->GetOutput(name))
    return output;
  return draft_state_->GetOutput(name);
}

void SpeculativeDecodingState::SetActiveAdapter(Adapters* adapters, const std::string& adapter_name) {
  // Apply the adapter only to the target; the draft may be a different, incompatible model.
  target_state_->SetActiveAdapter(adapters, adapter_name);
}

void SpeculativeDecodingState::SetRunOption(const char* key, const char* value) {
  if (key == nullptr || value == nullptr)
    throw std::runtime_error("Speculative decoding runtime option key and value must not be null.");

  const bool terminate = std::strcmp(key, "terminate_session") == 0;
  if (terminate && std::strcmp(value, "1") == 0)
    State::SetRunOption(key, value);

  target_state_->SetRunOption(key, value);
  draft_state_->SetRunOption(key, value);

  if (terminate && std::strcmp(value, "0") == 0)
    State::SetRunOption(key, value);
}

void SpeculativeDecodingState::SetExtraInputs(const std::vector<ExtraInput>& extra_inputs) {
  const auto target_names = model_.target_model().session_info_.GetInputNames();
  const auto draft_names = model_.draft_model().session_info_.GetInputNames();
  for (const auto& extra_input : extra_inputs) {
    const bool target_has_input = std::find(target_names.begin(), target_names.end(), extra_input.name) != target_names.end();
    const bool draft_has_input = std::find(draft_names.begin(), draft_names.end(), extra_input.name) != draft_names.end();
    if (!target_has_input && !draft_has_input)
      throw std::runtime_error("Speculative decoding model input '" + extra_input.name + "' was not found in the target or draft model.");
  }

  target_state_->SetExtraInputs(extra_inputs);
  draft_state_->SetExtraInputs(extra_inputs);
}

}  // namespace Generators
