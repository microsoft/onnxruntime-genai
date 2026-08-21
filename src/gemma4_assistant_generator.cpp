// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "gemma4_assistant_generator.h"

#include <algorithm>
#include <array>
#include <stdexcept>

#include "generators.h"
#include "models/decoder_only.h"
#include "models/kv_cache.h"
#include "models/model.h"
#include "models/multi_modal.h"
#include "models/utils.h"

namespace Generators {
namespace {

void RequireInputs(const Model& model, const std::vector<std::string>& names, const char* role) {
  for (const auto& name : names)
    if (name.empty() || !model.session_info_.HasInput(name))
      throw std::runtime_error(std::string{"Gemma4AssistantGenerator: missing "} + role + " input '" + name + "'");
}

void RequireOutputs(const Model& model, const std::vector<std::string>& names, const char* role) {
  for (const auto& name : names)
    if (name.empty() || !model.session_info_.HasOutput(name))
      throw std::runtime_error(std::string{"Gemma4AssistantGenerator: missing "} + role + " output '" + name + "'");
}

std::vector<const char*> AsCStrings(const std::vector<std::string>& names) {
  std::vector<const char*> result;
  result.reserve(names.size());
  for (const auto& name : names) result.push_back(name.c_str());
  return result;
}

template <typename T>
const T& RequireModel(const Model& model, const char* role) {
  const auto* typed = dynamic_cast<const T*>(&model);
  if (!typed)
    throw std::runtime_error(std::string{"Gemma4AssistantGenerator: "} + role + " model of type '" +
                             model.config_->model.type + "' is not compatible");
  return *typed;
}

}  // namespace

Gemma4AssistantGenerator::Gemma4AssistantGenerator(
    const Model& target_model, const Model& assistant_model, const GeneratorParams& params)
    : target_model_{RequireModel<MultiModalLanguageModel>(target_model, "target")},
      assistant_model_{RequireModel<DecoderOnly_Model>(assistant_model, "assistant")},
      assistant_run_options_{OrtRunOptions::Create()},
      embedding_run_options_{OrtRunOptions::Create()} {
  ValidateMtpPair(target_model_, assistant_model_, params);
  if (params.search.do_sample && params.search.temperature > 0.0f)
    throw std::runtime_error("Gemma4AssistantGenerator currently supports greedy generation only");

  const auto& mtp = target_model_.config_->model.mtp;
  const auto& target_decoder = target_model_.config_->model.decoder;
  hidden_size_ = target_decoder.hidden_size;
  if (hidden_size_ <= 0)
    throw std::runtime_error("Gemma4AssistantGenerator requires model.decoder.hidden_size in the target config");
  if (mtp.shared_kv_layers.empty())
    throw std::runtime_error("Gemma4AssistantGenerator requires model.mtp.shared_kv_layers in the target config");
  if (mtp.inputs.shared_key_names.size() != mtp.shared_kv_layers.size() ||
      mtp.inputs.shared_value_names.size() != mtp.shared_kv_layers.size())
    throw std::runtime_error(
        "Gemma4AssistantGenerator requires model.mtp.shared_kv_layers, shared_key_names and shared_value_names to be the same length");

  target_embeddings_name_ = mtp.main_inputs_embeds;
  target_hidden_states_name_ = mtp.main_hidden_states;
  target_attention_mask_name_ = target_decoder.inputs.attention_mask;

  assistant_input_name_storage_ = {mtp.inputs.hidden_states, mtp.inputs.attention_mask};
  for (size_t index = 0; index < mtp.shared_kv_layers.size(); ++index) {
    assistant_input_name_storage_.push_back(mtp.inputs.shared_key_names[index]);
    assistant_input_name_storage_.push_back(mtp.inputs.shared_value_names[index]);
    shared_kv_target_names_.push_back(
        ComposeKeyValueName(target_decoder.outputs.present_key_names, mtp.shared_kv_layers[index]));
    shared_kv_target_names_.push_back(
        ComposeKeyValueName(target_decoder.outputs.present_value_names, mtp.shared_kv_layers[index]));
  }
  assistant_output_name_storage_ = {mtp.outputs.logits, mtp.outputs.hidden_states};

  std::vector<std::string> target_outputs{target_embeddings_name_, target_hidden_states_name_};
  target_outputs.insert(target_outputs.end(), shared_kv_target_names_.begin(), shared_kv_target_names_.end());
  RequireOutputs(target_model_, target_outputs, "target");
  RequireInputs(assistant_model_, assistant_input_name_storage_, "assistant");
  RequireOutputs(assistant_model_, assistant_output_name_storage_, "assistant");

  assistant_input_names_ = AsCStrings(assistant_input_name_storage_);
  assistant_output_names_ = AsCStrings(assistant_output_name_storage_);
  assistant_inputs_.resize(assistant_input_names_.size());
  assistant_outputs_.resize(assistant_output_names_.size());

  target_params_ = std::make_shared<GeneratorParams>(target_model_);
  target_params_->search = params.search;
  target_params_->max_batch_size = params.max_batch_size;
  target_params_->use_graph_capture = params.use_graph_capture;
  num_speculative_tokens_ = std::max(1, params.speculative.max_draft_tokens);
  target_params_->max_graph_capture_length = num_speculative_tokens_;
  target_ = CreateGenerator(target_model_, *target_params_);

  vocab_size_ = target_model_.config_->model.vocab_size;
  max_length_ = params.search.max_length;
  eos_token_ids_ = target_model_.config_->model.eos_token_id;
  drafts_.resize(num_speculative_tokens_);
  verify_argmax_.resize(num_speculative_tokens_);

  const auto embedding_type = target_model_.session_info_.GetOutputDataType(target_embeddings_name_);
  const auto hidden_type = target_model_.session_info_.GetOutputDataType(target_hidden_states_name_);
  const auto assistant_input_type = assistant_model_.session_info_.GetInputDataType(mtp.inputs.hidden_states);
  if (embedding_type != hidden_type || hidden_type != assistant_input_type)
    throw std::runtime_error("Gemma4AssistantGenerator requires matching embedding and hidden-state types");

  const int64_t hidden = hidden_size_;
  current_embedding_ = std::make_shared<Tensor>(target_model_.p_device_inputs_, embedding_type);
  current_embedding_->CreateTensor(std::array<int64_t, 3>{1, 1, hidden});
  carried_hidden_ = std::make_shared<Tensor>(target_model_.p_device_inputs_, hidden_type);
  carried_hidden_->CreateTensor(std::array<int64_t, 3>{1, 1, hidden});
  // The head consumes the token embedding concatenated with the carried hidden state.
  assistant_input_ = std::make_shared<Tensor>(assistant_model_.p_device_inputs_, assistant_input_type);
  assistant_input_->CreateTensor(std::array<int64_t, 3>{1, 1, 2 * hidden});
  assistant_logits_ = std::make_shared<Tensor>(
      assistant_model_.p_device_logits_, assistant_model_.session_info_.GetOutputDataType(mtp.outputs.logits));
  assistant_logits_->CreateTensor(std::array<int64_t, 3>{1, 1, vocab_size_});
  assistant_projected_ = std::make_shared<Tensor>(
      assistant_model_.p_device_inputs_, assistant_model_.session_info_.GetOutputDataType(mtp.outputs.hidden_states));
  assistant_projected_->CreateTensor(std::array<int64_t, 3>{1, 1, hidden});

  const auto token_type = target_model_.session_info_.GetInputDataType(
      target_model_.config_->model.embedding.inputs.input_ids);
  embedding_token_ = OrtValue::CreateTensor(
      target_model_.p_device_inputs_->GetAllocator(), std::array<int64_t, 2>{1, 1}, token_type);
}

void Gemma4AssistantGenerator::CaptureTargetState(int row) {
  CopyTensorRow(*ResolveTargetOutput(target_embeddings_name_), row, *current_embedding_,
                *target_model_.p_device_);
  CopyTensorRow(*ResolveTargetOutput(target_hidden_states_name_), 0, *carried_hidden_,
                *target_model_.p_device_);
}

// A name can be a real graph output yet never be bound by the runtime, in which case GetOutput
// returns null; report that rather than dereferencing it.
OrtValue* Gemma4AssistantGenerator::ResolveTargetOutput(const std::string& name) const {
  OrtValue* value = target_->state_->GetOutput(name.c_str());
  if (!value)
    throw std::runtime_error("Gemma4AssistantGenerator: target output '" + name +
                             "' is not bound by the target model's state");
  return value;
}

void Gemma4AssistantGenerator::UpdateTargetPrediction(int row, bool capture_state) {
  auto logits = target_->GetLogits();
  if (!target_model_.p_device_->ArgMax(logits.Span().data(), Ort::TypeToTensorType<float>, 1,
                                       vocab_size_, &target_next_)) {
    auto host = logits.CopyDeviceToCpu();
    target_next_ = ArgmaxRow(host.data(), vocab_size_);
  }
  if (capture_state) CaptureTargetState(row);
}

void Gemma4AssistantGenerator::SynchronizeTarget() {
  const int32_t token = sequence_.back();
  target_->AppendTokens(cpu_span<const int32_t>(&token, 1));
  stats_.target_forward_passes++;
  UpdateTargetPrediction(0);
  target_sync_pending_ = false;
}

void Gemma4AssistantGenerator::EmbedToken(int32_t token) {
  const auto type = embedding_token_->GetTensorTypeAndShapeInfo()->GetElementType();
  if (type == Ort::TypeToTensorType<int64_t>)
    *embedding_token_->GetTensorMutableData<int64_t>() = token;
  else if (type == Ort::TypeToTensorType<int32_t>)
    *embedding_token_->GetTensorMutableData<int32_t>() = token;
  else
    throw std::runtime_error("Gemma4AssistantGenerator: embedding token input must be int32 or int64");

  const std::array<const char*, 3> names{
      target_model_.config_->model.embedding.inputs.input_ids.c_str(),
      target_model_.config_->model.embedding.inputs.image_features.c_str(),
      target_model_.config_->model.embedding.inputs.audio_features.c_str()};
  const std::array<const OrtValue*, 3> inputs{
      embedding_token_.get(), target_->state_->GetInput(names[1]), target_->state_->GetInput(names[2])};
  const char* output_name = target_model_.config_->model.embedding.outputs.embeddings.c_str();
  OrtValue* output = current_embedding_->GetOrtTensor();
  target_model_.embedding_session_->Run(embedding_run_options_.get(), names.data(), inputs.data(),
                                        inputs.size(), &output_name, &output, 1);
}

int32_t Gemma4AssistantGenerator::ArgmaxAssistant() {
  int32_t token{};
  auto* logits = assistant_logits_->GetOrtTensor();
  const auto type = logits->GetTensorTypeAndShapeInfo()->GetElementType();
  if (assistant_model_.p_device_->ArgMax(logits->GetTensorRawData(), type, 1, vocab_size_, &token))
    return token;
  auto cpu = assistant_logits_->GetByteSpan().CopyDeviceToCpu();
  return ArgmaxRow(cpu.data(), type, vocab_size_);
}

int32_t Gemma4AssistantGenerator::Draft() {
  auto input = assistant_input_->GetByteSpan();
  const size_t half = input.size() / 2;
  input.subspan(0, half).CopyFrom(current_embedding_->GetByteSpan());
  input.subspan(half, half).CopyFrom(carried_hidden_->GetByteSpan());
  assistant_inputs_[0] = assistant_input_->GetOrtTensor();
  assistant_inputs_[1] = target_->state_->GetInput(target_attention_mask_name_.c_str());
  for (size_t index = 0; index < shared_kv_target_names_.size(); ++index)
    assistant_inputs_[2 + index] = ResolveTargetOutput(shared_kv_target_names_[index]);
  assistant_outputs_[0] = assistant_logits_->GetOrtTensor();
  assistant_outputs_[1] = assistant_projected_->GetOrtTensor();
  assistant_model_.session_decoder_->Run(
      assistant_run_options_.get(), assistant_input_names_.data(), assistant_inputs_.data(), assistant_inputs_.size(),
      assistant_output_names_.data(), assistant_outputs_.data(), assistant_outputs_.size());
  stats_.draft_forward_passes++;
  const int32_t draft = ArgmaxAssistant();
  carried_hidden_->GetByteSpan().CopyFrom(assistant_projected_->GetByteSpan());
  EmbedToken(draft);
  return draft;
}

void Gemma4AssistantGenerator::ArgmaxTargetRows(int first_row, int count, int32_t* output) {
  auto& pipeline = dynamic_cast<MultiModalPipelineState&>(*target_->state_);
  auto cpu = pipeline.GetAllLogits().CopyDeviceToCpu();
  for (int row = 0; row < count; ++row)
    output[row] = ArgmaxRow(
        cpu.data() + (static_cast<size_t>(first_row + row) * vocab_size_), vocab_size_);
}

void Gemma4AssistantGenerator::AppendTokens(cpu_span<const int32_t> input_ids) {
  if (primed_) throw std::runtime_error("Gemma4AssistantGenerator: AppendTokens can only be called once");
  primed_ = true;
  target_->AppendTokens(input_ids);
  sequence_.assign(input_ids.begin(), input_ids.end());
  emitted_sequence_ = sequence_;
  length_ = sequence_.size();
  if (length_ >= static_cast<size_t>(max_length_)) {
    done_ = true;
    return;
  }
  UpdateTargetPrediction(static_cast<int>(input_ids.size()) - 1, false);

  // The prefill already predicted the first generated token, so commit it here; the target only
  // consumes it at the start of the first round.
  sequence_.push_back(target_next_);
  length_++;
  QueueCommittedTokens(length_ - 1);
  target_sync_pending_ = !done_;
}

void Gemma4AssistantGenerator::RunRound() {
  if (target_sync_pending_) SynchronizeTarget();
  const size_t remaining = length_ < static_cast<size_t>(max_length_) ? static_cast<size_t>(max_length_) - length_ : 0;
  const int count = static_cast<int>(std::min(static_cast<size_t>(num_speculative_tokens_), remaining));
  if (count <= 0) {
    done_ = true;
    return;
  }
  stats_.rounds++;
  stats_.completed_rounds++;
  stats_.draft_tokens_proposed += static_cast<size_t>(count);
  for (int index = 0; index < count; ++index) {
    try {
      drafts_[index] = Draft();
    } catch (const std::exception& error) {
      throw std::runtime_error(
          "Gemma4AssistantGenerator: assistant draft failed: " + std::string{error.what()});
    }
  }

  target_->SnapshotState();
  try {
    target_->AppendTokens(cpu_span<const int32_t>(drafts_.data(), count));
    stats_.target_forward_passes++;
    ArgmaxTargetRows(0, count, verify_argmax_.data());
  } catch (const std::exception& error) {
    throw std::runtime_error(
        "Gemma4AssistantGenerator: target verification failed: " + std::string{error.what()});
  }
  const int accepted = CountAcceptedDrafts(drafts_.data(), verify_argmax_.data(), count, target_next_);
  stats_.draft_tokens_evaluated += static_cast<size_t>(accepted < count ? accepted + 1 : count);
  stats_.draft_tokens_accepted += static_cast<size_t>(accepted);

  if (accepted == count) {
    stats_.bonus_tokens++;
    sequence_.insert(sequence_.end(), drafts_.begin(), drafts_.begin() + accepted);
    length_ += static_cast<size_t>(accepted);
    target_next_ = verify_argmax_[count - 1];
    CaptureTargetState(count - 1);
  } else {
    stats_.correction_tokens++;
    const int32_t correction = accepted == 0 ? target_next_ : verify_argmax_[accepted - 1];
    target_->RewindToLength(length_);
    std::vector<int32_t> committed(drafts_.begin(), drafts_.begin() + accepted);
    committed.push_back(correction);
    target_->AppendTokens(cpu_span<const int32_t>(committed.data(), committed.size()));
    stats_.target_forward_passes++;
    sequence_.insert(sequence_.end(), committed.begin(), committed.end());
    length_ += committed.size();
    UpdateTargetPrediction(static_cast<int>(committed.size()) - 1);
  }
}

}  // namespace Generators
