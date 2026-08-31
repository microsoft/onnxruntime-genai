// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "request.h"

#include <cmath>

#include "engine.h"
#include "sequence_positions.h"
#include "../constrained_logits_processor.h"
#include "../search.h"

namespace Generators {

void TurnOptions::ValidateOwnerThread() const {
  auto bound_request = request.lock();
  if (!bound_request) {
    throw std::runtime_error(
        "Cannot use Turn options after their Request has been destroyed.");
  }
  bound_request->ValidateOwnerThread();
}

void Request::ValidateOwnerThread() const {
  if (IsClosed(status_)) {
    throw std::runtime_error("Cannot use a closed request.");
  }
  auto engine = engine_.lock();
  if (!engine) {
    throw std::runtime_error(
        "Cannot use a Request after its Engine has been destroyed.");
  }
  engine->ValidateOwnerThread();
}

Request::Request(
    std::shared_ptr<GeneratorParams> params,
    size_t max_total_tokens,
    std::shared_ptr<std::atomic<bool>> abandonment_pending)
    : max_total_tokens_{max_total_tokens},
      params_{params},
      rng_{CreateRandomGenerator(params->search.random_seed)},
      search_{CreateSearch(*params)},
      abandonment_pending_{std::move(abandonment_pending)} {
  // A request is one sequence: the engine batches requests, not rows within a request. Several
  // places here read row 0 only (UnprocessedTokens, CurrentSequenceLength) or take the tail of the
  // next-token span, so a wider search would silently mirror the wrong row's tokens.
  if (params->search.batch_size != 1) {
    throw std::runtime_error(
        "Engine requests require search.batch_size == 1; actual value is " +
        std::to_string(params->search.batch_size) +
        ". Batch across requests instead.");
  }
  // Beam search does not implement the deferred completion contract below, so its next tokens would
  // never be copied back from the device.
  if (params->search.num_beams != 1) {
    throw std::runtime_error(
        "Engine requests require search.num_beams == 1; actual value is " +
        std::to_string(params->search.num_beams) +
        ". Beam search is not supported by the Engine.");
  }
  if (params->guidance_ff_tokens_enabled) {
    throw std::runtime_error("Guidance fast-forward tokens are not supported by the engine.");
  }

  const bool has_guidance_type = !params->guidance_type.empty();
  const bool has_guidance_data = !params->guidance_data.empty();
  if (has_guidance_type != has_guidance_data) {
    throw std::runtime_error("Guidance type and data must be provided together.");
  }
  const bool guidance_requested = has_guidance_type && has_guidance_data;
  if (guidance_requested && !params->model_) {
    throw std::runtime_error("Engine guidance requires request parameters associated with a model.");
  }
  if (guidance_requested) {
    guidance_logits_processor_ = CreateGuidanceLogitsProcessor(*params->model_, params);
  }
  if (guidance_requested && !guidance_logits_processor_) {
    throw std::runtime_error("Engine guidance is unavailable. Build with use_guidance=true.");
  }

  // The engine drives one independent search per request, so completion is batched: see
  // ScheduledRequests::GenerateNextTokens().
  search_->DeferCompletion(true);
  tokens_host_.reserve(max_total_tokens_);
}

Request::~Request() = default;

void Request::OnFirstExternalReference() noexcept {
  externally_abandoned_.store(false, std::memory_order_release);
}

void Request::OnLastExternalReference() noexcept {
  externally_abandoned_.store(true, std::memory_order_release);
  abandonment_pending_->store(true, std::memory_order_release);
}

bool Request::IsExternallyAbandoned() const noexcept {
  return externally_abandoned_.load(std::memory_order_acquire);
}

void Request::Schedule() {
  if (status_ != RequestStatus::Assigned) {
    throw std::runtime_error("Request cannot be scheduled unless it has been previously added to the engine.");
  }

  if (!search_ || CurrentSequenceLength() == 0) {
    throw std::runtime_error("Cannot schedule a request with no tokens.");
  }

  status_ = RequestStatus::Active;
}

void Request::Close() {
  if (IsClosed(status_)) {
    return;
  }

  auto engine = engine_.lock();
  if (!engine) {
    CompleteClose();
    return;
  }
  engine->CloseRequest(shared_from_this());
}

bool Request::Cancel(uint64_t turn_id) {
  auto engine = engine_.lock();
  if (!engine) {
    throw std::runtime_error(
        "Cannot cancel after the request's engine has been destroyed.");
  }
  return engine->CancelRequest(shared_from_this(), turn_id);
}

void Request::CompleteClose() noexcept {
  engine_.reset();
  status_ = RequestStatus::Closed;
  guidance_transaction_checkpoint_.reset();
  guidance_logits_processor_.reset();
  batched_sampler_state_.reset();
  search_.reset();
  params_.reset();
  std::vector<int32_t>{}.swap(tokens_host_);
}

uint64_t Request::BeginTurn(
    std::span<const int32_t> tokens,
    std::optional<size_t> max_generated_tokens) {
  if (IsClosed(status_)) {
    throw std::runtime_error("Cannot begin a turn for a closed request.");
  }
  auto engine = engine_.lock();
  if (!engine) {
    throw std::runtime_error(
        "Cannot begin a turn after the request's engine has been destroyed.");
  }
  return engine->BeginTurn(
      shared_from_this(), tokens, max_generated_tokens);
}

int64_t Request::CurrentSequenceLength() const {
  return search_->GetSequenceLength();
}

RequestStateSnapshot Request::Snapshot() const {
  const int64_t current = CurrentSequenceLength();
  RequestStateSnapshot snapshot;
  snapshot.request_id = this;
  snapshot.status = status_;
  snapshot.current_sequence_length = current;
  snapshot.processed_sequence_length = processed_sequence_length_;
  snapshot.is_prefill = IsPrefill();
  snapshot.has_current_turn = has_current_turn_;
  snapshot.current_turn_id = current_turn_id_;
  snapshot.finish_reason = finish_reason_;
  return snapshot;
}

int64_t Request::ProcessedSequenceLength() const {
  return processed_sequence_length_;
}

size_t Request::ScheduledTokenCount() const {
  const size_t unprocessed = static_cast<size_t>(CurrentSequenceLength() - processed_sequence_length_);
  return std::min(scheduled_token_count_, unprocessed);
}

void Request::ScheduleTokens() {
  // A completed or logically closed static row can remain in the shared physical batch while a
  // peer executes. It contributes padding only and must not replay an unprocessed terminal token.
  if (!IsExecuting(status_)) {
    scheduled_token_count_ = 0;
    return;
  }
  const size_t unprocessed = static_cast<size_t>(CurrentSequenceLength() - processed_sequence_length_);
  scheduled_token_count_ = Generators::ScheduledTokenCount(unprocessed, params_->search.chunk_size);
}

void Request::BindScheduledTokenCount(size_t token_count) {
  const int64_t remaining = CurrentSequenceLength() - processed_sequence_length_;
  if (remaining <= 0 || token_count == 0 ||
      token_count > static_cast<size_t>(remaining)) {
    throw std::runtime_error(
        "The dynamic step token count (" + std::to_string(token_count) +
        ") must be positive and no greater than the remaining token count (" +
        std::to_string(remaining) + ").");
  }
  scheduled_token_count_ = token_count;
}

bool Request::IsChunkComplete() const {
  return processed_sequence_length_ + static_cast<int64_t>(ScheduledTokenCount()) >= CurrentSequenceLength();
}

void Request::AdvanceChunk() {
  processed_sequence_length_ += static_cast<int64_t>(ScheduledTokenCount());
}

DeviceSpan<int32_t> Request::UnprocessedTokens() {
  auto sequence = search_->GetSequence(0);
  return sequence.subspan(processed_sequence_length_, ScheduledTokenCount());
}

std::span<const int32_t> Request::UnprocessedTokensCpu() const {
  const size_t begin = static_cast<size_t>(processed_sequence_length_);
  const size_t end = begin + ScheduledTokenCount();
  if (end > tokens_host_.size())
    throw std::runtime_error("The host token mirror is out of sync with the search sequence.");

  return std::span<const int32_t>{tokens_host_}.subspan(begin, end - begin);
}

bool Request::IsTurnComplete() const {
  return status_ == RequestStatus::TurnComplete;
}

bool Request::IsPrefill() const {
  return processed_sequence_length_ < prompt_sequence_length_;
}

void Request::GenerateNextTokens(DeviceSpan<float> logits) {
  PrepareGeneration(logits);

  auto& search_params = search_->params_->search;
  if (!search_params.do_sample || search_params.top_k == 1 || search_params.temperature == 0) {
    search_->SelectTop();
  } else {
    // The user explicitly called TopKTopP on a beam search
    if (search_params.num_beams != 1)
      throw std::runtime_error("TopK and TopP cannot be used with a beam search");

    // Sanity checks
    if (!std::isfinite(search_params.top_p) ||
        search_params.top_p < 0.0f || search_params.top_p > 1.0f)
      throw std::runtime_error(
          "top_p (" + std::to_string(search_params.top_p) +
          ") must be finite and between 0.0 and 1.0.");
    if (search_params.top_k < 0)
      throw std::runtime_error(
          "top_k (" + std::to_string(search_params.top_k) +
          ") must be 0 or greater.");

    if (search_params.top_p > 0.0f && search_params.top_p < 1.0f && search_params.top_k > 1) {
      search_->SampleTopKTopP(search_params.top_k, search_params.top_p, search_params.temperature,
                              rng_);
    } else if (search_params.top_k > 1) {
      search_->SampleTopK(search_params.top_k, search_params.temperature, rng_);
    } else {
      assert(search_params.top_k == 0);
      search_->SampleTopP(search_params.top_p, search_params.temperature, rng_);
    }
  }
}

void Request::ValidateEngineCompatibility() const {
  const auto& search = params_->search;
  if (search.batch_size != 1 || search.num_beams != 1) {
    throw std::runtime_error("Engine requests require batch_size and num_beams to both be 1.");
  }
  if (!std::isfinite(search.top_p) ||
      search.top_p < 0.0f || search.top_p > 1.0f) {
    throw std::runtime_error(
        "top_p (" + std::to_string(search.top_p) +
        ") must be finite and between 0.0 and 1.0.");
  }
  if (search.top_k < 0) {
    throw std::runtime_error(
        "top_k (" + std::to_string(search.top_k) +
        ") must be 0 or greater.");
  }
}

void Request::SaveStateForTransaction() {
  auto guidance_checkpoint = guidance_logits_processor_
                                 ? guidance_logits_processor_->Clone()
                                 : nullptr;
  search_->SaveStateForTransaction();
  guidance_transaction_checkpoint_ = std::move(guidance_checkpoint);
  transaction_rng_ = rng_;
}

void Request::SaveStateForExternalSamplingTransaction() {
  auto guidance_checkpoint = guidance_logits_processor_
                                 ? guidance_logits_processor_->Clone()
                                 : nullptr;
  search_->SaveStateForExternalSamplingTransaction();
  guidance_transaction_checkpoint_ = std::move(guidance_checkpoint);
  transaction_rng_ = rng_;
}

RequestStepResult Request::ApplyLogitsForTransaction(DeviceSpan<float> logits) {
  const auto sequence_length_before = CurrentSequenceLength();
  PrepareGenerationForTransaction(logits);
  SelectNextToken();
  return StageGeneration(sequence_length_before);
}

void Request::PrepareGenerationForTransaction(DeviceSpan<float> logits) {
  ApplyLogitsProcessors(logits);
}

RequestStepResult Request::StageGenerationForTransaction(
    const RequestStepPlan& plan) {
  return StageGeneration(plan.sequence_length_before);
}

void Request::RestoreStateForTransaction() {
  search_->RestoreStateForTransaction();
  rng_ = transaction_rng_;
  if (guidance_transaction_checkpoint_) {
    guidance_logits_processor_ = std::move(guidance_transaction_checkpoint_);
  }
}

void Request::QueueStateRestoreForTransaction() {
  search_->QueueStateRestoreForTransaction();
  rng_ = transaction_rng_;
}

void Request::CompleteStateRestoreForTransaction() {
  search_->CompleteStateRestoreForTransaction();
  if (guidance_transaction_checkpoint_) {
    guidance_logits_processor_ = std::move(guidance_transaction_checkpoint_);
  }
}

void Request::CommitStateForTransaction() {
  search_->CommitStateForTransaction();
  guidance_transaction_checkpoint_.reset();
}

void Request::CommitStep(const RequestStepPlan& plan,
                         const RequestStepResult& result) noexcept {
  if (result.token_appended) {
    // tokens_host_ retains its existing max-length reservation, so this append cannot allocate at
    // the commit boundary.
    tokens_host_.push_back(result.token);
    ++turn_generated_tokens_;
  }
  processed_sequence_length_ = static_cast<int64_t>(plan.target_cache_slots);
  status_ = result.done ? RequestStatus::TurnComplete : RequestStatus::Active;
  if (result.done) {
    finish_reason_ = result.finish_reason;
  }
}

void Request::ApplyLogitsProcessors(DeviceSpan<float> logits) {
  search_->SetLogits(logits);
  if (guidance_logits_processor_) {
    guidance_logits_processor_->ProcessLogits(logits);
  }
  auto& search_params = search_->params_->search;
  search_->ApplyMinLength(search_params.min_length);
  search_->ApplyRepetitionPenalty(search_params.repetition_penalty);
  search_->ApplyNoRepeatNgram(search_params.no_repeat_ngram_size);
}

void Request::ResetGuidanceForNewTurn() {
  if (guidance_logits_processor_) {
    guidance_logits_processor_->Reset();
  }
}

void Request::SelectNextToken() {
  auto& search_params = search_->params_->search;
  if (!search_params.do_sample || search_params.top_k == 1 || search_params.temperature == 0) {
    search_->SelectTop();
  } else if (search_params.top_p > 0.0f && search_params.top_p < 1.0f &&
             search_params.top_k > 1) {
    search_->SampleTopKTopP(search_params.top_k, search_params.top_p,
                            search_params.temperature, rng_);
  } else if (search_params.top_k > 1) {
    search_->SampleTopK(search_params.top_k, search_params.temperature, rng_);
  } else {
    search_->SampleTopP(search_params.top_p, search_params.temperature, rng_);
  }
}

RequestStepResult Request::StageGeneration(int64_t sequence_length_before) {
  search_->CompleteGeneration();
  const bool search_done = search_->IsDone();
  const bool token_appended = CurrentSequenceLength() > sequence_length_before;
  const auto next_tokens = search_->GetNextTokens().CpuSpan();
  const int32_t token = next_tokens.empty() ? 0 : next_tokens.back();
  const size_t generated_tokens_after_step =
      turn_generated_tokens_ + static_cast<size_t>(token_appended);
  const bool turn_limit_reached =
      turn_max_generated_tokens_ &&
      generated_tokens_after_step >= *turn_max_generated_tokens_;
  const bool context_limit_reached =
      static_cast<size_t>(CurrentSequenceLength()) >= max_total_tokens_;
  GenerationFinishReason finish_reason = GenerationFinishReason::None;
  if (search_done && !next_tokens.empty() &&
      contains(params_->config.model.eos_token_id, token)) {
    finish_reason = GenerationFinishReason::EosToken;
  } else if (turn_limit_reached) {
    finish_reason = GenerationFinishReason::TurnLimit;
  } else if (context_limit_reached || search_done) {
    finish_reason = GenerationFinishReason::ContextLimit;
  }
  const bool done = finish_reason != GenerationFinishReason::None;
  RequestStepResult result{
      token,
      token_appended,
      done,
      finish_reason,
  };
  CommitGuidanceToken(result);
  if (done && guidance_logits_processor_) {
    guidance_logits_processor_->Reset();
  }
  return result;
}

void Request::CommitGuidanceToken(const RequestStepResult& result) {
  if (guidance_logits_processor_ && result.token_appended) {
    int32_t token = result.token;
    guidance_logits_processor_->CommitTokens(std::span<int32_t>{&token, 1});
  }
}

void Request::PrepareGeneration(DeviceSpan<float> logits) {
  processed_sequence_length_ = search_->GetSequence(0).size();
  ApplyLogitsProcessors(logits);
}

const Config::Search& Request::SearchOptions() const {
  return search_->params_->search;
}

bool Request::BindNextTokensSlot(DeviceSpan<int32_t> slot) {
  return search_->BindNextTokensSlot(slot);
}

bool Request::SupportsBatchedSampling() const {
  return search_->SupportsBatchedSampling();
}

void Request::OnNextTokensSampled() {
  search_->OnNextTokensSampled();
}

BatchedSamplerState& Request::SamplingState(BatchedSampler& sampler) {
  if (!batched_sampler_state_ || !sampler.OwnsState(*batched_sampler_state_))
    batched_sampler_state_ = sampler.CreateState(search_->params_->search.random_seed);
  return *batched_sampler_state_;
}

RequestStepResult Request::CompleteGeneration() {
  search_->CompleteGeneration();
  const auto next_tokens = search_->GetNextTokens().CpuSpan();

  const size_t sequence_length = static_cast<size_t>(CurrentSequenceLength());
  size_t new_token_count{};
  int32_t token{};
  if (sequence_length > tokens_host_.size()) {
    new_token_count = sequence_length - tokens_host_.size();
    if (new_token_count > next_tokens.size())
      throw std::runtime_error("The search produced fewer tokens than it appended to the sequence.");
    auto new_tokens = next_tokens.last(new_token_count);

    tokens_host_.insert(tokens_host_.end(), new_tokens.begin(), new_tokens.end());
    token = new_tokens.back();
    if (guidance_logits_processor_) {
      guidance_logits_processor_->CommitTokens(new_tokens);
    }
    turn_generated_tokens_ += new_token_count;
  }

  const bool turn_limit_reached =
      turn_max_generated_tokens_ &&
      turn_generated_tokens_ >= *turn_max_generated_tokens_;
  const bool context_limit_reached =
      static_cast<size_t>(CurrentSequenceLength()) >= max_total_tokens_;
  if (search_->IsDone() || turn_limit_reached || context_limit_reached) {
    status_ = RequestStatus::TurnComplete;
    if (search_->IsDone() && !next_tokens.empty() &&
        contains(params_->config.model.eos_token_id, next_tokens.back())) {
      finish_reason_ = GenerationFinishReason::EosToken;
    } else if (turn_limit_reached) {
      finish_reason_ = GenerationFinishReason::TurnLimit;
    } else {
      finish_reason_ = GenerationFinishReason::ContextLimit;
    }
    if (guidance_logits_processor_) {
      guidance_logits_processor_->Reset();
    }
  }
  return RequestStepResult{
      token,
      new_token_count != 0,
      Generators::IsTurnComplete(status_),
      finish_reason_,
  };
}

}  // namespace Generators
